/*
 * Contains all the CUDA kernels for the 3D ETG model of Adkins et al. (2023).
 */

// A lot of basic functionality is already implemented here.
#include "flucs/solvers/fourier/fourier_system.cuh"

template <typename T_output, typename Functor, typename... InputArgs>
__global__ void spectral_pointwise(
    T_output* __restrict__ output,
    InputArgs... input_args
) {
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    if (index < HALFSIZE)
        output[index] = Functor{input_args...}(index);
}

extern "C" {

__device__ void get_linear_matrix(
    const size_t index, 
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step, 
    FLUCS_COMPLEX matrix[NUMBER_OF_FIELDS][NUMBER_OF_FIELDS]
){
    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);
    const FLUCS_FLOAT kz = kz_from_ikz(ikz);

    // Generate the linear matrix
    matrix[0][0] = FLUCS_COMPLEX(
        COEFFA * (1 + TAUBAR) * kz * kz,
        (2 * (1 + TAUBAR) * KAPPAB - TAUBAR * KAPPAN) * ky
    );

    matrix[0][1] = FLUCS_COMPLEX(
        -TAUBAR * (COEFFA + COEFFB) * kz * kz,
        -2 * TAUBAR * KAPPAB * ky
    );

    matrix[1][0] = FLUCS_COMPLEX(
        -(2.0/3) * (COEFFA + COEFFB) * (1 + 1/TAUBAR) * kz * kz,
        (KAPPAT - (4.0/3) * (1 + 1/TAUBAR) * KAPPAB) * ky
    );

    matrix[1][1] = FLUCS_COMPLEX(
        (2.0/3) * (
            COEFFC + COEFFA*(1 + COEFFB/COEFFA)*(1 + COEFFB/COEFFA)
        ) * kz * kz,
        (14.0/3) * KAPPAB * ky
    );
}

__global__ void find_derivatives(
    const FLUCS_FLOAT current_time,
    FLUCS_COMPLEX fields[NUMBER_OF_FIELDS][HALFSIZE],
    FLUCS_COMPLEX dft_derivatives[NUMBER_OF_DFT_DERIVATIVES][HALFSIZE]
) {
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;                                                                                                                                                                             
    if (!(index < HALFSIZE))
        return;

    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    if (is_mode_padded(ikz, ikx, iky)) {
        dft_derivatives[0][index] = 0;
        dft_derivatives[1][index] = 0;
        dft_derivatives[2][index] = 0;
        return;
    }

    const FLUCS_COMPLEX dx = dx_from_ikx(ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(iky);

    const FLUCS_COMPLEX phi = fields[0][index];
    const FLUCS_COMPLEX T = fields[1][index];

    dft_derivatives[0][index] = dx * phi;
    dft_derivatives[1][index] = dy * phi;
    dft_derivatives[2][index] = T;
}

__global__ void find_nonlinear_bits(
    FLUCS_FLOAT real_derivatives_global[NUMBER_OF_DFT_DERIVATIVES][FULLSIZE],
    FLUCS_FLOAT real_bits_global[NUMBER_OF_DFT_BITS][FULLSIZE],
    const bool calculate_cfl,
    FLUCS_FLOAT* cfl_rate_global
) {
    const size_t index = blockDim.x * blockIdx.x + threadIdx.x;
    const bool in_bounds = index < FULLSIZE;

    // Ensure that the entire block is handled correctly
    const FLUCS_FLOAT dxphi = in_bounds
        ? real_derivatives_global[0][index]
        : (FLUCS_FLOAT)0;

    const FLUCS_FLOAT dyphi = in_bounds
        ? real_derivatives_global[1][index]
        : (FLUCS_FLOAT)0;

    if (calculate_cfl) {
        const FLUCS_FLOAT cfl_rate =
              flucs_fabs(dxphi) * (NY_UNPADDED / LY)
            + flucs_fabs(dyphi) * (NX_UNPADDED / LX);
        update_cfl(cfl_rate, cfl_rate_global);
    }

    if (!in_bounds)
        return;

    const FLUCS_FLOAT T = real_derivatives_global[2][index];

    real_bits_global[0][index] = dxphi * T;
    real_bits_global[1][index] = dyphi * T;
}

__device__ void add_nonlinear_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFSIZE],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
){
    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;

    // Wavenumbers and indices 
    const FLUCS_COMPLEX dx = dx_from_ikx(ikx);
    const FLUCS_COMPLEX dy = dy_from_iky(iky);

    // Calculate nonlinear terms
    explicit_terms[1] += DFT_FULLSIZE_FACTOR * (
                            + dy * dft_bits_global[0][index]
                            - dx * dft_bits_global[1][index]
                        );

}


#ifdef COMPLETE_TIMESTEP
__device__ __forceinline__
void complete_finish_step(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX previous_fields_global[NUMBER_OF_FIELDS][HALFSIZE],
    FLUCS_COMPLEX current_fields_global[NUMBER_OF_FIELDS][HALFSIZE]
) {
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    const size_t abs_ikx = (ikx < HALF_NX) ? ikx : NX - ikx;  
    const size_t abs_ikz = (ikz < HALF_NZ) ? ikz : NZ - ikz;  

    if (abs_ikz <= FROZEN_CUTOFF_IKZ
        && abs_ikx <= FROZEN_CUTOFF_IKX
        && iky <= FROZEN_CUTOFF_IKY) {

        current_fields_global[0][index] = previous_fields_global[0][index];
        current_fields_global[1][index] = previous_fields_global[1][index];
    }

}
#endif


struct Heatflux_Functor {
    const FLUCS_COMPLEX* __restrict__ fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_COMPLEX* phi = fields;
        const FLUCS_COMPLEX* T = fields + HALFSIZE;
        return ((FLUCS_FLOAT)(-1.5))
            * (Dy_Functor{phi}(index) * CC_Functor{T}(index)).real();
    }
};

struct FreeEnergy_Functor {
    const FLUCS_COMPLEX* __restrict__ fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX T = fields[index + HALFSIZE];

        const FLUCS_FLOAT phi2_contribution = (
            phi.real() * phi.real() + phi.imag() * phi.imag()
        ) * (1 + 1 / TAUBAR) / (2 * TAUBAR);

        const FLUCS_FLOAT T2_contribution = (3.0/4) * (
            T.real() * T.real() + T.imag() * T.imag()
        );

        return phi2_contribution + T2_contribution;
    }
};

// Spectral diagnostics.  The reductions account for the omitted negative-ky
// half of the real-to-complex transform when appropriate.
struct PhiSquared_Functor {
    const FLUCS_COMPLEX* __restrict__ fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_COMPLEX phi = fields[index];
        return phi.real() * phi.real() + phi.imag() * phi.imag();
    }
};

struct TSquared_Functor {
    const FLUCS_COMPLEX* __restrict__ fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_COMPLEX temperature = fields[index + HALFSIZE];
        return temperature.real() * temperature.real()
            + temperature.imag() * temperature.imag();
    }
};

struct FreeEnergyColl_Functor {
    const FLUCS_COMPLEX* __restrict__ fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX T = fields[index + HALFSIZE];

        indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
        const size_t ikz = indices.ikz;
        const FLUCS_FLOAT kz = kz_from_ikz(ikz);

        const FLUCS_COMPLEX first_bit = (
            (1 + 1/TAUBAR) * phi - (1 + COEFFB/COEFFA) * T
        );

        return kz * kz * (
            COEFFA * (
                first_bit.real() * first_bit.real()
                + first_bit.imag() * first_bit.imag()
            ) + COEFFC * (
                T.real() *  T.real() + T.imag() * T.imag()
            )    
        );
    }
};

struct FreeEnergyHyperdissipation_Functor {
    const FLUCS_COMPLEX* fields;
    const FLUCS_FLOAT adaptive_rate;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * Hyperdissipation_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields},
                adaptive_rate
            }(index);
    }
};

struct FreeEnergyHyperdissipationComponent_Functor {
    const FLUCS_COMPLEX* __restrict__ fields;
    const FLUCS_FLOAT adaptive_rate;
    const int hyperdissipation_type;

    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        return (FLUCS_FLOAT)2.0
            * HyperdissipationSelector_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields},
                adaptive_rate,
                hyperdissipation_type
            }(index);
    }
};

} // extern "C"
