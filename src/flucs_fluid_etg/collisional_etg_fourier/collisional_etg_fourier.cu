/*
 * Contains all the CUDA kernels for the 3D ETG model of Adkins et al. (2023).
 */

// A lot of basic functionality is already implemented here.
#include "flucs/solvers/fourier/fourier_system.cuh"

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
        (2 * (1 + TAUBAR) * KAPPAB - TAUBAR * KAPPAN) * ky);

    matrix[0][1] = FLUCS_COMPLEX(
        -TAUBAR * (COEFFA + COEFFB) * kz * kz,
        -2 * TAUBAR * KAPPAB * ky);

    matrix[1][0] = FLUCS_COMPLEX(
        -(2.0/3) * (COEFFA + COEFFB) * (1 + 1/TAUBAR) * kz * kz,
        (KAPPAT - (4.0/3) * (1 + 1/TAUBAR) * KAPPAB) * ky);

    matrix[1][1] = FLUCS_COMPLEX(
        (2.0/3) * (COEFFC + COEFFA*(1 + COEFFB/COEFFA)*(1 + COEFFB/COEFFA)) * kz * kz,
        (14.0/3) * KAPPAB * ky);
}


__global__ void find_derivatives(const FLUCS_COMPLEX fields_global[NUMBER_OF_FIELDS][HALFUNPADDEDSIZE],
                                 FLUCS_COMPLEX dft_derivatives_global[NUMBER_OF_DFT_DERIVATIVES][HALFPADDEDSIZE],
                                 FLUCS_FLOAT* cfl_rate){
    const size_t padded_index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(padded_index < HALFPADDEDSIZE))
        return;

    indices3d_t padded_indices = get_indices3d<PADDED_NZ, PADDED_NX, HALF_PADDED_NY>(padded_index);
    const size_t padded_ikx = padded_indices.padded_ikx;
    const size_t padded_iky = padded_indices.padded_iky;
    const size_t padded_ikz = padded_indices.padded_ikz;

    if (padded_index == 0)
        *cfl_rate = 0;

    // Check if mode should be zeroed
    if (   (padded_ikx >= HALF_NX && padded_ikx < (HALF_NX + PADDED_NX) - NX)
        || (padded_ikz >= HALF_NZ && padded_ikz < (HALF_NZ + PADDED_NZ) - NZ)
        || padded_iky >= HALF_NY){

        dft_derivatives_global[0][padded_index] = 0;
        dft_derivatives_global[1][padded_index] = 0;
        dft_derivatives_global[2][padded_index] = 0;
        return;
    }
    

    const size_t ikx = ikx_from_padded_ikx(padded_ikx);
    const size_t ikz = ikz_from_padded_ikz(padded_ikz);

    const size_t index = index_from_3d<NZ, NX, HALF_NY>(ikz, ikx, padded_iky);

    const FLUCS_FLOAT kx = kx_from_ikx(ikx);

    // padded_iky and iky are the same for nonzero modes
    const FLUCS_FLOAT ky = ky_from_iky(padded_iky);

    const FLUCS_COMPLEX phi = fields_global[0][index];
    const FLUCS_COMPLEX T = fields_global[1][index];

    dft_derivatives_global[0][padded_index]\
        = FLUCS_COMPLEX(-kx * phi.imag(), kx * phi.real());

    dft_derivatives_global[1][padded_index]\
        = FLUCS_COMPLEX(-ky * phi.imag(), ky * phi.real());

    dft_derivatives_global[2][padded_index]\
        = T;
}


__global__ void find_nonlinear_bits(FLUCS_FLOAT real_derivatives_and_bits_global[NUMBER_OF_DFT_COMBINED][PADDEDSIZE],
                                    FLUCS_FLOAT* cfl_rate){
    // Shared memory for CFL calculations
    extern __shared__ FLUCS_FLOAT cfl_shared[];

    const size_t real_index = blockDim.x * blockIdx.x + threadIdx.x;
    const bool in_bounds = real_index < PADDEDSIZE;

    // Inactive threads do not contribute to the cfl reduction 
    const FLUCS_FLOAT dxphi = in_bounds
        ? real_derivatives_and_bits_global[0][real_index]
        : (FLUCS_FLOAT)0;
    const FLUCS_FLOAT dyphi = in_bounds
        ? real_derivatives_and_bits_global[1][real_index]
        : (FLUCS_FLOAT)0;

    const FLUCS_FLOAT cfl = flucs_fabs(dxphi) * (NY / LY)
        + flucs_fabs(dyphi) * (NX / LX);

    // Find max CFL using shared memory
    // TODO: Could we speed this up by reducing over warps?
    cfl_shared[threadIdx.x] = cfl;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            cfl_shared[threadIdx.x] = flucs_fmax(cfl_shared[threadIdx.x], cfl_shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // First thread in block writes to global max via atomic
    if (threadIdx.x == 0) {
        atomicMaxFloat(cfl_rate, cfl_shared[0]); // custom atomic for float
    }

    // Out-of-bounds threads should not contribute to nonlinear bits
    if (!in_bounds)
        return;

    const FLUCS_FLOAT T = real_derivatives_and_bits_global[2][real_index];

    // dxphi T
    real_derivatives_and_bits_global[0][real_index] = dxphi * T;

    // dyphi T
    real_derivatives_and_bits_global[1][real_index] = dyphi * T;
}

__device__ void add_nonlinear_terms(
    const size_t index,
    const FLUCS_FLOAT dt,
    const FLUCS_FLOAT current_time,
    const long long current_step,
    const FLUCS_COMPLEX dft_bits_global[NUMBER_OF_DFT_BITS][HALFPADDEDSIZE],
    FLUCS_COMPLEX explicit_terms[NUMBER_OF_FIELDS]
){
    // Indices
    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    // Wavenumbers and indices 
    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const size_t padded_ikx = padded_ikx_from_ikx(ikx);
    const size_t padded_ikz = padded_ikz_from_ikz(ikz);
    const size_t padded_index = index_from_3d<PADDED_NZ, PADDED_NX, HALF_PADDED_NY>(padded_ikz, padded_ikx, iky);

    // Calculate nonlinear terms
    explicit_terms[1] += DFT_PADDEDSIZE_FACTOR * (
                            + FLUCS_COMPLEX(-ky * dft_bits_global[0][padded_index].imag(),
                                             ky * dft_bits_global[0][padded_index].real())
                            + FLUCS_COMPLEX( kx * dft_bits_global[1][padded_index].imag(),
                                            -kx * dft_bits_global[1][padded_index].real()));

}

struct Heatflux_Functor {
    const FLUCS_COMPLEX* __restrict__ fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {
        const FLUCS_COMPLEX* phi = fields;
        const FLUCS_COMPLEX* T = fields + HALFUNPADDEDSIZE;
        return ((FLUCS_FLOAT)(-1.5))
            * (Dy_Functor{phi}(index) * CC_Functor{T}(index)).real();
    }
};

struct FreeEnergy_Functor {
    const FLUCS_COMPLEX* __restrict__ fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX T = fields[index + HALFUNPADDEDSIZE];

        const FLUCS_FLOAT phi2_contribution = (
            phi.real() * phi.real() + phi.imag() * phi.imag()
        ) * (1 + 1 / TAUBAR) / (2 * TAUBAR);

        const FLUCS_FLOAT T2_contribution = (3.0/4) * (T.real() * T.real() + T.imag() * T.imag());

        return phi2_contribution + T2_contribution;
    }
};

struct FreeEnergyColl_Functor {
    const FLUCS_COMPLEX* __restrict__ fields;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX T = fields[index + HALFUNPADDEDSIZE];

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
