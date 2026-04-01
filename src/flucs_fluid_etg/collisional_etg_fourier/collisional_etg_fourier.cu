/*
 * Contains all the CUDA kernels for the 2D ITG model of Ivanov et al. (2020).
 */

// A lot of basic functionality is already implemented here.
#include "flucs/solvers/fourier/fourier_system.cuh"

extern "C" {

// Array for AB3 nonlinear terms
__constant__ FLUCS_COMPLEX* multistep_nonlinear_terms = NULL;

__device__ void get_linear_matrix(const size_t index, const FLUCS_FLOAT dt, FLUCS_COMPLEX matrix[2][2]){
    // First, we need to figure out the kx and ky of the mode.
    // const size_t ikx = index / HALF_NY;
    // const size_t iky = index % HALF_NY;

    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    // const FLUCS_FLOAT kx = (ikx < HALF_NX) ? TWOPI_OVER_LX * ikx : TWOPI_OVER_LX * (ikx - NX);
    // const FLUCS_FLOAT ky = TWOPI_OVER_LY * iky;
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


__global__ void find_derivatives(const FLUCS_COMPLEX* fields,
                                 FLUCS_COMPLEX* dft_derivatives,
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
        cfl_rate[0] = 0;

    // Check if mode should be zeroed
    if (   (padded_ikx >= HALF_NX && padded_ikx < (HALF_NX + PADDED_NX) - NX)
        || (padded_ikz >= HALF_NZ && padded_ikz < (HALF_NZ + PADDED_NZ) - NZ)
        || padded_iky >= HALF_NY){

        dft_derivatives[padded_index] = 0;
        dft_derivatives[padded_index + HALFPADDEDSIZE] = 0;
        dft_derivatives[padded_index + 2*HALFPADDEDSIZE] = 0;
        return;
    }
    

    const size_t ikx = ikx_from_padded_ikx(padded_ikx);
    const size_t ikz = ikz_from_padded_ikz(padded_ikz);

    const size_t index = index_from_3d<NZ, NX, HALF_NY>(ikz, ikx, padded_iky);

    const FLUCS_FLOAT kx = kx_from_ikx(ikx);

    // padded_iky and iky are the same for nonzero modes
    const FLUCS_FLOAT ky = ky_from_iky(padded_iky);

    const FLUCS_COMPLEX phi = fields[index];
    const FLUCS_COMPLEX T = fields[index + HALFUNPADDEDSIZE];

    dft_derivatives[padded_index]\
        = FLUCS_COMPLEX(-kx * phi.imag(), kx * phi.real());

    dft_derivatives[padded_index + HALFPADDEDSIZE]\
        = FLUCS_COMPLEX(-ky * phi.imag(), ky * phi.real());

    dft_derivatives[padded_index + 2*HALFPADDEDSIZE]\
        = T;
}


__global__ void find_nonlinear_bits(FLUCS_FLOAT* real_derivatives_and_bits,
                                    FLUCS_FLOAT* cfl_rate){
    // Shared memory for CFL calculations
    extern __shared__ float cfl_shared[];

    const size_t real_index = blockDim.x * blockIdx.x + threadIdx.x;

    // Check if we are within bounds
    if (!(real_index < PADDEDSIZE))
        return;

    const FLUCS_FLOAT dxphi = real_derivatives_and_bits[real_index];
    const FLUCS_FLOAT dyphi = real_derivatives_and_bits[real_index + PADDEDSIZE];
    const FLUCS_FLOAT T = real_derivatives_and_bits[real_index + 2*PADDEDSIZE];

    const FLUCS_FLOAT cfl = flucs_fabs(dxphi) * (NY / LY) + flucs_fabs(dyphi) * (NX / LX);
    // cfl_array[real_index] = cfl;

    // Find max CFL using shared memory
    // TODO: Could we speed this up by reducing over warps?
    cfl_shared[threadIdx.x] = cfl;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            cfl_shared[threadIdx.x] = fmaxf(cfl_shared[threadIdx.x], cfl_shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    // First thread in block writes to global max via atomic
    if (threadIdx.x == 0) {
        atomicMaxFloat(cfl_rate, cfl_shared[0]); // custom atomic for float
    }

    // dxphi T
    real_derivatives_and_bits[real_index] = dxphi * T;

    // dyphi T
    real_derivatives_and_bits[real_index + PADDEDSIZE] = dyphi * T;
}

__device__ void add_nonlinear_terms(const size_t index,
                                    const FLUCS_FLOAT dt,
                                    const long long current_step,
                                    const FLUCS_FLOAT AB0,
                                    const FLUCS_FLOAT AB1,
                                    const FLUCS_FLOAT AB2,
                                    const FLUCS_COMPLEX* dft_bits,
                                    FLUCS_COMPLEX* rhs_fields){

    indices3d_t indices = get_indices3d<NZ, NX, HALF_NY>(index);
    const size_t ikx = indices.ikx;
    const size_t iky = indices.iky;
    const size_t ikz = indices.ikz;

    const FLUCS_FLOAT kx = kx_from_ikx(ikx);
    const FLUCS_FLOAT ky = ky_from_iky(iky);

    const size_t padded_ikx = padded_ikx_from_ikx(ikx);
    const size_t padded_ikz = padded_ikz_from_ikz(ikz);

    const size_t padded_index = index_from_3d<PADDED_NZ, PADDED_NX, HALF_PADDED_NY>(padded_ikz, padded_ikx, iky);

    const FLUCS_COMPLEX TNL = DFT_PADDEDSIZE_FACTOR * (
                              FLUCS_COMPLEX(-ky * dft_bits[padded_index].imag(),
                                             ky * dft_bits[padded_index].real())
                             +FLUCS_COMPLEX( kx * dft_bits[padded_index + HALFPADDEDSIZE].imag(),
                                            -kx * dft_bits[padded_index + HALFPADDEDSIZE].real()));

    const size_t multistep_index_0 = ((current_step      % 3 + 3) % 3) * HALFUNPADDEDSIZE + index;
    const size_t multistep_index_1 = ((current_step + 2) % 3)          * HALFUNPADDEDSIZE + index;
    const size_t multistep_index_2 = ((current_step + 1) % 3)          * HALFUNPADDEDSIZE + index;

    // T
    rhs_fields[1] -= dt * (AB0*TNL
                           +AB1*multistep_nonlinear_terms[multistep_index_1]
                           +AB2*multistep_nonlinear_terms[multistep_index_2]);

    multistep_nonlinear_terms[multistep_index_0] = TNL;
}

__global__
void heatflux_kzkx(
    const FLUCS_COMPLEX* phi,
    const FLUCS_COMPLEX* T,
    FLUCS_COMPLEX* output){

    multiply_and_sum_last_axis<HALF_NY, true>(
            COMPLEX_ONE,
            output,
            Dy_Functor{phi},
            CC_Functor{T}
        );

}

struct FreeEnergy_Functor {
    const FLUCS_COMPLEX* __restrict__ fields;
    const FLUCS_FLOAT multiplier;
    __device__ __forceinline__ FLUCS_FLOAT operator()(size_t index) const {

        const FLUCS_COMPLEX phi = fields[index];
        const FLUCS_COMPLEX T = fields[index + HALFUNPADDEDSIZE];

        const FLUCS_FLOAT phi2_bit = (
            phi.real() * phi.real() + phi.imag() * phi.imag()
        ) * (1 + 1 / TAUBAR) / (2 * TAUBAR);

        const FLUCS_FLOAT T2_bit = (3.0/4) * (T.real() * T.real() + T.imag() * T.imag());

        return multiplier * (phi2_bit + T2_bit);
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

__global__
void free_energy_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            FreeEnergy_Functor{fields, FLOAT_ONE}
        );

}


__global__
void dW_kzkx(
    const FLUCS_COMPLEX* fields_now,
    const FLUCS_COMPLEX* fields_prev,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            (FLUCS_FLOAT)1.0,
            output,
            FreeEnergy_Functor{fields_now, FLOAT_ONE},
            FreeEnergy_Functor{fields_prev, -FLOAT_ONE}
        );

}

__global__
void free_energy_collisional_loss_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            FreeEnergyColl_Functor{fields}
        );

}

__global__
void W_hyperdissipation_perp_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            HyperdissipationPerp_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields, (FLUCS_FLOAT)2.0}
            }
        );

}

__global__
void W_hyperdissipation_kx_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            HyperdissipationKx_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields, (FLUCS_FLOAT)2.0}
            }
        );

}

__global__
void W_hyperdissipation_ky_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            HyperdissipationKy_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields, (FLUCS_FLOAT)2.0}
            }
        );

}

__global__
void W_hyperdissipation_kz_kzkx(
    const FLUCS_COMPLEX* fields,
    FLUCS_FLOAT* output){

    add_and_sum_last_axis<HALF_NY, true>(
            FLOAT_ONE,
            output,
            HyperdissipationKz_Functor<FreeEnergy_Functor>{
                FreeEnergy_Functor{fields, (FLUCS_FLOAT)2.0}
            }
        );

}

} // extern "C"
