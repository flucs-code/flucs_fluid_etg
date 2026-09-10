"""
Pseudospectral Fourier implementation of collisional ETG model of Adkins et al.
(2023).The nonlinear term is handled explicitly using the Adams-Bashforth
3-step method.
"""
from typing import ClassVar

import cupy as cp
import numpy as np

from .collisional_etg_fourier_diagnostics import HeatfluxDiag
from .collisional_etg_fourier_diagnostics import FreeEnergyDiag

from flucs.diagnostic import FlucsDiagnostic
from flucs.utilities.cupy import KernelWrapper
from flucs.solvers.fourier.fourier_system import FourierSystem


class CollisionalETGFourier(FourierSystem):
    """Fourier solver for the 3D collisional ETG system."""
    number_of_fields = 2
    number_of_dft_derivatives = 3
    number_of_dft_bits = 2

    # Direct pointers to the phi and T arrays
    phi: list
    T: list

    # CUDA grids and kernels 
    find_derivatives_kernel: KernelWrapper
    find_nonlinear_bits_kernel: KernelWrapper

    # Supported diagnostics
    diags: ClassVar[set[type[FlucsDiagnostic]]] = {
        HeatfluxDiag, FreeEnergyDiag
    }

    def ready(self):
        # Anything system-specific goes here
        super().ready()

    def register_kernels(self):
        super().register_kernels()

        nonlinear_bits_shared_mem = (
            self.cuda_block_size * self.float().nbytes
        )

        # System-specific kernels
        self.find_derivatives_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="find_derivatives",
            grid=(self.half_cuda_grid_size,),
            block=(self.cuda_block_size,),
        )

        self.find_nonlinear_bits_kernel = KernelWrapper(
            system=self,
            cuda_kernel_name="find_nonlinear_bits",
            grid=(self.full_cuda_grid_size,),
            block=(self.cuda_block_size,),
            shared_mem=nonlinear_bits_shared_mem,
        )

        # Define functions from kernels
        def find_nonlinear_bits_function(
            current_dt,
            current_time,
            current_step: int,
            calculate_cfl: bool,
            memory_dict: dict,
        ) -> None:
            real_derivatives = memory_dict["first_intermediates_real"]
            real_bits = memory_dict["second_intermediates_real"]
            self.find_nonlinear_bits_kernel(
                real_derivatives,
                real_bits,
                calculate_cfl,
                self.cfl_rate,
            )

        def find_derivatives_function(
            current_dt,
            current_time,
            current_step: int,
            fields: cp.ndarray,
            memory_dict: dict,
        ) -> None:
            self.find_derivatives_kernel(
                self.float(current_time),
                fields,
                memory_dict["first_intermediates_fourier"],
            )

        if not self.input["setup.linear"]:
            self.dft_derivatives_operation, self.dft_bits = (
                self.create_dealiased_operation(
                    n_in=self.number_of_dft_derivatives,
                    n_out=self.number_of_dft_bits,
                    create_first_intermediates=find_derivatives_function,
                    create_second_intermediates=find_nonlinear_bits_function,
                    combine_first_and_second_intermediates=True,
                )
            )

    def _allocate_memory(self):
        # GPU arrays

        # First, call FourierSystem's method which allocates
        # self.fields among other things.
        super()._allocate_memory()

        # Direct pointers to fields
        self.phi = [cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[0][0, 0, 0, 0].data),
                    cp.ndarray((self.nz, self.nx, self.half_ny),
                               dtype=self.complex,
                               memptr=self.fields[1][0, 0, 0, 0].data),]

        self.T = [cp.ndarray((self.nz, self.nx, self.half_ny),
                             dtype=self.complex,
                             memptr=self.fields[0][1, 0, 0, 0].data),
                  cp.ndarray((self.nz, self.nx, self.half_ny),
                             dtype=self.complex,
                             memptr=self.fields[1][1, 0, 0, 0].data),]

        # All fields and derivatives to be transformed to real space
        # are kept in one huge array (dft_derivatives).
        # The first index indexes the fields and it's meaning is
        # 0 dxphi,
        # 1 dyphi,
        # 2 T

        # The NL bits here are
        # 0 dxphi * T
        # 1 dyphi * T

        # The arrays for the above are handled by FourierSystem.
        # There are no system-specific arrays that we need to allocate here 

    def _set_initial_conditions(self) -> None:
        super()._set_initial_conditions()

        frozen_amplitude = self.input["parameters.frozen.amplitude"]

        if frozen_amplitude >= 0.0:
            
            # set region below cutoff to 0:
            co_ikz = self.input["parameters.frozen.cutoff_ikz"]
            co_ikx = self.input["parameters.frozen.cutoff_ikx"]
            co_iky = self.input["parameters.frozen.cutoff_iky"]

            if co_ikx<0 or co_iky<0 or co_ikz<0:
                from flucs.input import InvalidFlucsInputFileError
                raise InvalidFlucsInputFileError('Freezing cut-offs must be positive.')

            iz = np.r_[
                np.arange(co_ikz+1),
                np.arange(self.nz - co_ikz , self.nz)
            ]

            ix = np.r_[
                np.arange(co_ikx+1),
                np.arange(self.nx - co_ikx , self.nx)
            ]

            iy = np.arange(co_iky+1)

            self.fields_initial[
                :,
                iz[:, None, None],
                ix[None, :, None],
                iy[None, None, :]
            ] = 0
            
            from flucs.utilities.messages import flucsprint
            flucsprint(f"Freezing fields below (ikz,ikx,iky) ="
                       +f" ({co_ikz,co_ikx,co_iky})")

            # set streamer initial condition
            if self.input["parameters.frozen.use_eigenmode"]:
                
                eigsys = self.compute_linear_eigensystem_cpu()
                eigvals = eigsys["eigvals"][:,1,0,1]
                # (mode,          nz, nx, half_ny) 
                idx_unstable = np.argmax(eigvals.imag)
                eigvec = eigsys["eigvecs"][idx_unstable,:,1,0,1]
                # (mode, nfields, nz, nx, half_ny) 
                
                # normalise
                box_mode = frozen_amplitude * eigvec / eigvec[1]

            else:
                # convention is that phase and ratio determine
                # \varphi / \delta T
                frozen_phase = self.input["parameters.frozen.streamer_phase"]
                frozen_ratio = self.input["parameters.frozen.streamer_ratio"]
                frozen_ratio *= np.cos(frozen_phase)+np.sin(frozen_phase)*1j
                box_mode = frozen_amplitude * np.array([frozen_ratio,1])

            flucsprint(f"Set box mode amplitude to phase {np.angle(box_mode[0]):.2f} and ratio {np.abs(box_mode[0]/frozen_amplitude):.2f}")
            
            self.fields_initial[:,1,0,1] = box_mode


    def _interpret_input(self):
        """Checks if the input file makes sense"""

        # Make sure to call the parent method to do some standard setup
        # (resolution checks, etc)
        super()._interpret_input()

        # Anything custom goes here

        # Setting default values of collisional coefficients
        charge = self.input["parameters.charge"]

        coeffa = self.input["parameters.coeffa"]
        coeffb = self.input["parameters.coeffb"]
        coeffc = self.input["parameters.coeffc"]

        if coeffa < 0:
            coeffa = (
                (217/64 + 151/(8 * np.sqrt(2) * charge) + 9/(2 * charge**2))
                / (1 + 61/(8 * np.sqrt(2) * charge) + 9/(2 * charge**2))
            )

        if coeffb < 0:
            coeffb = 2.5 * (
                (33/16 + 45/(8 * np.sqrt(2) * charge))
                / (1 + 61/(8 * np.sqrt(2) * charge) + 9/(2 * charge**2))
            )

        if coeffc < 0:
            coeffc = 6.25 * (
                (13/4 + 45/(8 * np.sqrt(2) * charge))
                / (1 + 61/(8 * np.sqrt(2) * charge) + 9/(2 * charge**2))
            )
            coeffc = coeffc - (coeffb**2)/coeffa

        # Hack, remove, cannot push this kind of stuff...
        self.input._initialised = False
        self.input["parameters.coeffa"] = coeffa
        self.input["parameters.coeffb"] = coeffb
        self.input["parameters.coeffc"] = coeffc
        self.input._initialised = True

    def setup_cuda_definitions(self) -> None:
        # System-specific constants for the kernels

        self.module_options.define_float("KAPPAT",
                                            self.input["parameters.kappaT"])
        self.module_options.define_float("KAPPAN",
                                            self.input["parameters.kappaN"])
        self.module_options.define_float("KAPPAB",
                                            self.input["parameters.kappaB"])

        self.module_options.define_float("COEFFA",
                                            self.input["parameters.coeffa"])
        self.module_options.define_float("COEFFB",
                                            self.input["parameters.coeffb"])
        self.module_options.define_float("COEFFC",
                                            self.input["parameters.coeffc"])

        if self.input["parameters.frozen.amplitude"] > 0:
            self.module_options.define_flag("COMPLETE_TIMESTEP")
            
            self.module_options.define_int(
                "FROZEN_CUTOFF_IKX",
                self.input["parameters.frozen.cutoff_ikx"]
            )
            self.module_options.define_int(
                "FROZEN_CUTOFF_IKY",
                self.input["parameters.frozen.cutoff_iky"]
            )
            self.module_options.define_int(
                "FROZEN_CUTOFF_IKZ",
                self.input["parameters.frozen.cutoff_ikz"]
            )


        charge = self.input["parameters.charge"]
        tratio = self.input["parameters.tratio"]
        self.module_options.define_float("TAUBAR",
                                            tratio / charge)

        # Call this setup the CUDA definitions
        super().setup_cuda_definitions()

    def begin_time_step(self) -> None:
        # Do anything model-specific here, then call the parent's method
        super().begin_time_step()

    def compute_nonlinear_terms(self, current_dt, current_time, current_step, fields: cp.ndarray, calculate_cfl) -> None:
        """
        Computes the nonlinear terms for the supplied fields. Here, we also
        determine the nonlinear CFL coefficient.

        """
        self.dft_derivatives_operation(
            current_dt,
            current_time,
            current_step,
            fields,
            calculate_cfl=calculate_cfl,
        )

    def finish_time_step(self) -> None:
        super().finish_time_step()

    def compute_linear_matrix_reference(self) -> np.ndarray:
        # Initialise linear matrix
        linear_matrix = np.zeros(
            (
                self.number_of_fields,
                self.number_of_fields,
                *self.half_tuple
            ),
            dtype=self.complex,
        )

        # Get wavenumbers
        kz, kx, ky = self.get_broadcast_wavenumbers()

        # Get parameters
        kappaT = self.input["parameters.kappaT"]
        kappaN = self.input["parameters.kappaN"]
        kappaB = self.input["parameters.kappaB"]

        coeffa = self.input["parameters.coeffa"]
        coeffb = self.input["parameters.coeffb"]
        coeffc = self.input["parameters.coeffc"]

        taubar = (
            self.input["parameters.tratio"] / self.input["parameters.charge"]
        )

        # phi-phi
        linear_matrix[0, 0, :, :, :] = (
            coeffa * (1.0 + taubar) * (kz**2)
            + 1j * (2.0 * (1.0 + taubar) * kappaB - taubar * kappaN) * ky
        )

        # phi-T
        linear_matrix[0, 1, :, :, :] = (
            -taubar * (coeffa + coeffb) * (kz**2)
            - 1j * 2.0 * taubar * kappaB * ky
        )

        # T-phi
        linear_matrix[1, 0, :, :, :] = (
            -(2.0 / 3.0) * (coeffa + coeffb) * (1.0 + 1.0 / taubar) * (kz**2)
            + 1j * (kappaT - (4.0 / 3.0) * (1.0 + 1.0 / taubar) * kappaB) * ky
        )

        # T-T
        linear_matrix[1, 1, :, :, :] = (
            (2.0 / 3.0) * (coeffc + coeffa * (1.0 + coeffb/coeffa)**2) * (kz**2)
            + 1j * (14.0 / 3.0) * kappaB * ky
        )

        return linear_matrix


    def compute_linear_eigensystem_cpu(self):
        """
        Computes both the eigenvalues and (normalised) eigenvectors
        of the linear matrix that is used by the system.

        The eigenvalues are the complex frequencies of
        Fourier modes of the form exp(-i*omega*t).

        The eigenvectors are normalised to unit L2 norm and a phase
        where the component with largest absolute value is real and positive.

        This is similar to FourierSystem.compute_linear_eigensystem
        """
        linear_matrix = self.compute_linear_matrix_reference()

        # Shape: (field, field, z, x, ky)
        matrix = np.moveaxis(linear_matrix, (0, 1), (-2, -1))

        eigvals, eigvecs = np.linalg.eig(matrix)

        # Match FourierSystem convention:
        eigvals = (-1j * eigvals).transpose(3, 0, 1, 2)
        eigvecs = eigvecs.transpose(4, 3, 0, 1, 2)

        eigvecs /= np.linalg.norm(eigvecs, axis=1, keepdims=True)

        indices = np.abs(eigvecs).argmax(axis=1, keepdims=True)
        components = np.take_along_axis(eigvecs, indices, axis=1)

        phase = np.where(
            np.abs(components) > 0,
            np.sign(components),
            1.0 + 0.0j,
        )
        eigvecs *= np.conj(phase)

        # Compute inverse of solver eigenvectors for projection
        eigvecs_inverse = np.linalg.inv(
            eigvecs.transpose(2, 3, 4, 1, 0)
        ).transpose(3, 4, 0, 1, 2)

        # Return dict
        return {
            "eigvals": eigvals,
            "eigvecs": eigvecs,
            "eigvecs_inverse": eigvecs_inverse,
        }


