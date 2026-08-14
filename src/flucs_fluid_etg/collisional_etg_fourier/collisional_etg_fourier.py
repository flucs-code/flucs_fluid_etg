"""
Pseudospectral Fourier implementation of collisional ETG model of Adkins et al.
(2023).The nonlinear term is handled explicitly using the Adams-Bashforth
3-step method.
"""
from typing import ClassVar

import cupy as cp
import numpy as np
from cupy.cuda import cufft

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
            real_derivatives: cp.ndarray,
            real_bits: cp.ndarray,
            calculate_cfl: bool,
        ) -> None:
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
            dft_derivatives: cp.ndarray,
        ) -> None:
            self.find_derivatives_kernel(
                self.float(current_time),
                fields,
                dft_derivatives,
            )

        if not self.input["setup.linear"]:
            self.dft_derivatives_operation = (
                self.create_dft_derivatives_operation(
                    find_derivatives_function=find_derivatives_function,
                    find_real_bits_function=find_nonlinear_bits_function,
                )
            )

    def _allocate_memory(self):
        # GPU arrays

        # First, call FourierSystem's method which allocates
        # self.fields among other things.
        super()._allocate_memory(
            allocate_derivatives_and_bits=True,
            combine_derivatives_and_bits=True
        )

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
            self.dft_bits,
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
