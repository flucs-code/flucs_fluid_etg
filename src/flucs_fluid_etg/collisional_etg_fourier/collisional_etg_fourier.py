"""
Pseudospectral Fourier implementation of collisional ETG model of Adkins et al.
(2023).The nonlinear term is handled explicitly using the Adams-Bashforth
3-step method.
"""

from typing import ClassVar

import cupy as cp
import numpy as np
from flucs.diagnostic import FlucsDiagnostic
from flucs.input import InvalidFlucsInputFileError
from flucs.solvers.fourier.fourier_system import FourierSystem
from flucs.utilities.cupy import KernelWrapper
from flucs.utilities.messages import flucsprint

from .collisional_etg_fourier_diagnostics import (
    FreeEnergyDiag,
    HeatfluxDiag,
    SpectraDiag,
    SpectralFreeEnergyDiag,
    StreamerExchangeDiag,
)


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

    # Streamer ([parameters.streamer]); see _setup_streamer
    streamer_mode: str
    streamer_iky: int
    streamer_ky: float
    streamer_kc: float
    streamer_kc2: float
    streamer_dT_amplitude: float
    streamer_seed_W: float
    streamer_initialised: bool = False

    # Relative tolerance on kc^2 that excludes modes with kperp = kc exactly
    # (e.g. kx = 0, ky = 2 ky_I) regardless of round-off
    streamer_kc2_tolerance: ClassVar[float] = 1e-6

    # Supported diagnostics
    diags: ClassVar[set[type[FlucsDiagnostic]]] = {
        HeatfluxDiag,
        FreeEnergyDiag,
        SpectralFreeEnergyDiag,
        SpectraDiag,
        StreamerExchangeDiag,
    }

    def ready(self):
        # Anything system-specific goes here
        super().ready()

        # The initial fields are on the GPU now
        if self.streamer_initialised:
            self._check_streamer_amplitude()

    def register_kernels(self):
        super().register_kernels()

        nonlinear_bits_shared_mem = self.cuda_block_size * self.float().nbytes

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
                self.time_float(current_time),
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
        self.phi = [
            cp.ndarray(
                (self.nz, self.nx, self.half_ny),
                dtype=self.complex,
                memptr=self.fields[0][0, 0, 0, 0].data,
            ),
            cp.ndarray(
                (self.nz, self.nx, self.half_ny),
                dtype=self.complex,
                memptr=self.fields[1][0, 0, 0, 0].data,
            ),
        ]

        self.T = [
            cp.ndarray(
                (self.nz, self.nx, self.half_ny),
                dtype=self.complex,
                memptr=self.fields[0][1, 0, 0, 0].data,
            ),
            cp.ndarray(
                (self.nz, self.nx, self.half_ny),
                dtype=self.complex,
                memptr=self.fields[1][1, 0, 0, 0].data,
            ),
        ]

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

        # Streamer initial condition (restart data takes precedence)
        self.streamer_initialised = False
        if self.streamer_mode and self.restart_manager.data is None:
            self._set_streamer_initial_conditions()

        frozen_amplitude = self.input["parameters.frozen.amplitude"]

        if frozen_amplitude >= 0.0:
            # set region below cutoff to 0:
            co_ikz = self.input["parameters.frozen.cutoff_ikz"]
            co_ikx = self.input["parameters.frozen.cutoff_ikx"]
            co_iky = self.input["parameters.frozen.cutoff_iky"]

            if co_ikx < 0 or co_iky < 0 or co_ikz < 0:
                raise InvalidFlucsInputFileError("Freezing cut-offs must be positive.")

            iz = np.r_[np.arange(co_ikz + 1), np.arange(self.nz - co_ikz, self.nz)]

            ix = np.r_[np.arange(co_ikx + 1), np.arange(self.nx - co_ikx, self.nx)]

            iy = np.arange(co_iky + 1)

            self.fields_initial[
                :, iz[:, None, None], ix[None, :, None], iy[None, None, :]
            ] = 0

            flucsprint(
                "Freezing fields below (ikz,ikx,iky) =" + f" ({co_ikz, co_ikx, co_iky})"
            )

            # set streamer initial condition
            if self.input["parameters.frozen.use_eigenmode"]:
                eigsys = self.compute_linear_eigensystem_cpu()
                eigvals = eigsys["eigvals"][:, 1, 0, 1]
                # (mode,          nz, nx, half_ny)
                idx_unstable = np.argmax(eigvals.imag)
                eigvec = eigsys["eigvecs"][idx_unstable, :, 1, 0, 1]
                # (mode, nfields, nz, nx, half_ny)

                # normalise
                box_mode = frozen_amplitude * eigvec / eigvec[1]

            else:
                # convention is that phase and ratio determine
                # \varphi / \delta T
                frozen_phase = self.input["parameters.frozen.streamer_phase"]
                frozen_ratio = self.input["parameters.frozen.streamer_ratio"]
                frozen_ratio *= np.cos(frozen_phase) + np.sin(frozen_phase) * 1j
                box_mode = frozen_amplitude * np.array([frozen_ratio, 1])

            flucsprint(
                f"Set box mode amplitude to phase {np.angle(box_mode[0]):.2f} and ratio {np.abs(box_mode[0] / frozen_amplitude):.2f}"
            )

            self.fields_initial[:, 1, 0, 1] = box_mode

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
                217 / 64 + 151 / (8 * np.sqrt(2) * charge) + 9 / (2 * charge**2)
            ) / (1 + 61 / (8 * np.sqrt(2) * charge) + 9 / (2 * charge**2))

        if coeffb < 0:
            coeffb = 2.5 * (
                (33 / 16 + 45 / (8 * np.sqrt(2) * charge))
                / (1 + 61 / (8 * np.sqrt(2) * charge) + 9 / (2 * charge**2))
            )

        if coeffc < 0:
            coeffc = 6.25 * (
                (13 / 4 + 45 / (8 * np.sqrt(2) * charge))
                / (1 + 61 / (8 * np.sqrt(2) * charge) + 9 / (2 * charge**2))
            )
            coeffc = coeffc - (coeffb**2) / coeffa

        # Hack, remove, cannot push this kind of stuff...
        self.input._initialised = False
        self.input["parameters.coeffa"] = coeffa
        self.input["parameters.coeffb"] = coeffb
        self.input["parameters.coeffc"] = coeffc
        self.input._initialised = True

        self._interpret_streamer_input()

    def _interpret_streamer_input(self) -> None:
        """Validates [parameters.streamer]. The streamer's ky is chosen later,
        in setup_cuda_definitions_init, once the solved grid is known."""

        self.streamer_mode = self.input["parameters.streamer.mode"]
        if not self.streamer_mode:
            return

        if self.streamer_mode not in ("free", "frozen"):
            raise InvalidFlucsInputFileError(
                "parameters.streamer.mode must be '', 'free' or 'frozen', "
                f"not {self.streamer_mode!r}."
            )

        if self.streamer_mode == "frozen" and (
            self.input["parameters.frozen.amplitude"] >= 0.0
        ):
            raise InvalidFlucsInputFileError(
                "parameters.streamer.mode = 'frozen' cannot be combined with "
                "parameters.frozen.amplitude >= 0."
            )

        if self.nz < 3:
            raise InvalidFlucsInputFileError(
                "The streamer lives at kz = +-2 pi/Lz, which needs nz >= 3."
            )

        # ky_over_kpeak is the default; the others override it
        ky_overrides = [
            self.input["parameters.streamer.iky"] > 0,
            self.input["parameters.streamer.ky"] > 0.0,
            self.input["parameters.streamer.ky_over_kmarg"] > 0.0,
        ]
        if sum(ky_overrides) > 1:
            raise InvalidFlucsInputFileError(
                "Set at most one of parameters.streamer.iky, "
                "parameters.streamer.ky and parameters.streamer.ky_over_kmarg."
            )
        if not any(ky_overrides) and (
            self.input["parameters.streamer.ky_over_kpeak"] <= 0.0
        ):
            raise InvalidFlucsInputFileError(
                "parameters.streamer.ky_over_kpeak must be positive when iky, "
                "ky and ky_over_kmarg are unset."
            )

        amplitude_mode = self.input["parameters.streamer.amplitude_mode"]
        if amplitude_mode not in ("explicit", "predicted"):
            raise InvalidFlucsInputFileError(
                "parameters.streamer.amplitude_mode must be 'explicit' or "
                f"'predicted', not {amplitude_mode!r}."
            )
        if (
            amplitude_mode == "explicit"
            and self.input["parameters.streamer.dT_amplitude"] <= 0.0
        ):
            raise InvalidFlucsInputFileError(
                "parameters.streamer.dT_amplitude must be positive when "
                "amplitude_mode = 'explicit'."
            )

        seed_mode = self.input["parameters.streamer.seed_mode"]
        if seed_mode not in ("predicted", "fraction"):
            raise InvalidFlucsInputFileError(
                "parameters.streamer.seed_mode must be 'predicted' or "
                f"'fraction', not {seed_mode!r}."
            )
        if seed_mode == "predicted" and not (
            self.input["hyperdissipation.kperp"] > 0.0
            and self.input["hyperdissipation.kperp_power"] == 2
            and not self.input["hyperdissipation.kperp_normalised"]
            and not self.input["hyperdissipation.kperp_adaptive"]
        ):
            raise InvalidFlucsInputFileError(
                "seed_mode = 'predicted' uses nu_perp = hyperdissipation.kperp "
                "and needs kperp > 0, kperp_power = 2, kperp_normalised = false "
                "and kperp_adaptive = false."
            )
        if not 0.0 <= self.input["parameters.streamer.seed_kz0_fraction"] <= 1.0:
            raise InvalidFlucsInputFileError(
                "parameters.streamer.seed_kz0_fraction must lie in [0, 1]."
            )

        if self.input["parameters.streamer.kz_structure"] not in ("both", "plus"):
            raise InvalidFlucsInputFileError(
                "parameters.streamer.kz_structure must be 'both' or 'plus'."
            )

        if self.input["parameters.streamer.cutoff_factor"] <= 1.0:
            raise InvalidFlucsInputFileError(
                "parameters.streamer.cutoff_factor must exceed 1, otherwise "
                "the streamer is not in set I."
            )

    def setup_cuda_definitions_init(self) -> None:
        super().setup_cuda_definitions_init()

        # The solved grid and hyperdissipation normalisation are known here
        if self.streamer_mode:
            self._setup_streamer()

            if self.streamer_mode == "frozen":
                self.module_options.define_flag("COMPLETE_TIMESTEP")
                self.module_options.define_flag("FROZEN_STREAMER")
                self.module_options.define_flag(
                    "FROZEN_STREAMER_KC2", repr(self.streamer_kc2)
                )

    def setup_cuda_definitions(self) -> None:
        # System-specific constants for the kernels

        self.module_options.define_float("KAPPAT", self.input["parameters.kappaT"])
        self.module_options.define_float("KAPPAN", self.input["parameters.kappaN"])
        self.module_options.define_float("KAPPAB", self.input["parameters.kappaB"])

        self.module_options.define_float("COEFFA", self.input["parameters.coeffa"])
        self.module_options.define_float("COEFFB", self.input["parameters.coeffb"])
        self.module_options.define_float("COEFFC", self.input["parameters.coeffc"])

        # Same threshold as _set_initial_conditions: amplitude = 0 zeroes and
        # freezes the box
        if self.input["parameters.frozen.amplitude"] >= 0.0:
            self.module_options.define_flag("COMPLETE_TIMESTEP")

            self.module_options.define_int(
                "FROZEN_CUTOFF_IKX", self.input["parameters.frozen.cutoff_ikx"]
            )
            self.module_options.define_int(
                "FROZEN_CUTOFF_IKY", self.input["parameters.frozen.cutoff_iky"]
            )
            self.module_options.define_int(
                "FROZEN_CUTOFF_IKZ", self.input["parameters.frozen.cutoff_ikz"]
            )

        charge = self.input["parameters.charge"]
        tratio = self.input["parameters.tratio"]
        self.module_options.define_float("TAUBAR", tratio / charge)

        # Call this setup the CUDA definitions
        super().setup_cuda_definitions()

    def begin_time_step(self) -> None:
        # Do anything model-specific here, then call the parent's method
        super().begin_time_step()

    def compute_nonlinear_terms(
        self, current_dt, current_time, current_step, fields: cp.ndarray, calculate_cfl
    ) -> None:
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
        kz, kx, ky = self.get_broadcast_wavenumbers()
        return self.compute_linear_matrix_cpu(kz, kx, ky, dtype=self.complex)

    def compute_linear_matrix_cpu(self, kz, kx, ky, dtype=np.complex128) -> np.ndarray:
        """
        Linear matrix M of dfields/dt = -M fields for arbitrary wavenumbers,
        broadcast against each other. Returns shape (2, 2, *broadcast_shape).
        Hyperdissipation is not included; see _hyperdissipation_rate_cpu.
        """
        kz, kx, ky = np.broadcast_arrays(kz, kx, ky)

        # Initialise linear matrix
        linear_matrix = np.zeros(
            (self.number_of_fields, self.number_of_fields, *kz.shape),
            dtype=dtype,
        )

        # Get parameters
        kappaT = self.input["parameters.kappaT"]
        kappaN = self.input["parameters.kappaN"]
        kappaB = self.input["parameters.kappaB"]

        coeffa = self.input["parameters.coeffa"]
        coeffb = self.input["parameters.coeffb"]
        coeffc = self.input["parameters.coeffc"]

        taubar = self.input["parameters.tratio"] / self.input["parameters.charge"]

        # phi-phi
        linear_matrix[0, 0] = (
            coeffa * (1.0 + taubar) * (kz**2)
            + 1j * (2.0 * (1.0 + taubar) * kappaB - taubar * kappaN) * ky
        )

        # phi-T
        linear_matrix[0, 1] = (
            -taubar * (coeffa + coeffb) * (kz**2) - 1j * 2.0 * taubar * kappaB * ky
        )

        # T-phi
        linear_matrix[1, 0] = (
            -(2.0 / 3.0) * (coeffa + coeffb) * (1.0 + 1.0 / taubar) * (kz**2)
            + 1j * (kappaT - (4.0 / 3.0) * (1.0 + 1.0 / taubar) * kappaB) * ky
        )

        # T-T
        linear_matrix[1, 1] = (2.0 / 3.0) * (
            coeffc + coeffa * (1.0 + coeffb / coeffa) ** 2
        ) * (kz**2) + 1j * (14.0 / 3.0) * kappaB * ky

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
        eigvecs_inverse = np.linalg.inv(eigvecs.transpose(2, 3, 4, 1, 0)).transpose(
            3, 4, 0, 1, 2
        )

        # Return dict
        return {
            "eigvals": eigvals,
            "eigvecs": eigvecs,
            "eigvecs_inverse": eigvecs_inverse,
        }

    ###########################################################################
    # Streamer
    ###########################################################################

    def _hyperdissipation_rate_cpu(self, kz, kx, ky) -> np.ndarray:
        """
        Total hyperdissipation rate for arbitrary wavenumbers, matching
        get_hyperdissipation in hyperdissipation.cuh. Adaptive components use
        the initial time step. Needs the hyperdissipation normalisation, so it
        can only be called after the initialisation kernels.
        """
        k2 = {
            "kz": kz**2,
            "kx": kx**2,
            "ky": ky**2,
            "kperp": kx**2 + ky**2,
            "kmod": kz**2 + kx**2 + ky**2,
        }

        rate = np.zeros(np.broadcast(kz, kx, ky).shape)
        for index, component in enumerate(self.hyperdissipation_components):
            coefficient = self.input[f"hyperdissipation.{component}"]
            if coefficient <= 0.0:
                continue

            k2_component = k2[component]
            if self.input[f"hyperdissipation.{component}_normalised"]:
                kmax = self.hyperdissipation_components_kmax[index]
                k2_component = (
                    k2_component / kmax**2 if kmax > 0 else 0.0 * k2_component
                )

            component_rate = (
                coefficient
                * k2_component ** (self.input[f"hyperdissipation.{component}_power"])
            )
            if self.input[f"hyperdissipation.{component}_adaptive"]:
                component_rate = component_rate / self.init_dt

            rate = rate + component_rate

        return rate

    def compute_most_unstable_mode_cpu(
        self, kz, kx, ky, include_hyperdissipation: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Complex frequency omega (fields ~ exp(-i omega t)) and phi/T ratio of
        the most unstable linear eigenmode for arbitrary wavenumbers.

        Hyperdissipation damps both fields equally, so it shifts the growth
        rate but not the eigenvector.
        """
        matrix = np.moveaxis(
            self.compute_linear_matrix_cpu(kz, kx, ky), (0, 1), (-2, -1)
        )
        eigvals, eigvecs = np.linalg.eig(matrix)
        omega = -1j * eigvals

        index = np.argmax(omega.imag, axis=-1)[..., None]
        omega = np.take_along_axis(omega, index, axis=-1)[..., 0]
        eigvec = np.take_along_axis(eigvecs, index[..., None, :], axis=-1)[..., 0]
        ratio = eigvec[..., 0] / eigvec[..., 1]

        if include_hyperdissipation:
            omega = omega - 1j * self._hyperdissipation_rate_cpu(kz, kx, ky)

        return omega, ratio

    def _find_marginal_ky(self) -> float:
        """
        Lowest unstable ky at kx = 0, kz = 2 pi/Lz of the inviscid growth rate,
        found by bisection on gamma_lin(ky) = 0 over the solved ky range.
        """
        kz = 2.0 * np.pi / self.input["dimensions.Lz"]

        def growth_rate(ky):
            return self.compute_most_unstable_mode_cpu(kz, 0.0, ky)[0].imag

        ky_max = self.ky[self.half_ny_unpadded - 1]
        ky_scan = np.geomspace(1e-4 * self.ky[1], ky_max, 4000)
        unstable = np.nonzero(growth_rate(ky_scan) > 0.0)[0]

        if unstable.size == 0:
            raise InvalidFlucsInputFileError(
                "No unstable ky at kx = 0, kz = 2 pi/Lz; ky_over_kmarg is undefined."
            )
        if unstable[0] == 0:
            raise InvalidFlucsInputFileError(
                f"Still unstable at ky = {ky_scan[0]:.3e}; cannot find the marginal ky."
            )

        ky_stable, ky_unstable = ky_scan[unstable[0] - 1], ky_scan[unstable[0]]
        for _ in range(100):
            ky_mid = 0.5 * (ky_stable + ky_unstable)
            if growth_rate(ky_mid) > 0.0:
                ky_unstable = ky_mid
            else:
                ky_stable = ky_mid

        return 0.5 * (ky_stable + ky_unstable)

    def _find_peak_ky(self, ky_marg: float) -> float:
        """
        ky that maximises the inviscid gamma_lin/ky at kx = 0, kz = 2 pi/Lz,
        searched above the marginal ky (gamma_lin/ky has a single maximum
        there) and refined by golden-section search in ln(ky).
        """
        kz = 2.0 * np.pi / self.input["dimensions.Lz"]

        def gamma_over_ky(ky):
            return self.compute_most_unstable_mode_cpu(kz, 0.0, ky)[0].imag / ky

        ky_max = self.ky[self.half_ny_unpadded - 1]
        ky_scan = np.geomspace(ky_marg, ky_max, 4000)
        index = int(np.argmax(gamma_over_ky(ky_scan)))
        if index == len(ky_scan) - 1:
            raise InvalidFlucsInputFileError(
                "gamma_lin/ky still grows at the largest solved ky; "
                "ky_over_kpeak is undefined."
            )

        low = np.log(ky_scan[max(index - 1, 0)])
        high = np.log(ky_scan[index + 1])
        golden = (np.sqrt(5.0) - 1.0) / 2.0
        for _ in range(100):
            mid_low = high - golden * (high - low)
            mid_high = low + golden * (high - low)
            if gamma_over_ky(np.exp(mid_low)) > gamma_over_ky(np.exp(mid_high)):
                high = mid_high
            else:
                low = mid_low

        return float(np.exp(0.5 * (low + high)))

    def get_streamer_kc2(self, kc: float) -> float:
        """kc^2 used by the set-I mask, reduced by streamer_kc2_tolerance so
        that kperp = kc is excluded robustly."""
        return float(kc) ** 2 * (1.0 - self.streamer_kc2_tolerance)

    def get_streamer_ky(self, iky: int) -> float:
        """ky of index iky, evaluated as in in_streamer_set."""
        return iky * (2.0 * np.pi / float(self.float(self.input["dimensions.Ly"])))

    def get_streamer_mask(self, kc2: float) -> np.ndarray:
        """
        Boolean set-I mask on the half grid: solved modes with kz = +-2 pi/Lz
        and kperp^2 < kc2. Mirrors in_streamer_set in the .cu, including the
        precision of the box lengths, so that the CPU and GPU masks agree.
        """
        signed_ikx = np.rint(np.fft.fftfreq(self.nx) * self.nx)
        Lx = float(self.float(self.input["dimensions.Lx"]))
        Ly = float(self.float(self.input["dimensions.Ly"]))
        kx = signed_ikx * (2.0 * np.pi / Lx)
        ky = np.arange(self.half_ny) * (2.0 * np.pi / Ly)

        kz_mask = np.zeros(self.nz, dtype=bool)
        kz_mask[[1, self.nz - 1]] = True

        kperp_mask = kx[:, None] ** 2 + ky[None, :] ** 2 < kc2

        return (
            kz_mask[:, None, None]
            & kperp_mask[None, :, :]
            & self.get_solved_grid_mask().astype(bool)
        )

    def _setup_streamer(self) -> None:
        """
        Chooses the streamer's ky from [parameters.streamer] and reports the
        linear properties of the streamer mode.
        """
        dky = self.ky[1]
        kz = 2.0 * np.pi / self.input["dimensions.Lz"]

        ky_marg = self._find_marginal_ky()
        ky_peak = self._find_peak_ky(ky_marg)

        if self.input["parameters.streamer.iky"] > 0:
            iky = self.input["parameters.streamer.iky"]
        elif self.input["parameters.streamer.ky"] > 0.0:
            iky = int(np.rint(self.input["parameters.streamer.ky"] / dky))
        elif self.input["parameters.streamer.ky_over_kmarg"] > 0.0:
            ky_target = self.input["parameters.streamer.ky_over_kmarg"] * ky_marg
            iky = int(np.rint(ky_target / dky))
        else:
            ky_target = self.input["parameters.streamer.ky_over_kpeak"] * ky_peak
            iky = int(np.rint(ky_target / dky))

        solved_grid_mask = self.get_solved_grid_mask().astype(bool)
        if not (0 < iky < self.half_ny and solved_grid_mask[1, 0, iky]):
            raise InvalidFlucsInputFileError(
                f"Streamer index iky = {iky} is not a solved mode with ky > 0."
            )

        self.streamer_iky = iky
        self.streamer_ky = self.get_streamer_ky(iky)
        self.streamer_kc = (
            self.input["parameters.streamer.cutoff_factor"] * self.streamer_ky
        )
        self.streamer_kc2 = self.get_streamer_kc2(self.streamer_kc)

        omega, ratio = self.compute_most_unstable_mode_cpu(kz, 0.0, self.streamer_ky)
        omega_hyper, _ = self.compute_most_unstable_mode_cpu(
            kz, 0.0, self.streamer_ky, include_hyperdissipation=True
        )
        gamma = float(omega.imag)

        if self.input["parameters.streamer.amplitude_mode"] == "predicted":
            self.streamer_dT_amplitude = (
                self.input["parameters.streamer.C_T"] * gamma / self.streamer_ky
            )
        else:
            self.streamer_dT_amplitude = self.input["parameters.streamer.dT_amplitude"]

        # Seed energy, prediction (a) of paper_figures.ipynb: in curvy3d units
        # W_II = C_W gamma_lin dT_I^(2/3) nu_perp^(1/3) / ky_I^(4/3); flucs'
        # W is half of curvy3d's
        if self.input["parameters.streamer.seed_mode"] == "predicted":
            self.streamer_seed_W = (
                0.5
                * self.input["parameters.streamer.C_W"]
                * gamma
                * self.streamer_dT_amplitude ** (2.0 / 3.0)
                * self.input["hyperdissipation.kperp"] ** (1.0 / 3.0)
                / self.streamer_ky ** (4.0 / 3.0)
            )

        if float(omega_hyper.imag) <= 0.0:
            raise InvalidFlucsInputFileError(
                f"The streamer mode at ky = {self.streamer_ky:.4e} is linearly "
                f"stable (gamma_lin = {gamma:.4e}, with hyperdissipation "
                f"{float(omega_hyper.imag):.4e}; unstable for ky > "
                f"{ky_marg:.4e} until hyperdissipation cuts it off)."
            )

        flucsprint(
            f"Streamer ({self.streamer_mode}): iky = {iky}, "
            f"ky = {self.streamer_ky:.4e}, ky_marg = {ky_marg:.4e} "
            f"(ky/ky_marg = {self.streamer_ky / ky_marg:.3f}), "
            f"ky_peak = {ky_peak:.4e} (ky/ky_peak = "
            f"{self.streamer_ky / ky_peak:.3f}), kc = {self.streamer_kc:.4e}\n"
            f"Streamer linear mode at kx = 0, kz = 2 pi/Lz: "
            f"gamma_lin = {gamma:.4e} (with hyperdissipation "
            f"{float(omega_hyper.imag):.4e}), omega = {float(omega.real):.4e}, "
            f"phi/T = {abs(ratio):.4f} exp({np.angle(ratio):.4f} i)\n"
            f"Streamer amplitude: dT_I = {self.streamer_dT_amplitude:.4e} "
            f"({self.input['parameters.streamer.amplitude_mode']})"
        )

    def _set_streamer_initial_conditions(self) -> None:
        """
        Overwrites fields_initial with the streamer plus a set-II seed.

        Amplitude convention. Fields are forward-normalised Fourier
        coefficients, f(r) = sum_k f_k exp(i k.r), over all k. Only ky >= 0 is
        stored, and every stored mode with ky > 0 stands for itself and its
        conjugate at -k, so the real-space mean square is

            <dT^2> = sum_{ky = 0} |T_k|^2 + 2 sum_{ky > 0} |T_k|^2.

        The streamer occupies the set-I harmonics at ky = ky_I, kz = +-2 pi/Lz
        (n_kz = 2 for kz_structure = "both", 1 for "plus") and kx_j with
        kperp < kc, with amplitudes T(kz, kx_j) = a g_j exp(i theta_kz), where
        g_j = exp(-kx_j^2 / (2 kx_width^2)) (g = delta_{kx, 0} for
        kx_width = 0) and theta_{+} = 0, theta_{-} = kz_phase. Hence

            dT_I^2 = <dT_I^2> = 2 n_kz a^2 sum_j g_j^2,

        e.g. a = dT_I/2 for a single kx = 0 harmonic at both kz. Each harmonic
        gets the phi/T ratio of its own most unstable eigenvector.

        The seed puts noise on every solved set-II mode, with random phases,
        equal |phi| and |T| at each mode and conjugate symmetry at ky = 0. It
        is flat in (kx, ky) within each kz plane, and its free energy per
        folded parallel harmonic m (kz = +-2 pi m/Lz together) is

            W_II(m = 0) = f_0 W_II,
            W_II(m) = (1 - f_0) W_II m^p / sum_m' m'^p   for m >= 1,

        with f_0 = seed_kz0_fraction and p = seed_kz_power. The total is
        W_II = (C_W/2) gamma_lin dT_I^(2/3) nu_perp^(1/3) / ky_I^(4/3) for
        seed_mode = "predicted" (C_W in curvy3d units, whose W is twice
        flucs'), or set by <dT_II^2>^(1/2) = seed_fraction * dT_I for
        seed_mode = "fraction".
        """
        streamer = "parameters.streamer"
        iky = self.streamer_iky
        kz = 2.0 * np.pi / self.input["dimensions.Lz"]
        set_I = self.get_streamer_mask(self.streamer_kc2)

        # Streamer harmonics: set-I kx at ky = ky_I
        ikx = np.nonzero(set_I[1, :, iky])[0]
        kx = self.kx[ikx]

        kx_width = self.input[f"{streamer}.kx_width"]
        if kx_width > 0.0:
            envelope = np.exp(-(kx**2) / (2.0 * kx_width**2))
        else:
            envelope = (ikx == 0).astype(float)

        if self.input[f"{streamer}.kz_structure"] == "both":
            kz_components = {1: 0.0, self.nz - 1: self.input[f"{streamer}.kz_phase"]}
        else:
            kz_components = {1: 0.0}

        amplitude = self.streamer_dT_amplitude / np.sqrt(
            2.0 * len(kz_components) * np.sum(envelope**2)
        )

        _, ratio = self.compute_most_unstable_mode_cpu(kz, kx, self.ky[iky])

        fields_initial = np.zeros(
            (self.number_of_fields, *self.half_tuple), dtype=self.complex
        )
        for ikz, phase in kz_components.items():
            T = amplitude * envelope * np.exp(1j * phase)
            fields_initial[0, ikz, ikx, iky] = ratio * T
            fields_initial[1, ikz, ikx, iky] = T

        # Seed: white noise on set II
        set_II = self.get_solved_grid_mask().astype(bool) & ~set_I
        random = np.random.default_rng(self.input[f"{streamer}.rand_seed"])
        seed = np.exp(
            2j * np.pi * random.random((self.number_of_fields, *self.half_tuple))
        )
        seed[:, ~set_II] = 0.0

        # Conjugate symmetry at ky = 0 (the set-II mask is symmetric there)
        conjugate_ikz = (-np.arange(self.nz)) % self.nz
        conjugate_ikx = (-np.arange(self.nx)) % self.nx
        seed_ky0 = seed[:, :, :, 0]
        seed_ky0[:] = 0.5 * (
            seed_ky0
            + np.conj(seed_ky0[:, conjugate_ikz[:, None], conjugate_ikx[None, :]])
        )

        # Free energy per mode, counting the conjugate modes at ky > 0
        taubar = self.input["parameters.tratio"] / self.input["parameters.charge"]
        ky_weight = np.where(np.arange(self.half_ny) > 0, 2.0, 1.0)

        def free_energy(fields):
            return ky_weight * (
                (1.0 + 1.0 / taubar) / (2.0 * taubar) * np.abs(fields[0]) ** 2
                + 0.75 * np.abs(fields[1]) ** 2
            )

        # Shape: flat in (kx, ky), m^p across the folded parallel harmonics
        m = np.abs(np.rint(np.fft.fftfreq(self.nz) * self.nz)).astype(int)
        seed_energy = free_energy(seed)
        harmonics = [h for h in np.unique(m) if seed_energy[m == h].sum() > 0.0]
        kz0_fraction = self.input[f"{streamer}.seed_kz0_fraction"]
        weights = {
            h: float(h) ** self.input[f"{streamer}.seed_kz_power"]
            for h in harmonics
            if h > 0
        }
        norm = sum(weights.values())
        if 0 in harmonics:
            # Without kz = 0 modes, or without any other, one side takes all
            kz0 = kz0_fraction if weights else 1.0
            weights = {h: (1.0 - kz0) * w / norm for h, w in weights.items()}
            weights[0] = kz0
        else:
            weights = {h: w / norm for h, w in weights.items()}

        for h, weight in weights.items():
            seed[:, m == h] *= np.sqrt(weight / seed_energy[m == h].sum())

        # Total
        if self.input[f"{streamer}.seed_mode"] == "predicted":
            seed_W = self.streamer_seed_W
        else:
            seed_dT_rms = np.sqrt(np.sum(np.abs(seed[1]) ** 2 * ky_weight))
            seed_W = (
                self.input[f"{streamer}.seed_fraction"]
                * self.streamer_dT_amplitude
                / seed_dT_rms
            ) ** 2
        seed *= np.sqrt(seed_W)

        streamer_W = float(np.sum(free_energy(fields_initial)))
        seed_W = float(np.sum(free_energy(seed)))
        seed_kz0 = float(np.sum(free_energy(seed)[m == 0])) / seed_W if seed_W else 0.0

        self.fields_initial = fields_initial + seed.astype(self.complex)
        self.streamer_initialised = True

        flucsprint(
            f"Streamer initial condition: {np.count_nonzero(envelope)} kx "
            f"harmonic(s) x "
            f"{len(kz_components)} kz, Fourier amplitude |T| = {amplitude:.4e} "
            f"at kx = 0, W_I = {streamer_W:.4e}\n"
            f"Set-II seed ({self.input[f'{streamer}.seed_mode']}): "
            f"W_II = {seed_W:.4e}, W_II/W_I = {seed_W / streamer_W:.3e}, "
            f"kz = 0 fraction {seed_kz0:.3f}, kz spectrum ~ |m|^"
            f"{self.input[f'{streamer}.seed_kz_power']:g}"
        )

    def _check_streamer_amplitude(self) -> None:
        """
        Measures the real-space RMS of dT over set I and over set II of the
        initial fields on the GPU.
        """
        set_I = cp.asarray(self.get_streamer_mask(self.streamer_kc2))
        T = self.get_fields()[1]

        def realspace_rms(T_hat):
            T_real = cp.fft.irfftn(T_hat, s=self.full_tuple, norm="forward")
            rms = float(cp.sqrt(cp.mean(T_real**2)))
            del T_real
            return rms

        rms_I = realspace_rms(cp.where(set_I, T, 0))
        rms_II = realspace_rms(cp.where(set_I, 0, T))

        # Free energy of set II and its kz = 0 share
        taubar = self.input["parameters.tratio"] / self.input["parameters.charge"]
        fields = self.get_fields()
        ky_weight = cp.where(cp.arange(self.half_ny) > 0, 2.0, 1.0)
        W = ky_weight * (
            (1.0 + 1.0 / taubar) / (2.0 * taubar) * cp.abs(fields[0]) ** 2
            + 0.75 * cp.abs(fields[1]) ** 2
        )
        W_II = float(cp.sum(cp.where(set_I, 0, W)))
        W_II_kz0 = float(cp.sum(W[0]))

        flucsprint(
            f"Initial real-space RMS: dT_I = {rms_I:.6e} "
            f"(requested {self.streamer_dT_amplitude:.6e}, "
            f"relative error {rms_I / self.streamer_dT_amplitude - 1:.2e}), "
            f"dT_II = {rms_II:.6e}; W_II = {W_II:.6e}, "
            f"kz = 0 fraction {W_II_kz0 / W_II if W_II else 0.0:.4f}"
        )
