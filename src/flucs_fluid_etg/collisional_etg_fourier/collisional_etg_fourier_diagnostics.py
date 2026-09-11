from collections.abc import Callable
from typing import ClassVar

import cupy as cp
from flucs.diagnostic import FlucsDiagnostic, FlucsDiagnosticVariable
from flucs.solvers.fourier.fourier_system_reductions import FourierReductions
from flucs.utilities.cupy import KernelWrapper


class HeatfluxDiag(FlucsDiagnostic):
    name = "heatflux"

    get_heatflux: Callable[..., cp.ndarray]

    def init_vars(self):
        reductions = FourierReductions(self.system)

        # Diagnostic variables
        self.add_var(
            FlucsDiagnosticVariable(
                name="heatflux", shape=(), dimensions={}, is_complex=False
            )
        )

        # Reductions
        self.get_heatflux = reductions.get_reduction(
            reduction_output="scalar",
            functor="Heatflux_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )

    def ready(self):
        pass

    def execute(self):
        fields = self.system.get_fields()

        self.vars["heatflux"].data_cache.append(self.get_heatflux(fields).get().item())


class FreeEnergyDiag(FlucsDiagnostic):
    name = "free_energy"

    get_W: Callable[..., cp.ndarray]
    get_heatflux: Callable[..., cp.ndarray]
    get_dWdt_coll: Callable[..., cp.ndarray]
    get_dWdt_hyperdissipation_component: Callable[..., cp.ndarray]

    def init_vars(self):
        reductions = FourierReductions(self.system)

        # Diagnostic variables
        self.add_var(
            FlucsDiagnosticVariable(name="W", shape=(), dimensions={}, is_complex=False)
        )
        self.add_var(
            FlucsDiagnosticVariable(
                name="dWdt", shape=(), dimensions={}, is_complex=False
            )
        )
        self.add_var(
            FlucsDiagnosticVariable(
                name="dWdt_coll", shape=(), dimensions={}, is_complex=False
            )
        )
        self.add_var(
            FlucsDiagnosticVariable(
                name="dWdt_inj", shape=(), dimensions={}, is_complex=False
            )
        )
        self.add_var(
            FlucsDiagnosticVariable(
                name="dWdt_error", shape=(), dimensions={}, is_complex=False
            )
        )

        for component in self.system.hyperdissipation_components:
            self.add_var(
                FlucsDiagnosticVariable(
                    name=f"dWdt_hyperdissipation_{component}",
                    shape=(),
                    dimensions={},
                    is_complex=False,
                )
            )

        # Reductions
        self.get_W = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergy_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )
        self.get_heatflux = reductions.get_reduction(
            reduction_output="scalar",
            functor="Heatflux_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )
        self.get_dWdt_coll = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergyColl_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )
        self.get_dWdt_hyperdissipation_component = reductions.get_reduction(
            reduction_output="scalar",
            functor="FreeEnergyHyperdissipationComponent_Functor",
            input_args="FLUCS_COMPLEX*,FLUCS_FLOAT,int",
            complex_output=False,
        )

    def ready(self):
        pass

    def execute(self):
        current_dt = self.system.float(self.system.current_dt)
        adaptive_rate = self.system.float(self.system.adaptive_rate)

        # fields
        fields = self.system.get_fields()
        fields_prev = self.system.get_fields(1)

        # W
        W = self.get_W(fields).get().item()
        self.save_data("W", W)

        # dWdt
        W_prev = self.get_W(fields_prev).get().item()
        dWdt = (W - W_prev) / current_dt
        self.save_data("dWdt", dWdt)

        # dWdt_coll
        dWdt_coll = -self.get_dWdt_coll(fields).get().item()
        self.save_data("dWdt_coll", dWdt_coll)

        # dWdt_inj
        heatflux = self.get_heatflux(fields).get().item()
        dWdt_inj = self.system.input["parameters.kappaT"] * heatflux
        self.save_data("dWdt_inj", dWdt_inj)

        # dWdt_hyperdissipation
        dWdt_hyperdissipation_total = 0.0
        for index, component in enumerate(self.system.hyperdissipation_components):
            result = self.get_dWdt_hyperdissipation_component(
                fields, adaptive_rate, index
            )

            dWdt_hyperdissipation_component = -result.get().item()
            self.save_data(
                f"dWdt_hyperdissipation_{component}", dWdt_hyperdissipation_component
            )
            dWdt_hyperdissipation_total += dWdt_hyperdissipation_component

        # dWdt_error
        self.save_data(
            "dWdt_error",
            dWdt - dWdt_inj - dWdt_coll - dWdt_hyperdissipation_total,
        )


class SpectralFreeEnergyDiag(FlucsDiagnostic):
    """Save contracted free-energy exchange terms.

    The diagnostic saves ``W``, ``dWdt``, temperature-gradient injection,
    parallel collisional dissipation, and all hyperdissipation components
    clumped into one term.  The nonlinear exchange is intentionally left for
    postprocessing as ``dWdt - injection - parallel - hyperdissipation``.

    Supported contractions are ``kx``, ``ky``, ``kz``, ``kperp``, ``kxky`` and
    ``kxkykz``.  The last one is the unreduced Fourier-space half grid.
    """

    name = "spectral_free_energy"
    option_defaults: ClassVar[dict[str, object]] = {
        "spectra": ["kx", "ky", "kz", "kperp"]
    }

    _term_names = (
        "W",
        "dWdt",
        "dWdt_inj",
        "dWdt_coll",
        "dWdt_hyperdissipation",
    )

    def _full_reduction(self, functor, input_args):
        output = self.system.get_temp_array(
            self.system.half_size, is_complex=False
        ).reshape(self.system.half_tuple)
        kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name=(
                f"spectral_pointwise<FLUCS_FLOAT,{functor},{input_args}>"
            ),
            grid=(self.system.half_cuda_grid_size,),
            block=(self.system.cuda_block_size,),
        )

        def reduction(*args):
            kernel(output, *args)
            return output

        return reduction

    def init_vars(self):
        reductions = FourierReductions(self.system)
        valid_spectra = ("kx", "ky", "kz", "kperp", "kxky", "kxkykz")
        spectra = self.spectra
        invalid = set(spectra) - set(valid_spectra)
        if invalid:
            raise ValueError(
                f"{self.name} supports contractions {valid_spectra}; "
                f"invalid values: {sorted(invalid)}."
            )

        self.reductions = {}
        for spectrum in dict.fromkeys(spectra):
            if spectrum == "kxkykz":
                dimensions = {
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                }
                get_reduction = self._full_reduction
            else:
                dimensions = reductions.get_dimensions(spectrum)
                get_reduction = reductions.get_reduction

            shape = tuple(dimensions)
            for name in self._term_names:
                self.add_var(
                    FlucsDiagnosticVariable(
                        name=f"{spectrum}_spectra/{name}",
                        shape=shape,
                        dimensions=dimensions,
                        is_complex=False,
                    )
                )

            self.reductions[spectrum] = {
                "W": get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergy_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False,
                )
                if spectrum != "kxkykz"
                else get_reduction("FreeEnergy_Functor", "FLUCS_COMPLEX*"),
                "inj": get_reduction(
                    reduction_output=spectrum,
                    functor="Heatflux_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False,
                )
                if spectrum != "kxkykz"
                else get_reduction("Heatflux_Functor", "FLUCS_COMPLEX*"),
                "coll": get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyColl_Functor",
                    input_args="FLUCS_COMPLEX*",
                    complex_output=False,
                )
                if spectrum != "kxkykz"
                else get_reduction("FreeEnergyColl_Functor", "FLUCS_COMPLEX*"),
                "hyper": get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyHyperdissipation_Functor",
                    input_args="FLUCS_COMPLEX*,FLUCS_FLOAT",
                    complex_output=False,
                )
                if spectrum != "kxkykz"
                else get_reduction(
                    "FreeEnergyHyperdissipation_Functor",
                    "FLUCS_COMPLEX*,FLUCS_FLOAT",
                ),
            }

    def ready(self):
        pass

    def execute(self):
        current_dt = self.system.float(self.system.current_dt)
        adaptive_rate = self.system.float(self.system.adaptive_rate)
        fields = self.system.get_fields()
        fields_prev = self.system.get_fields(1)
        kappaT = self.system.input["parameters.kappaT"]

        for spectrum, reductions in self.reductions.items():
            W = reductions["W"](fields)
            W_prev = reductions["W"](fields_prev)
            self.save_data(f"{spectrum}_spectra/W", W.get())
            self.save_data(
                f"{spectrum}_spectra/dWdt",
                ((W - W_prev) / current_dt).get(),
            )
            self.save_data(
                f"{spectrum}_spectra/dWdt_inj",
                (kappaT * reductions["inj"](fields)).get(),
            )
            self.save_data(
                f"{spectrum}_spectra/dWdt_coll",
                (-reductions["coll"](fields)).get(),
            )
            self.save_data(
                f"{spectrum}_spectra/dWdt_hyperdissipation",
                (-reductions["hyper"](fields, adaptive_rate)).get(),
            )


class SpectraDiag(FlucsDiagnostic):
    """Compute spectra of ``abs(phi)**2`` and ``abs(T)**2``.

    ``spectra`` is a list of Fourier contractions.  The supported one- and
    two-dimensional contractions are ``kx``, ``ky``, ``kz``, ``kperp`` and
    ``kxky``.  ``save_3d`` additionally saves the unreduced Fourier-space
    arrays on the real-to-complex half grid, with dimensions ``kz, kx, ky``.

    For example::

        {name = "spectra", options = {
            spectra = ["kx", "ky", "kz", "kperp", "kxky"],
            save_3d = true,
        }}

    Each requested contraction is saved in its own subgroup, for example
    ``kx_spectra/phi2`` and ``kx_spectra/T2``.
    """

    name = "spectra"
    option_defaults: ClassVar[dict[str, object]] = {
        "spectra": ["kx", "ky", "kz", "kperp"],
        "save_2d": False,
        "save_3d": False,
    }

    get_phi2: dict[str, Callable[..., cp.ndarray]]
    get_T2: dict[str, Callable[..., cp.ndarray]]

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)
        valid_spectra = ("kx", "ky", "kz", "kperp", "kxky")
        spectra = [self.spectra] if isinstance(self.spectra, str) else self.spectra
        spectra = list(dict.fromkeys(spectra))
        if self.save_2d and "kxky" not in spectra:
            spectra.append("kxky")
        spectra = tuple(spectra)

        invalid = set(spectra) - set(valid_spectra)
        if invalid:
            raise ValueError(
                f"{self.name} supports contractions {valid_spectra}; "
                f"invalid values: {sorted(invalid)}."
            )
        if not spectra and not self.save_3d:
            raise ValueError("At least one spectrum or save_3d must be enabled.")

        self.get_phi2 = {}
        self.get_T2 = {}
        for spectrum in spectra:
            dimensions = reductions.get_dimensions(spectrum)
            shape = tuple(dimensions)
            for name in ("phi2", "T2"):
                self.add_var(
                    FlucsDiagnosticVariable(
                        name=f"{spectrum}_spectra/{name}",
                        shape=shape,
                        dimensions=dimensions,
                        is_complex=False,
                    )
                )

            self.get_phi2[spectrum] = reductions.get_reduction(
                reduction_output=spectrum,
                functor="PhiSquared_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )
            self.get_T2[spectrum] = reductions.get_reduction(
                reduction_output=spectrum,
                functor="TSquared_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            )

        if self.save_3d:
            dimensions = {
                "kz": self.system.kz,
                "kx": self.system.kx,
                "ky": self.system.ky,
            }
            for name in ("phi2", "T2"):
                self.add_var(
                    FlucsDiagnosticVariable(
                        name=f"full_spectra/{name}",
                        shape=("kz", "kx", "ky"),
                        dimensions=dimensions,
                        is_complex=False,
                    )
                )

    def ready(self) -> None:
        pass

    def execute(self) -> None:
        fields = self.system.get_fields()
        for spectrum in self.get_phi2:
            self.save_data(
                f"{spectrum}_spectra/phi2", self.get_phi2[spectrum](fields).get()
            )
            self.save_data(
                f"{spectrum}_spectra/T2", self.get_T2[spectrum](fields).get()
            )

        if self.save_3d:
            self.save_data("full_spectra/phi2", (cp.abs(fields[0]) ** 2).get())
            self.save_data("full_spectra/T2", (cp.abs(fields[1]) ** 2).get())
