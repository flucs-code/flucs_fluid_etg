from collections.abc import Callable
from typing import ClassVar

import cupy as cp
import numpy as np
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
    clumped into one term, and the directly evaluated nonlinear exchange.

    Supported contractions are ``kx``, ``ky``, ``kz``, ``kperp``, ``kzkperp``,
    ``kxky`` and ``kxkykz``.  The last one is the unreduced Fourier-space half
    grid.
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
        "dWdt_nonlinear",
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
            grid=(
                getattr(
                    self.system,
                    "half_cuda_grid_size",
                    (self.system.half_size + self.system.cuda_block_size - 1)
                    // self.system.cuda_block_size,
                ),
            ),
            block=(self.system.cuda_block_size,),
        )

        def reduction(*args):
            kernel(output, *args)
            return output

        return reduction

    def init_vars(self):
        reductions = FourierReductions(self.system)
        valid_spectra = (
            "kx",
            "ky",
            "kz",
            "kperp",
            "kzkperp",
            "kxky",
            "kxkykz",
        )
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
                "nonlinear": get_reduction(
                    reduction_output=spectrum,
                    functor="FreeEnergyNonlinear_Functor",
                    input_args=(
                        "FLUCS_COMPLEX*,FLUCS_FLOAT,FLUCS_FLOAT,long long,"
                        "const FLUCS_COMPLEX (*)[HALFSIZE]"
                    ),
                    complex_output=False,
                )
                if spectrum != "kxkykz" and not self.system.input["setup.linear"]
                else (
                    get_reduction(
                        "FreeEnergyNonlinear_Functor",
                        "FLUCS_COMPLEX*,FLUCS_FLOAT,FLUCS_FLOAT,long long,"
                        "const FLUCS_COMPLEX (*)[HALFSIZE]",
                    )
                    if not self.system.input["setup.linear"]
                    else None
                ),
            }

    def ready(self):
        pass

    def execute(self):
        current_dt = self.system.float(self.system.current_dt)
        current_time = self.system.float(self.system.current_time)
        current_step = self.system.int(self.system.current_step)
        adaptive_rate = self.system.float(self.system.adaptive_rate)
        fields = self.system.get_fields()
        fields_prev = self.system.get_fields(1)
        kappaT = self.system.input["parameters.kappaT"]
        if not self.system.input["setup.linear"]:
            self.system.compute_nonlinear_terms(
                current_dt, current_time, current_step, fields, False
            )
            dft_bits = self.system.dft_bits

        for spectrum, reductions in self.reductions.items():
            # Reductions of the same shape share their output array, so copy
            # W to the host before computing W_prev
            W = reductions["W"](fields).get()
            W_prev = reductions["W"](fields_prev).get()
            self.save_data(f"{spectrum}_spectra/W", W)
            self.save_data(f"{spectrum}_spectra/dWdt", (W - W_prev) / current_dt)
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
            dWdt_nonlinear = (
                reductions["nonlinear"](
                    fields, current_dt, current_time, current_step, dft_bits
                ).get()
                if reductions["nonlinear"] is not None
                else np.zeros_like(W)
            )
            self.save_data(f"{spectrum}_spectra/dWdt_nonlinear", dWdt_nonlinear)


class SpectraDiag(FlucsDiagnostic):
    """Compute spectra of ``abs(phi)**2`` and ``abs(T)**2``.

    ``spectra`` is a list of Fourier contractions.  The supported one- and
    two-dimensional contractions are ``kx``, ``ky``, ``kz``, ``kperp``,
    ``kzkperp``, ``kxky`` and ``kxkykz``.  The latter is the unreduced
    Fourier-space half grid, with dimensions ``kz, kx, ky``.  ``save_3d``
    additionally saves the same data under ``full_spectra``.

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

    def _full_reduction(self, functor, input_args):
        output = self.system.get_temp_array(
            self.system.half_size, is_complex=False
        ).reshape(self.system.half_tuple)
        kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name=(
                f"spectral_pointwise<FLUCS_FLOAT,{functor},{input_args}>"
            ),
            grid=(
                getattr(
                    self.system,
                    "half_cuda_grid_size",
                    (self.system.half_size + self.system.cuda_block_size - 1)
                    // self.system.cuda_block_size,
                ),
            ),
            block=(self.system.cuda_block_size,),
        )

        def reduction(*args):
            kernel(output, *args)
            return output

        return reduction

    def init_vars(self) -> None:
        reductions = FourierReductions(self.system)
        valid_spectra = (
            "kx",
            "ky",
            "kz",
            "kperp",
            "kzkperp",
            "kxky",
            "kxkykz",
        )
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
            for name in ("phi2", "T2"):
                self.add_var(
                    FlucsDiagnosticVariable(
                        name=f"{spectrum}_spectra/{name}",
                        shape=shape,
                        dimensions=dimensions,
                        is_complex=False,
                    )
                )

            self.get_phi2[spectrum] = get_reduction(
                reduction_output=spectrum,
                functor="PhiSquared_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            ) if spectrum != "kxkykz" else get_reduction(
                "PhiSquared_Functor", "FLUCS_COMPLEX*"
            )
            self.get_T2[spectrum] = get_reduction(
                reduction_output=spectrum,
                functor="TSquared_Functor",
                input_args="FLUCS_COMPLEX*",
                complex_output=False,
            ) if spectrum != "kxkykz" else get_reduction(
                "TSquared_Functor", "FLUCS_COMPLEX*"
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



class StreamerExchangeDiag(FlucsDiagnostic):
    """Free-energy exchange between the streamer (set I) and the fuzz (set II).

    Set I holds the modes with kz = +-2 pi/Lz and kperp < kc, where
    kc = cutoff_factor * ky_I and ky_I is the streamer's ky; set II holds every
    other solved mode. The fields are split with these Fourier-space masks and,
    for a, b in {I, II}, N_ab is the nonlinear term of the dT equation with
    phi -> phi_a and T -> T_b. It is evaluated with the system's own dealiased
    operation (two-thirds or phase-shift), so that sum_ab N_ab = N.

    The transfer into destination d is

        T_{d,ab}(k) = -mask_d(k) (3/2) Re[conj(T_k) N_ab(k)],

    i.e. FreeEnergyNonlinear_Functor restricted to the split fields, with the
    flucs normalisation W = (1 + 1/taubar)/(2 taubar) |phi|^2 + (3/4) |T|^2.
    add_nonlinear_terms builds N = dy(dxphi T) - dx(dyphi T) = {phi, T} and
    the timesteppers advance dfields/dt = -M fields - explicit_terms, so flucs
    evolves dT/dt = ... - {phi, T} and T_{d,ab} > 0 means that {phi_a, T_b}
    feeds set d. These are the curvy3d equations, whose W is twice this one,
    so on identical fields every term is half of curvy3d's
    -3 Re[conj(T_k) {phi_a, T_b}_k].

    The eight terms are keyed as in curvy3d's NL_EXCHANGE_TERMS,
    key: (destination, phi part, T part). Since sum_k Re[conj(T_d) N_ab] is
    antisymmetric under d <-> b, summed over k
    nl_streamer_to_small = -nl_small_to_streamer and
    nl_streamer_self_to_small = -nl_small_to_streamer_via_streamer_flow, while
    the terms with d = b (nl_batchelor, nl_small_flow_on_streamer,
    nl_small_self and nl_streamer_self) vanish.

    Saved scalars: the eight terms, W_I, W_II, their time derivatives (from
    the previous step), and the linear budget per set: injection
    (kappaT * heat flux), collisional (D_par) and hyperdissipation (D_perp)
    terms, so that dW_d/dt = inj_d + coll_d + hyper_d + sum_ab T_{d,ab}. In
    frozen streamer runs the hold is an extra, unsaved, source for set I.
    Also saved: the iky_streamer, ky_streamer and kperp_cutoff used.

    Saved spectra: for every requested contraction (``kx``, ``ky``, ``kz``,
    ``kperp``, ``kzkperp``, ``kxky``, ``kxkykz``), the eight terms and W_I,
    W_II under ``<contraction>_spectra/<term>``. The ``kxky`` contraction also
    holds the set-II budget terms dWdt_inj_II, dWdt_coll_II and
    dWdt_hyperdissipation_II. Contractions follow FourierReductions: kxky and
    kxkykz are plain sums over the stored half grid (no factor 2 for ky > 0),
    the others count the conjugate modes.

    The streamer's ky index is set by ``streamer_iky``: a positive value fixes
    it, -1 picks the peak over ky of sum_{kx, kz=+-1} |phi|^2, smoothed by an
    exponential running average with weight ``auto_smoothing`` per save and
    switching only when a new peak exceeds the current one by the factor
    ``auto_hysteresis``. The default, 0, uses [parameters.streamer] when it is
    enabled and the automatic choice otherwise. ``cutoff_factor`` < 0 uses
    parameters.streamer.cutoff_factor.

    For example::

        {name = "streamer_exchange", options = {
            spectra = ["kx", "ky", "kz", "kperp", "kxky"],
        }}
    """

    name = "streamer_exchange"
    option_defaults: ClassVar[dict[str, object]] = {
        "spectra": ["kx", "ky", "kz", "kperp", "kxky"],
        "streamer_iky": 0,
        "cutoff_factor": -1.0,
        "auto_smoothing": 0.2,
        "auto_hysteresis": 1.1,
    }

    # key: (destination, phi part, T part), with 0 = set I and 1 = set II
    exchange_terms: ClassVar[dict[str, tuple[int, int, int]]] = {
        "nl_streamer_to_small": (1, 1, 0),
        "nl_small_to_streamer": (0, 1, 1),
        "nl_batchelor": (1, 0, 1),
        "nl_small_self": (1, 1, 1),
        "nl_streamer_self": (0, 0, 0),
        "nl_streamer_self_to_small": (1, 0, 0),
        "nl_small_to_streamer_via_streamer_flow": (0, 0, 1),
        "nl_small_flow_on_streamer": (0, 1, 0),
    }
    part_names = ("I", "II")

    _part_args = "FLUCS_COMPLEX*,double,int"
    _hyper_args = "FLUCS_COMPLEX*,FLUCS_FLOAT,double,int"
    _nonlinear_args = (
        "FLUCS_COMPLEX*,FLUCS_FLOAT,FLUCS_FLOAT,long long,"
        "const FLUCS_COMPLEX (*)[HALFSIZE],double,int"
    )

    def _half_grid_size(self) -> int:
        # The system's CUDA grid sizes are only set when compiling
        return (
            self.system.half_size + self.system.cuda_block_size - 1
        ) // self.system.cuda_block_size

    def _full_reduction(self, functor, input_args):
        output = self.system.get_temp_array(
            self.system.half_size, is_complex=False
        ).reshape(self.system.half_tuple)
        kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name=(
                f"spectral_pointwise<FLUCS_FLOAT,{functor},{input_args}>"
            ),
            grid=(self._half_grid_size(),),
            block=(self.system.cuda_block_size,),
        )

        def reduction(*args):
            kernel(output, *args)
            return output

        return reduction

    def _get_reduction(self, spectrum, functor, input_args):
        if spectrum == "kxkykz":
            return self._full_reduction(functor, input_args)

        return self.reductions_factory.get_reduction(
            reduction_output=spectrum,
            functor=functor,
            input_args=input_args,
            complex_output=False,
        )

    def init_vars(self) -> None:
        self.reductions_factory = FourierReductions(self.system)
        valid_spectra = ("kx", "ky", "kz", "kperp", "kzkperp", "kxky", "kxkykz")
        spectra = [self.spectra] if isinstance(self.spectra, str) else self.spectra
        self.spectra = tuple(dict.fromkeys(spectra))

        invalid = set(self.spectra) - set(valid_spectra)
        if invalid:
            raise ValueError(
                f"{self.name} supports contractions {valid_spectra}; "
                f"invalid values: {sorted(invalid)}."
            )

        self.nonlinear = not self.system.input["setup.linear"]

        # Scalars
        scalar_names = [
            *self.exchange_terms,
            *(
                f"{name}_{part}"
                for name in (
                    "W",
                    "dWdt",
                    "dWdt_inj",
                    "dWdt_coll",
                    "dWdt_hyperdissipation",
                )
                for part in self.part_names
            ),
            "iky_streamer",
            "ky_streamer",
            "kperp_cutoff",
        ]
        for name in scalar_names:
            self.add_var(
                FlucsDiagnosticVariable(
                    name=name, shape=(), dimensions={}, is_complex=False
                )
            )

        # Spectra
        for spectrum in self.spectra:
            if spectrum == "kxkykz":
                dimensions = {
                    "kz": self.system.kz,
                    "kx": self.system.kx,
                    "ky": self.system.ky,
                }
            else:
                dimensions = self.reductions_factory.get_dimensions(spectrum)

            names = [*self.exchange_terms, "W_I", "W_II"]
            if spectrum == "kxky":
                names += [
                    "dWdt_inj_II",
                    "dWdt_coll_II",
                    "dWdt_hyperdissipation_II",
                ]

            for name in names:
                self.add_var(
                    FlucsDiagnosticVariable(
                        name=f"{spectrum}_spectra/{name}",
                        shape=tuple(dimensions),
                        dimensions=dimensions,
                        is_complex=False,
                    )
                )

        # Reductions
        self.reductions = {}
        for spectrum in ("scalar", *self.spectra):
            self.reductions[spectrum] = {
                "W": self._get_reduction(
                    spectrum, "FreeEnergyPart_Functor", self._part_args
                ),
                "nonlinear": self._get_reduction(
                    spectrum, "FreeEnergyNonlinearPart_Functor", self._nonlinear_args
                )
                if self.nonlinear
                else None,
            }

        for spectrum in ("scalar", "kxky"):
            if spectrum not in self.reductions:
                continue

            self.reductions[spectrum].update(
                {
                    "inj": self._get_reduction(
                        spectrum, "HeatfluxPart_Functor", self._part_args
                    ),
                    "coll": self._get_reduction(
                        spectrum, "FreeEnergyCollPart_Functor", self._part_args
                    ),
                    "hyper": self._get_reduction(
                        spectrum,
                        "FreeEnergyHyperdissipationPart_Functor",
                        self._hyper_args,
                    ),
                }
            )

        # Power at kz = +-2 pi/Lz per ky, for the automatic choice of ky_I
        self.get_phi2_kz1_ky = self.reductions_factory.get_reduction(
            reduction_output="ky",
            functor="PhiSquaredKzOne_Functor",
            input_args="FLUCS_COMPLEX*",
            complex_output=False,
        )

        # Splits the fields into (phi_a, T_b)
        self.split_fields_kernel = KernelWrapper(
            system=self.system,
            cuda_kernel_name="split_streamer_fields",
            grid=(self._half_grid_size(),),
            block=(self.system.cuda_block_size,),
        )

    def ready(self) -> None:
        system = self.system

        # Resolve the streamer's ky index and the cut-off
        self.fixed_iky = None
        if self.streamer_iky > 0:
            self.fixed_iky = self.streamer_iky
        elif self.streamer_iky == 0 and system.streamer_mode:
            self.fixed_iky = system.streamer_iky
        elif self.streamer_iky not in (0, -1):
            raise ValueError(
                f"{self.name}: streamer_iky must be positive, 0 or -1, "
                f"not {self.streamer_iky}."
            )

        if self.fixed_iky is not None and not 0 < self.fixed_iky < system.half_ny:
            raise ValueError(
                f"{self.name}: streamer_iky = {self.fixed_iky} is out of range."
            )

        if self.cutoff_factor < 0.0:
            self.cutoff_factor = system.input["parameters.streamer.cutoff_factor"]

        self.current_iky = self.fixed_iky
        self.ky_power = None

        if self.nonlinear:
            self.split_fields = cp.zeros_like(system.get_fields())

    def _update_streamer_iky(self, fields) -> int:
        """Automatic ky_I: smoothed peak of sum_{kx, kz=+-1} |phi|^2."""
        power = self.get_phi2_kz1_ky(fields).get()
        power[0] = 0.0

        if self.ky_power is None:
            self.ky_power = power
        else:
            self.ky_power = (
                1.0 - self.auto_smoothing
            ) * self.ky_power + self.auto_smoothing * power

        candidate = int(np.argmax(self.ky_power))
        if self.current_iky is None or (
            candidate != self.current_iky
            and self.ky_power[candidate]
            > self.auto_hysteresis * self.ky_power[self.current_iky]
        ):
            self.current_iky = candidate

        return self.current_iky

    def execute(self) -> None:
        system = self.system
        current_dt = system.float(system.current_dt)
        current_time = system.float(system.current_time)
        current_step = system.int(system.current_step)
        adaptive_rate = system.float(system.adaptive_rate)
        kappaT = system.input["parameters.kappaT"]

        fields = system.get_fields()
        fields_prev = system.get_fields(1)

        # Masks
        if self.fixed_iky is None:
            iky = self._update_streamer_iky(fields)
        else:
            iky = self.fixed_iky

        ky_streamer = system.get_streamer_ky(iky)
        kperp_cutoff = self.cutoff_factor * ky_streamer
        kc2 = np.float64(system.get_streamer_kc2(kperp_cutoff))

        self.save_data("iky_streamer", iky)
        self.save_data("ky_streamer", ky_streamer)
        self.save_data("kperp_cutoff", kperp_cutoff)

        # Free energy and linear budget per set. Reductions share their
        # output arrays, so each result is copied to the host right away.
        scalar = self.reductions["scalar"]
        for part, part_name in enumerate(self.part_names):
            part = np.int32(part)
            W = scalar["W"](fields, kc2, part).get().item()
            W_prev = scalar["W"](fields_prev, kc2, part).get().item()
            self.save_data(f"W_{part_name}", W)
            self.save_data(f"dWdt_{part_name}", (W - W_prev) / current_dt)
            self.save_data(
                f"dWdt_inj_{part_name}",
                kappaT * scalar["inj"](fields, kc2, part).get().item(),
            )
            self.save_data(
                f"dWdt_coll_{part_name}",
                -scalar["coll"](fields, kc2, part).get().item(),
            )
            self.save_data(
                f"dWdt_hyperdissipation_{part_name}",
                -scalar["hyper"](fields, adaptive_rate, kc2, part).get().item(),
            )

        for spectrum in self.spectra:
            reductions = self.reductions[spectrum]
            for part, part_name in enumerate(self.part_names):
                self.save_data(
                    f"{spectrum}_spectra/W_{part_name}",
                    reductions["W"](fields, kc2, np.int32(part)).get(),
                )

            if spectrum == "kxky":
                set_II = np.int32(1)
                self.save_data(
                    "kxky_spectra/dWdt_inj_II",
                    kappaT * reductions["inj"](fields, kc2, set_II).get(),
                )
                self.save_data(
                    "kxky_spectra/dWdt_coll_II",
                    -reductions["coll"](fields, kc2, set_II).get(),
                )
                self.save_data(
                    "kxky_spectra/dWdt_hyperdissipation_II",
                    -reductions["hyper"](fields, adaptive_rate, kc2, set_II).get(),
                )

        # Nonlinear exchange
        if not self.nonlinear:
            for key in self.exchange_terms:
                self.save_data(key, 0.0)
                for spectrum in self.spectra:
                    shape = tuple(
                        np.size(dimension)
                        for dimension in self.vars[
                            f"{spectrum}_spectra/{key}"
                        ].dimensions.values()
                    )
                    self.save_data(f"{spectrum}_spectra/{key}", np.zeros(shape))
            return

        for phi_part in (0, 1):
            for T_part in (0, 1):
                # Bits of {phi_a, T_b} in system.dft_bits
                self.split_fields_kernel(
                    fields,
                    self.split_fields,
                    kc2,
                    np.int32(phi_part),
                    np.int32(T_part),
                )
                system.compute_nonlinear_terms(
                    current_dt, current_time, current_step, self.split_fields, False
                )
                dft_bits = system.dft_bits

                for key, term in self.exchange_terms.items():
                    if term[1:] != (phi_part, T_part):
                        continue

                    destination = np.int32(term[0])
                    args = (
                        fields,
                        current_dt,
                        current_time,
                        current_step,
                        dft_bits,
                        kc2,
                        destination,
                    )
                    self.save_data(
                        key, self.reductions["scalar"]["nonlinear"](*args).get().item()
                    )
                    for spectrum in self.spectra:
                        self.save_data(
                            f"{spectrum}_spectra/{key}",
                            self.reductions[spectrum]["nonlinear"](*args).get(),
                        )
