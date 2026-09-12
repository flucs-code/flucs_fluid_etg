"""Plot spectral free-energy injection, dissipation and nonlinear exchange."""

import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from flucs.postprocessing import FlucsPostProcessing

SPECTRA = ("kx", "ky", "kperp", "kz")
SPECTRAL_COORDINATES = {
    "kx": r"$k_x$",
    "ky": r"$k_y$",
    "kz": r"$k_z$",
    "kperp": r"$k_\perp$",
}
TERMS = {
    "dWdt_inj": (r"$\epsilon$", "tab:red"),
    "dWdt_coll": (r"$\mathcal{D}_\parallel$", "tab:blue"),
    "dWdt_hyperdissipation": (r"$\mathcal {D}_\perp$", "tab:green"),
    "nl": (r"$\Pi_{\rm NL}$", "black"),
}


def _fold_signed(wavenumber, spectrum):
    positive = wavenumber >= 0.0
    positive_wavenumber = wavenumber[positive]
    positive_spectrum = spectrum[positive]
    folded = np.zeros_like(positive_spectrum)

    for index, value in enumerate(positive_wavenumber):
        if value == 0.0:
            folded[index] = positive_spectrum[index]
        else:
            folded[index] = (
                positive_spectrum[index]
                + spectrum[np.isclose(wavenumber, -value)].sum()
            )

    return positive_wavenumber, folded


def _load(post, nc_path, dimension, fraction, groups):
    prefix = f"spectral_free_energy/{dimension}_spectra/"
    names = ("dWdt", "dWdt_inj", "dWdt_coll", "dWdt_hyperdissipation")
    loaded = {}
    wavenumber = None
    for name in names:
        data, _, dimensions = post.load_netcdf_variable(
            nc_path, prefix + name, groups=groups
        )
        dimension_dict = next((dims for dims in reversed(dimensions) if dims), None)
        if dimension_dict is None or len(dimension_dict) != 1:
            raise ValueError(f"Expected a one-dimensional spectrum: {prefix + name}")
        if wavenumber is None:
            wavenumber = next(iter(dimension_dict.values()))

        time = post.load_netcdf_variable(nc_path, "time", groups=groups)[0]
        first = int((1.0 - fraction) * len(time))
        loaded[name] = np.nanmean(data[first:], axis=0)

    loaded["nl"] = (
        loaded["dWdt"]
        - loaded["dWdt_inj"]
        - loaded["dWdt_coll"]
        - loaded["dWdt_hyperdissipation"]
    )
    if dimension in ("kx", "kz"):
        wavenumber, _ = _fold_signed(wavenumber, loaded["dWdt"])
        for name, value in loaded.items():
            _, loaded[name] = _fold_signed(next(iter(dimension_dict.values())), value)

    return wavenumber, loaded


def _plot_term(ax, wavenumber, spectrum, label, colour, linestyle, marker):
    valid = (
        (wavenumber > 0.0)
        & np.isfinite(wavenumber)
        & np.isfinite(spectrum)
        & (np.abs(spectrum) > 0.0)
    )
    ax.plot(
        wavenumber[valid],
        np.abs(spectrum[valid]),
        color=colour,
        linestyle=linestyle,
        linewidth=1.5,
        label=label,
    )
    if marker is not None and len(wavenumber) > 1 and np.isfinite(spectrum[0]):
        ax.scatter(
            wavenumber[1],
            max(abs(spectrum[0]), np.finfo(float).tiny),
            color=colour,
            marker=marker,
            s=36,
            zorder=4,
            label="_nolegend_",
        )


def _plot_nonlinear_parts(ax, wavenumber, spectrum, label, colour, linestyle, marker):
    """Plot positive NL and the magnitude of negative NL separately."""
    neg_linestyle = {
        "-": ":", 
        "--": "-."
        }
    for positive, suffix, part_linestyle in (
        (True, "$> 0$", linestyle),
        (False, "$< 0$", neg_linestyle[linestyle]), #  Note labels have the opposite signs from the calculation
    ):
        part = (
            np.where(spectrum > 0.0, spectrum, 0.0)
            if positive
            else np.where(spectrum < 0.0, -spectrum, 0.0)
        )
        valid = np.isfinite(wavenumber) & np.isfinite(part) & (wavenumber > 0.0)
        ax.plot(
            wavenumber[valid],
            part[valid],
            color=colour,
            linestyle=part_linestyle,
            linewidth=1.5,
            label=f"{label} {suffix}",
        )

        if (
            marker is not None
            and len(wavenumber) > 1
            and np.isfinite(spectrum[0])
            and (
                (positive and spectrum[0] > 0.0) or (not positive and spectrum[0] < 0.0)
            )
        ):
            ax.scatter(
                wavenumber[1],
                abs(spectrum[0]),
                color=colour,
                marker=marker,
                s=36,
                zorder=4,
                label="_nolegend_",
            )


def _new_figure(title):
    fig, ax = plt.subplots(1, 1, layout="constrained")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel(r"$|dW/dt|$")
    ax.grid(True, alpha=0.25)
    return fig, ax


def _set_line_ylim(ax):
    values = np.concatenate([line.get_ydata() for line in ax.lines])
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values):
        ax.set_ylim(np.min(values) / 1.5, np.max(values) * 1.5)


def plot_free_energy_exchange(post, fraction=0.2, groups=None, dimensions=None):
    """Plot time-averaged exchange terms for all available simulations."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be greater than 0 and no larger than 1")

    dimensions = SPECTRA if dimensions is None else tuple(dimensions)
    invalid = set(dimensions) - set(SPECTRA)
    if invalid:
        raise ValueError(f"Unknown spectral dimensions: {sorted(invalid)}")
    dimensions = tuple(dimension for dimension in SPECTRA if dimension in dimensions)
    if not dimensions:
        raise ValueError("At least one spectral dimension must be selected.")

    nc_paths = sorted(
        {
            path
            for dimension in dimensions
            for path in post.get_valid_netcdf_paths(
                f"spectral_free_energy/{dimension}_spectra/dWdt"
            )
        }
    )
    if not nc_paths:
        raise ValueError("No spectral free-energy diagnostics were found.")

    single_file = len(nc_paths) == 1
    file_colours = plt.cm.rainbow(np.linspace(0.0, 1.0, len(nc_paths)))
    figures = []
    plot_groups = []
    if single_file:
        fig, ax = _new_figure("Free-energy exchange: kx and ky")
        figures.append((fig, "free_energy_exchange_kx_ky"))
        plot_groups.extend(
            (ax, dimension) for dimension in ("kx", "ky") if dimension in dimensions
        )
    else:
        for dimension in ("kx", "ky"):
            if dimension not in dimensions:
                continue
            fig, ax = _new_figure(
                f"Free-energy exchange: {SPECTRAL_COORDINATES[dimension]}"
            )
            figures.append((fig, f"free_energy_exchange_{dimension}"))
            plot_groups.append((ax, dimension))

    for dimension in ("kperp", "kz"):
        if dimension not in dimensions:
            continue
        fig, ax = _new_figure(
            f"Free-energy exchange: {SPECTRAL_COORDINATES[dimension]}"
        )
        figures.append((fig, f"free_energy_exchange_{dimension}"))
        plot_groups.append((ax, dimension))

    for ax, dimension in plot_groups:
        ax.set_xlabel(SPECTRAL_COORDINATES[dimension])
        for file_index, nc_path in enumerate(nc_paths):
            try:
                wavenumber, terms = _load(post, nc_path, dimension, fraction, groups)
            except ValueError:
                continue

            simulation = pl.Path(nc_path).parent.name
            for term, (term_label, term_colour) in TERMS.items():
                if single_file:
                    colour = term_colour
                    linestyle = "--" if dimension == "kx" else "-"
                    label = f'{term_label}$(${SPECTRAL_COORDINATES[dimension]}$)$'
                else:
                    colour = file_colours[file_index]
                    linestyle = "-" if term == "nl" else "--"
                    label = f"{simulation}: {term_label}"

                marker = (
                    None
                    if dimension == "kperp"
                    else ("v" if dimension == "ky" else "x")
                )
                if term == "nl":
                    _plot_nonlinear_parts(
                        ax,
                        wavenumber,
                        terms[term],
                        label,
                        colour,
                        linestyle,
                        marker,
                    )
                else:
                    _plot_term(
                        ax,
                        wavenumber,
                        terms[term],
                        label,
                        colour,
                        linestyle,
                        marker,
                    )

        _set_line_ylim(ax)
        ax.legend(frameon=False)

    for fig, name in figures:
        fig.canvas.manager.set_window_title(name)
        post.save(fig, name=name, suffix="png", save_kwargs={"dpi": 300})
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description="Plot spectral free-energy exchange terms.",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List available netCDF variables and exit.",
    )
    parser.add_argument(
        "--dimension",
        "-d",
        choices=SPECTRA,
        default=None,
        help="Spectral dimension to plot; omit to plot all dimensions.",
    )
    parser.add_argument(
        "--fraction",
        "-f",
        type=float,
        default=0.2,
        help="Final fraction of saved data to average (default: 0.2).",
    )
    args = parser.parse_args()
    post = FlucsPostProcessing(
        io_paths=args.io_path,
        save_directory=args.save_directory,
        output_files="output.*.nc",
        constraint="both",
    )
    if args.list:
        post.list_netcdf_variables()
        raise SystemExit
    dimensions = None if args.dimension is None else (args.dimension,)
    plot_free_energy_exchange(
        post,
        fraction=args.fraction,
        groups=args.groups,
        dimensions=dimensions,
    )
