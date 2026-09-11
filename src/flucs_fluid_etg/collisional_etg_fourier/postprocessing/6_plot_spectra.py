"""Plot one-dimensional ETG spectra."""

import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from flucs.postprocessing import FlucsPostProcessing

SPECTRA = ("kx", "ky", "kz", "kperp")
FIELD_VARIABLES = {"phi2": r"$|\varphi|^2$", "T2": r"$|\delta T|^2$"}
FIELD_COLOURS = {"phi2": "blue", "T2": "red"}
SPECTRAL_COORDINATES = {
    "kx": r"$k_x$",
    "ky": r"$k_y$",
    "kz": r"$k_z$",
    "kperp": r"$k_\perp$",
}


def _fold_signed_spectrum(wavenumber, spectrum):
    """Fold a signed FFT spectrum onto nonnegative wavenumbers."""

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


def _load_spectrum(post, nc_path, dimension, field, fraction, groups):
    variable = f"spectra/{dimension}_spectra/{field}"
    data, _, dimensions = post.load_netcdf_variable(nc_path, variable, groups=groups)
    dimension_dict = next((dims for dims in reversed(dimensions) if dims), None)
    if dimension_dict is None or len(dimension_dict) != 1:
        raise ValueError(f"Expected a one-dimensional spectrum: {variable}")

    wavenumber = next(iter(dimension_dict.values()))
    time = post.load_netcdf_variable(nc_path, "time", groups=groups)[0]
    first = int((1.0 - fraction) * len(time))
    spectrum = np.nanmean(data[first:], axis=0)

    if dimension in ("kx", "kz"):
        wavenumber, spectrum = _fold_signed_spectrum(wavenumber, spectrum)

    return wavenumber, spectrum


def _plot_curve(ax, wavenumber, spectrum, label, colour, linestyle, scatter_k0, marker):
    positive = wavenumber > 0.0
    valid = (
        positive & np.isfinite(wavenumber) & np.isfinite(spectrum) & (spectrum > 0.0)
    )
    ax.plot(
        wavenumber[valid],
        spectrum[valid],
        color=colour,
        linestyle=linestyle,
        linewidth=1.5,
        label=label,
    )

    if (
        scatter_k0
        and len(wavenumber) > 1
        and np.isfinite(spectrum[0])
        and spectrum[0] > 0.0
    ):
        # Zero cannot be displayed on a logarithmic x-axis. Place its value
        # at the first positive coordinate and identify it in the legend.
        ax.scatter(
            wavenumber[1],
            spectrum[0],
            color=colour,
            marker=marker,
            s=36,
            zorder=4,
            label="_nolegend_",
        )


def _set_line_ylim(ax):
    """Set logarithmic y-limits from lines, excluding origin markers."""

    values = np.concatenate(
        [line.get_ydata() for line in ax.lines if len(line.get_ydata())]
    )
    values = values[np.isfinite(values) & (values > 0.0)]
    if len(values):
        lower = np.min(values)
        upper = np.max(values)
        if lower == upper:
            lower /= 2.0
            upper *= 2.0
        else:
            lower /= 1.5
            upper *= 1.5
        ax.set_ylim(lower, upper)


def _new_figure(title):
    fig, ax = plt.subplots(1, 1, layout="constrained")
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("spectrum")
    ax.grid(True, alpha=0.25)
    return fig, ax


def plot_spectra(post, fraction=0.2, groups=None, dimensions=None):
    """Plot time-averaged ``kx``, ``ky``, ``kperp`` and ``kz`` spectra.

    For one input file, ``kx`` and ``ky`` share one figure: blue/red indicate
    ``phi``/``T`` and solid/dashed lines indicate ``ky``/``kx``. For multiple
    files, ``kx`` and ``ky`` are separate figures and file colours identify
    the simulations. The final ``fraction`` of saved data is averaged.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be greater than 0 and no larger than 1")

    variables = [
        f"spectra/{dimension}_spectra/{field}"
        for dimension in SPECTRA
        for field in FIELD_VARIABLES
    ]
    nc_paths = sorted(
        {
            path
            for variable in variables
            for path in post.get_valid_netcdf_paths(variable)
        }
    )
    if not nc_paths:
        raise ValueError("No one-dimensional spectra were found.")

    dimensions = SPECTRA if dimensions is None else tuple(dimensions)
    invalid = set(dimensions) - set(SPECTRA)
    if invalid:
        raise ValueError(f"Unknown spectral dimensions: {sorted(invalid)}")
    dimensions = tuple(dimension for dimension in SPECTRA if dimension in dimensions)
    if not dimensions:
        raise ValueError("At least one spectral dimension must be selected.")

    single_file = len(nc_paths) == 1
    file_colours = plt.cm.rainbow(np.linspace(0.0, 1.0, len(nc_paths)))
    figures = []

    if single_file:
        fig, ax = _new_figure("Perpendicular spectra")
        figures.append((fig, ax, "spectra_kx_ky"))
        plot_groups = [
            (ax, dimension) for dimension in ("kx", "ky") if dimension in dimensions
        ]
    else:
        plot_groups = []
        for dimension in ("kx", "ky"):
            if dimension not in dimensions:
                continue
            fig, ax = _new_figure(f"{SPECTRAL_COORDINATES[dimension]} spectra")
            figures.append((fig, ax, f"spectra_{dimension}"))
            plot_groups.append((ax, dimension))

    for dimension in ("kperp", "kz"):
        if dimension not in dimensions:
            continue
        fig, ax = _new_figure(f"{SPECTRAL_COORDINATES[dimension]} spectra")
        figures.append((fig, ax, f"spectra_{dimension}"))
        plot_groups.append((ax, dimension))

    for ax, dimension in plot_groups:
        ax.set_xlabel(SPECTRAL_COORDINATES[dimension])
        for file_index, nc_path in enumerate(nc_paths):
            simulation = pl.Path(nc_path).parent.name
            for field, field_label in FIELD_VARIABLES.items():
                variable = f"spectra/{dimension}_spectra/{field}"
                if nc_path not in post.get_valid_netcdf_paths(variable):
                    continue

                wavenumber, spectrum = _load_spectrum(
                    post, nc_path, dimension, field, fraction, groups
                )
                if single_file:
                    colour = FIELD_COLOURS[field]
                    linestyle = "--" if dimension == "kx" else "-"
                    label = f"{field_label} $(${SPECTRAL_COORDINATES[dimension]}$)$"
                else:
                    colour = file_colours[file_index]
                    linestyle = "-" if field == "phi2" else "--"
                    label = f"{simulation}: {field_label}"

                _plot_curve(
                    ax,
                    wavenumber,
                    spectrum,
                    label,
                    colour,
                    linestyle,
                    scatter_k0=dimension != "kperp",
                    marker="v" if dimension == "ky" else "x",
                )

        _set_line_ylim(ax)
        ax.legend(frameon=False)

    for fig, _, figure_name in figures:
        fig.canvas.manager.set_window_title(figure_name)
        post.save(fig, name=figure_name, suffix="png", save_kwargs={"dpi": 300})

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description="Plot the one-dimensional ETG spectra.",
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
    plot_spectra(
        post,
        fraction=args.fraction,
        groups=args.groups,
        dimensions=dimensions,
    )
