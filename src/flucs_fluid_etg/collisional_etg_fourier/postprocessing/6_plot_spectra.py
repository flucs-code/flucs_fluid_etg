"""Plot ETG spectra.

The diagnostic also supports the two-dimensional ``kzkperp`` output, but it
is intentionally excluded here because this module plots only 1-D spectra.
"""

import argparse
import pathlib as pl

import matplotlib.pyplot as plt
import numpy as np
from flucs.postprocessing import FlucsPostProcessing

SPECTRA = ("kx", "ky", "kz", "kperp")
FIELD_VARIABLES = {"phi2": r"$|\varphi|^2$", "T2": r"$|\delta T|^2$"}
FIELD_COLOURS = {"phi2": "blue", "T2": "red"}
PLOT_TYPES = ("1d", "3d")
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


def _load_spectrum_history(post, nc_path, dimension, field, fraction, groups):
    """Load individual saved 1-D spectra from the final time fraction."""

    variable = f"spectra/{dimension}_spectra/{field}"
    data, _, dimensions = post.load_netcdf_variable(nc_path, variable, groups=groups)
    dimension_dict = next((dims for dims in reversed(dimensions) if dims), None)
    if dimension_dict is None or len(dimension_dict) != 1:
        raise ValueError(f"Expected a one-dimensional spectrum: {variable}")

    wavenumber = next(iter(dimension_dict.values()))
    time = post.load_netcdf_variable(nc_path, "time", groups=groups)[0]
    first = int((1.0 - fraction) * len(time))
    time = time[first:]
    data = data[first:]
    if dimension in ("kx", "kz"):
        wavenumber, _ = _fold_signed_spectrum(wavenumber, data[0])
        data = np.asarray(
            [
                _fold_signed_spectrum(next(iter(dimension_dict.values())), value)[1]
                for value in data
            ]
        )

    return wavenumber, time, data


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


def _plot_spectrum_history(post, nc_path, dimension, field, fraction, groups):
    """Plot saved spectra coloured by time, with their average in black."""

    wavenumber, time, spectra = _load_spectrum_history(
        post, nc_path, dimension, field, fraction, groups
    )
    fig, ax = _new_figure(
        f"{FIELD_VARIABLES[field]} {SPECTRAL_COORDINATES[dimension]} spectra"
    )
    valid_wavenumber = wavenumber > 0.0
    count = min(50, len(time))
    indices = np.linspace(0, len(time) - 1, count, dtype=int)
    norm = plt.Normalize(
        np.min(time), np.max(time) if len(time) > 1 else np.min(time) + 1.0
    )
    cmap = plt.get_cmap("viridis")
    for index in indices:
        valid = (
            valid_wavenumber
            & np.isfinite(spectra[index])
            & (spectra[index] > 0.0)
        )
        ax.plot(
            wavenumber[valid],
            spectra[index][valid],
            color=cmap(norm(time[index])),
            linewidth=1.0,
        )

    average = np.nanmean(spectra, axis=0)
    valid = valid_wavenumber & np.isfinite(average) & (average > 0.0)
    ax.plot(
        wavenumber[valid],
        average[valid],
        color="black",
        linewidth=2.0,
        label="time average",
    )
    ax.set_xlabel(SPECTRAL_COORDINATES[dimension])
    ax.legend(frameon=False)
    _set_line_ylim(ax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="Time")

    simulation = pl.Path(nc_path).parent.name
    figure_name = f"spectra_{dimension}_{field}_time_{simulation}"
    fig.canvas.manager.set_window_title(figure_name)
    post.save(fig, name=figure_name, suffix="png", save_kwargs={"dpi": 300})


def _load_3d_spectrum(post, nc_path, field, fraction, groups):
    variable = f"spectra/kxkykz_spectra/{field}"
    data, _, dimensions = post.load_netcdf_variable(nc_path, variable, groups=groups)
    dimension_dict = next((dims for dims in reversed(dimensions) if dims), None)
    if dimension_dict is None or set(dimension_dict) != {"kz", "kx", "ky"}:
        raise ValueError(f"Expected a kxkykz spectrum: {variable}")

    time = post.load_netcdf_variable(nc_path, "time", groups=groups)[0]
    first = int((1.0 - fraction) * len(time))
    data = np.nanmean(data[first:], axis=0)


    return (
        np.asarray(dimension_dict["kz"]),
        np.asarray(dimension_dict["kx"]),
        np.asarray(dimension_dict["ky"]),
        data,
    )


def _fold_axis(wavenumber, data, axis):
    """Fold a full FFT axis using its standard real-FFT index layout."""

    if wavenumber[-1] >= 0.0:
        data = data.copy()
        data[np.abs(data) == 0.0] = np.nan
        return wavenumber, data

    size = len(wavenumber)
    positive_count = size // 2 + 1
    positive_wavenumber = wavenumber[:positive_count]
    folded = np.take(data, np.arange(positive_count), axis=axis).copy()
    negative = np.take(data, np.arange(size - 1, positive_count - 1, -1), axis=axis)

    positive_slice = [slice(None)] * data.ndim
    positive_slice[axis] = slice(1, 1 + negative.shape[axis])
    folded[tuple(positive_slice)] += negative

    folded[np.abs(folded) == 0.0] = np.nan
    return positive_wavenumber, folded


def _integrate_spectrum(data, axis):
    """Integrate while retaining empty or zero-valued bins as gaps."""

    valid = np.any(np.isfinite(data), axis=axis)
    spectrum = np.nansum(data, axis=axis)
    spectrum[~valid | (spectrum == 0.0)] = np.nan
    return spectrum


def plot_3d_spectra(post, fraction=0.2, groups=None):
    """Plot kxkykz spectra integrated over ky or kx, coloured by kz."""

    variables = [
        f"spectra/kxkykz_spectra/{field}" for field in FIELD_VARIABLES
    ]
    nc_paths = sorted(
        {
            path
            for variable in variables
            for path in post.get_valid_netcdf_paths(variable)
        }
    )
    if not nc_paths:
        raise ValueError("No kxkykz spectra were found.")
    if len(nc_paths) != 1:
        raise ValueError("The 3-D spectra plot currently requires one input file.")

    fig, axes = plt.subplots(
        2, 2, figsize=(11, 8), sharex="col", layout="constrained"
    )
    axes = {
        ("kx", "phi2"): axes[0, 0],
        ("kx", "T2"): axes[1, 0],
        ("ky", "phi2"): axes[0, 1],
        ("ky", "T2"): axes[1, 1],
    }
    cmap = plt.get_cmap("gist_rainbow")
    norm = None
    for field in FIELD_VARIABLES:
        kz, kx, ky, data = _load_3d_spectrum(
            post, nc_paths[0], field, fraction, groups
        )
        kz, data = _fold_axis(kz, data, axis=0)
        kx, data = _fold_axis(kx, data, axis=1)
        ky, data = _fold_axis(ky, data, axis=2)
        if norm is None:
            norm = plt.Normalize(
                np.min(kz), np.max(kz) if len(kz) > 1 else np.min(kz) + 1.0
            )



        for index, value in enumerate(kz):
            if field == 'phi2' and index == 0:
                continue
            colour = cmap(norm(value))
            kx_valid = kx > 0.0
            ky_valid = ky > 0.0
            axes[("kx", field)].plot(
                kx[kx_valid],
                _integrate_spectrum(data[index][:, ky_valid], axis=1)[kx_valid],
                color=colour,
            )
            axes[("ky", field)].plot(
                ky[ky_valid],
                _integrate_spectrum(data[index][kx_valid, :], axis=0)[ky_valid],
                color=colour,
            )

    for (coordinate, field), ax in axes.items():
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel(SPECTRAL_COORDINATES[coordinate])
        ax.set_ylabel(FIELD_VARIABLES[field])
        ax.grid(True, alpha=0.25)
        ax.set_title(
            f"{FIELD_VARIABLES[field]} vs {SPECTRAL_COORDINATES[coordinate]}"
        )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=list(axes.values()),
        label=r"$k_z$",
    )
    fig.canvas.manager.set_window_title("spectra_kxkykz")
    post.save(fig, name="spectra_kxkykz", suffix="png", save_kwargs={"dpi": 300})
    plt.show()


def plot_spectra(
    post,
    fraction=0.2,
    groups=None,
    dimensions=None,
    plot_type="1d",
    time_history=False,
):
    """Plot time-averaged ``kx``, ``ky``, ``kperp`` and ``kz`` spectra.

    For one input file, ``kx`` and ``ky`` share one figure: blue/red indicate
    ``phi``/``T`` and solid/dashed lines indicate ``ky``/``kx``. For multiple
    files, ``kx`` and ``ky`` are separate figures and file colours identify
    the simulations. The final ``fraction`` of saved data is averaged.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be greater than 0 and no larger than 1")
    if plot_type == "3d":
        if time_history:
            raise ValueError("--time applies only to the 1-D spectra plot")
        if dimensions is not None:
            raise ValueError("--dimension applies only to the 1-D spectra plot")
        return plot_3d_spectra(post, fraction=fraction, groups=groups)
    if plot_type not in PLOT_TYPES:
        raise ValueError(f"Unknown plot type: {plot_type}")

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

    if time_history:
        for dimension in dimensions:
            for nc_path in nc_paths:
                for field in FIELD_VARIABLES:
                    variable = f"spectra/{dimension}_spectra/{field}"
                    if nc_path in post.get_valid_netcdf_paths(variable):
                        _plot_spectrum_history(
                            post, nc_path, dimension, field, fraction, groups
                        )

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        parents=[FlucsPostProcessing.parser()],
        description="Plot ETG spectra (1-D by default; use --type 3d for kxkykz).",
    )
    parser.add_argument(
        "--type",
        choices=PLOT_TYPES,
        default="1d",
        help="Plot type: 1d (default) or 3d kxkykz spectra.",
    )
    parser.add_argument(
        "--time",
        "-t",
        action="store_true",
        help="Additionally plot up to 50 spectra over time for each 1-D field.",
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
        plot_type=args.type,
        time_history=args.time,
    )
