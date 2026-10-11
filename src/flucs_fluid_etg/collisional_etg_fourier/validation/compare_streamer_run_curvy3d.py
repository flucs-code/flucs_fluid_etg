"""Statistical comparison of a flucs streamer run with a curvy3d run.

Usage (with a Python that can import curvy3d)::

    python compare_streamer_run_curvy3d.py RUN_DIR METADATA MATCH \
        [--curvy3d ~/source/curvy3d_etg] [--fraction 0.5] [--group N]

RUN_DIR holds a flucs run with streamer_exchange saving the kxky contraction.
METADATA is curvy3d's simulation_metadata.json (or a list of its records);
MATCH is a substring that selects one record by filename.

Both runs are averaged over the last ``fraction`` of their saves (curvy3d's
analysis uses the second half). flucs values are converted to curvy3d units
(W and every rate times 2) and reduced with curvy3d's kperp shells. Reported:

- the time-averaged Pi_sf, W_I and W_II;
- the drain ratio Pi_sf / (2 W_I gamma_bar), where gamma_bar is the inviscid
  gamma_lin(ky = kperp, kx = 0, kz = 2 pi/Lz), clipped at 0, averaged over the
  W_I kperp spectrum, as in paper_figures.ipynb;
- the shape of the Pi_sf kperp spectrum: the cumulative deposition fraction,
  its 20/50/80 % wavenumbers and the largest difference between the two
  cumulative fractions over the common kperp range.

Agreement is statistical: the runs are independent realisations.
"""

import argparse
import contextlib
import json
import pathlib as pl
import sys

import netCDF4 as nc
import numpy as np
import toml

sys.path.insert(0, str(pl.Path(__file__).parent))
from compare_streamer_exchange_curvy3d import (
    curvy3d_input_string,
    fft_modes,
)


def find_group(run_dir, group=None):
    for path in sorted(pl.Path(run_dir).glob("output*.nc")):
        dataset = nc.Dataset(path)
        for name, sub in dataset.groups.items():
            if group is not None and name != str(group):
                continue
            if "streamer_exchange" in sub.groups and (
                "kxky_spectra" in sub["streamer_exchange"].groups
            ):
                return dataset, sub
        dataset.close()
    raise FileNotFoundError(f"No streamer_exchange kxky output in {run_dir}")


def gamma_bar(pars, kperp, W_I_kperp, disp_solutions):
    """W_I-weighted inviscid growth rate, as in paper_figures.ipynb."""
    pars.perpvisc = 0.0
    with contextlib.redirect_stdout(None):
        roots = disp_solutions(kz=2 * np.pi / pars.Lz, kx=0.0, ky=kperp, Pars=pars)
    root = np.take_along_axis(roots, np.argmax(roots.real, axis=0)[None], axis=0)[0]
    gamma = np.clip(root.real, 0, None)
    dk = np.gradient(kperp)
    return np.sum(gamma * W_I_kperp * dk) / np.sum(W_I_kperp * dk)


def deposition(kperp, spectrum):
    """Cumulative fraction of Pi_sf below kperp and its 20/50/80 % points."""
    cumulative = np.cumsum(spectrum * np.gradient(kperp))
    fraction = cumulative / cumulative[-1]
    points = [float(np.interp(p, fraction, kperp)) for p in (0.2, 0.5, 0.8)]
    return fraction, points


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("metadata")
    parser.add_argument("match")
    parser.add_argument("--curvy3d", default=str(pl.Path.home() / "source/curvy3d_etg"))
    parser.add_argument("--fraction", type=float, default=0.5)
    parser.add_argument("--group", default=None)
    args = parser.parse_args()

    sys.path.insert(0, args.curvy3d)
    with contextlib.redirect_stdout(None):
        import multiplot3d_etg as m3d
        from curvy3dinput import Curvy3DInput
        from linear.curvy3d_linear_etg import disp_solutions

    # curvy3d record
    with open(args.metadata) as f:
        records = json.load(f)
    records = [r for r in records if args.match in r.get("filename", "")]
    if len(records) != 1:
        raise ValueError(
            f"{len(records)} metadata records match {args.match!r}: "
            f"{[r.get('filename') for r in records]}"
        )
    record = records[0]
    times = np.asarray(record["nl_exchange_time"], dtype=float)
    selected = times >= times[-1] - args.fraction * (times[-1] - times[0])

    def curvy_average(key):
        return np.asarray(record[key], dtype=float)[selected].mean(axis=0)

    curvy_kperp = np.asarray(record["kperp"], dtype=float)
    curvy = {
        "Pi_sf": curvy_average("nl_streamer_to_small_kx").sum()
        * 2
        * np.pi
        / record["Lx"],
        "W_I": curvy_average("free_energy_large_kx").sum() * 2 * np.pi / record["Lx"],
        "W_II": curvy_average("free_energy_small_kx").sum() * 2 * np.pi / record["Lx"],
        "Pi_sf_kperp": curvy_average("nl_streamer_to_small_kperp"),
        "W_I_kperp": curvy_average("free_energy_large_kperp"),
    }
    with contextlib.redirect_stdout(None):
        curvy_pars = Curvy3DInput(input_str="")
    curvy_pars.electrostatic = True
    for name in ("Lx", "Ly", "Lz", "kappaN", "kappaT"):
        setattr(curvy_pars, name, float(record[name]))
    curvy["gamma_bar"] = gamma_bar(
        curvy_pars, curvy_kperp, curvy["W_I_kperp"], disp_solutions
    )

    # flucs run
    dataset, group = find_group(args.run_dir, args.group)
    diag = group["streamer_exchange"]
    inputs = toml.loads(str(group["input_file"][...]))
    flucs_times = np.asarray(group["time"][:])
    flucs_selected = flucs_times >= flucs_times[-1] - args.fraction * (
        flucs_times[-1] - flucs_times[0]
    )

    if inputs.get("dealiasing", {}).get("method", "two-thirds") == "two-thirds":
        n_unpadded = tuple(
            inputs["dealiasing"][f"n{dim}_unpadded"] for dim in ("z", "x", "y")
        )
    else:
        raise ValueError("Only two-thirds dealiasing is supported.")

    with contextlib.redirect_stdout(None):
        pars = Curvy3DInput(input_str=curvy3d_input_string(inputs, n_unpadded))
    geometry = m3d._kxy_spectral_geometry(pars)
    nx = len(diag["kx"])
    ix = fft_modes(n_unpadded[1]) % nx
    half_ny_u = n_unpadded[2] // 2 + 1

    def flucs_average(name):
        """Time-averaged kxky data in curvy3d units, reduced on its shells."""
        kxky = np.asarray(diag["kxky_spectra"][name][:])[flucs_selected].mean(axis=0)
        return m3d._reduce_kxy_spectrum(2 * kxky[ix][:, :half_ny_u], pars, geometry)

    flucs_kperp = geometry["kperp_centers"]
    reduced = {
        name: flucs_average(name) for name in ("nl_streamer_to_small", "W_I", "W_II")
    }
    dkx = 2 * np.pi / pars.Lx
    flucs = {
        "Pi_sf": reduced["nl_streamer_to_small"]["kx_spectrum"].sum() * dkx,
        "W_I": reduced["W_I"]["kx_spectrum"].sum() * dkx,
        "W_II": reduced["W_II"]["kx_spectrum"].sum() * dkx,
        "Pi_sf_kperp": reduced["nl_streamer_to_small"]["kperp_spectrum"],
        "W_I_kperp": reduced["W_I"]["kperp_spectrum"],
    }
    flucs["gamma_bar"] = gamma_bar(
        pars, flucs_kperp, flucs["W_I_kperp"], disp_solutions
    )

    # The saved scalars give the same totals without the shells' corners
    scalar_Pi_sf = 2 * np.asarray(diag["nl_streamer_to_small"][:])
    scalar_times = np.asarray(group["time"][:])

    print(f"curvy3d: {record['filename']}")
    print(f"   {int(selected.sum())} saves, t = {times[selected][0]:g}..{times[-1]:g}")
    print(f"flucs:   {args.run_dir}")
    print(
        f"   {int(flucs_selected.sum())} saves, "
        f"t = {flucs_times[flucs_selected][0]:.4g}..{flucs_times[-1]:.4g}, "
        f"ky_I = {float(diag['ky_streamer'][-1]):.4f}, "
        f"mean scalar Pi_sf = {scalar_Pi_sf[flucs_selected].mean():.4e} "
        f"(over {scalar_times[flucs_selected].size} saves)"
    )
    print(
        f"   curvy3d ky_I = {record['nl_exchange_ky_streamer']:.4f}, "
        f"kc = {record['nl_exchange_kperp_cutoff']:.4f}"
    )

    print("\nIn curvy3d units (W twice flucs')")
    print(f"{'':28s}{'flucs':>14s}{'curvy3d':>14s}{'ratio':>10s}")
    for name in ("Pi_sf", "W_I", "W_II", "gamma_bar"):
        print(
            f"{name:28s}{flucs[name]:14.5e}{curvy[name]:14.5e}"
            f"{flucs[name] / curvy[name]:10.3f}"
        )
    drain = {
        code: values["Pi_sf"] / (2 * values["W_I"] * values["gamma_bar"])
        for code, values in (("flucs", flucs), ("curvy3d", curvy))
    }
    print(
        f"{'Pi_sf/(2 W_I gamma_bar)':28s}{drain['flucs']:14.4f}"
        f"{drain['curvy3d']:14.4f}{drain['flucs'] / drain['curvy3d']:10.3f}"
    )

    print("\nPi_sf kperp spectrum shape (cumulative deposition)")
    fraction_flucs, points_flucs = deposition(flucs_kperp, flucs["Pi_sf_kperp"])
    fraction_curvy, points_curvy = deposition(curvy_kperp, curvy["Pi_sf_kperp"])
    for label, p_f, p_c in zip(
        ("k_dep,20", "k_dep,50", "k_dep,80"), points_flucs, points_curvy
    ):
        print(f"{label:28s}{p_f:14.4f}{p_c:14.4f}{p_f / p_c:10.3f}")
    k_max = min(flucs_kperp[-1], curvy_kperp[-1])
    common = np.linspace(max(flucs_kperp[0], curvy_kperp[0]), k_max, 400)
    difference = np.max(
        np.abs(
            np.interp(common, flucs_kperp, fraction_flucs)
            - np.interp(common, curvy_kperp, fraction_curvy)
        )
    )
    print(f"{'max |F_flucs - F_curvy3d|':28s}{difference:14.4f}")

    dataset.close()


if __name__ == "__main__":
    main()
