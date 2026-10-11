"""Compare the streamer_exchange diagnostic with curvy3d on the same snapshot.

Usage (with a Python that can import curvy3d, i.e. scipy and netCDF4)::

    python compare_streamer_exchange_curvy3d.py RESTART RUN_DIR \
        --curvy3d ~/source/curvy3d_etg [--group N] [--save S]

RESTART is the flucs restart file holding the fields; RUN_DIR holds the output
of a flucs run that executed streamer_exchange on exactly these fields, with
the kxky and kz contractions (e.g. a run restarted from RESTART with
tfinal equal to its time, so that only the t = 0 save is made).

The fields are written to a curvy3d-format file on curvy3d's unpadded odd
grid and passed to multiplot3d_etg.compute_file_nl_exchange with the same
ky_I and cut-off. The flucs kxky (summed over kz on the half grid) and kz
contractions are reduced with curvy3d's own _reduce_kxy_spectrum and
_reduce_kz_spectrum, so the kx, ky, kz and kperp spectra use curvy3d's
shells, folds and ky > 0 conjugate weighting (and, where curvy3d stores
them, its log-binned (|kx|, ky) spectra, <key>_kxky). flucs and curvy3d evolve the
same equations and curvy3d's W is twice flucs', so flucs * 2 must equal
curvy3d for every term, W_I and W_II. The run must use two-thirds dealiasing
(the solved modes then form the box that curvy3d stores).
"""

import argparse
import contextlib
import pathlib as pl
import sys
import tempfile

import netCDF4 as nc
import numpy as np
import toml

TERMS = (
    "nl_streamer_to_small",
    "nl_small_to_streamer",
    "nl_batchelor",
    "nl_small_self",
    "nl_streamer_self",
    "nl_streamer_self_to_small",
    "nl_small_to_streamer_via_streamer_flow",
    "nl_small_flow_on_streamer",
)
# flucs W parts -> curvy3d free-energy keys
ENERGIES = {"W_I": "free_energy_large", "W_II": "free_energy_small"}
AXES = ("kx", "ky", "kz", "kperp")


def fft_modes(n):
    return np.rint(np.fft.fftfreq(n) * n).astype(int)


def curvy3d_input_string(inputs, n_unpadded):
    """curvy3d input file for a flucs input on the unpadded grid
    n_unpadded = (nz, nx, ny), with the same parameters."""
    nz_u, nx_u, ny_u = n_unpadded
    parameters = inputs["parameters"]
    dimensions = inputs["dimensions"]
    hyper = inputs.get("hyperdissipation", {})
    return "\n".join(
        [
            "precision double",
            "electrostatic 1",
            f"kappaT {parameters.get('kappaT', 1.0)}",
            f"kappaN {parameters.get('kappaN', 0.0)}",
            f"kappaB {parameters.get('kappaB', 0.0)}",
            f"charge {parameters.get('charge', 1.0)}",
            f"tratio {parameters.get('tratio', 1.0)}",
            f"perpvisc {max(hyper.get('kperp', 0.0), 0.0)}",
            f"perpvisc_power {hyper.get('kperp_power', 2)}",
            f"Lx {dimensions['Lx']}",
            f"Ly {dimensions['Ly']}",
            f"Lz {dimensions['Lz']}",
            f"padded_nx {3 * (nx_u // 2) + 3}",
            f"padded_ny {3 * (ny_u // 2) + 3}",
            f"padded_nz {3 * (nz_u // 2) + 3}",
            f"nx {nx_u}",
            f"ny {ny_u}",
            f"nz {nz_u}",
            "save_time 1",
            "tfinal 1",
            "",
        ]
    )


def write_curvy3d_file(path, fields, inputs, n_unpadded):
    """Writes fields (2, nz, nx, half_ny) on flucs' padded grid as a curvy3d
    output file on the unpadded odd grid n_unpadded = (nz, nx, ny)."""
    nz_u, nx_u, ny_u = n_unpadded
    half_ny_u = ny_u // 2 + 1
    nz, nx = fields.shape[1:3]

    iz = fft_modes(nz_u) % nz
    ix = fft_modes(nx_u) % nx
    data = fields[:, iz][:, :, ix][:, :, :, :half_ny_u]

    input_str = curvy3d_input_string(inputs, n_unpadded)

    with nc.Dataset(path, "w", format="NETCDF4") as dataset:
        dataset.createDimension("nt", 1)
        dataset.createDimension("nz", nz_u)
        dataset.createDimension("nx", nx_u)
        dataset.createDimension("halfny", half_ny_u)
        dataset.createDimension("input_dim", len(input_str))
        variable = dataset.createVariable("input", "S1", ("input_dim",))
        variable[:] = np.array(list(input_str), dtype="S1")
        for name, field in (("phi", data[0]), ("T", data[1])):
            for suffix, part in (("r", field.real), ("i", field.imag)):
                variable = dataset.createVariable(
                    f"{name}_rfft_{suffix}", "f8", ("nt", "nz", "nx", "halfny")
                )
                variable[0] = part

    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("restart")
    parser.add_argument("run_dir")
    parser.add_argument("--curvy3d", default=str(pl.Path.home() / "source/curvy3d_etg"))
    parser.add_argument("--group", default=None)
    parser.add_argument("--save", type=int, default=0)
    args = parser.parse_args()

    sys.path.insert(0, args.curvy3d)
    with contextlib.redirect_stdout(None):
        import multiplot3d_etg as m3d
        from curvy3dinput import Curvy3DInput

    # flucs fields and grid
    with nc.Dataset(args.restart) as restart:
        fields = np.asarray(restart["fields_real"][:]) + 1j * np.asarray(
            restart["fields_imag"][:]
        )
        restart_inputs = toml.loads(str(restart["input_file"][...]))

    # flucs diagnostic output
    output_path = sorted(pl.Path(args.run_dir).glob("output*.nc"))
    dataset = group = None
    for path in output_path:
        dataset = nc.Dataset(path)
        for name, sub in dataset.groups.items():
            if (args.group is None or name == str(args.group)) and (
                "streamer_exchange" in sub.groups
            ):
                group = sub
                break
        if group is not None:
            break
        dataset.close()
    if group is None:
        raise FileNotFoundError(f"No streamer_exchange output in {args.run_dir}")

    diag = group["streamer_exchange"]
    inputs = toml.loads(str(group["input_file"][...]))
    if inputs.get("dealiasing", {}).get("method", "two-thirds") != "two-thirds":
        raise ValueError("The comparison needs two-thirds dealiasing.")

    n_unpadded = tuple(
        inputs["dealiasing"][f"n{dim}_unpadded"] for dim in ("z", "x", "y")
    )
    ky_I = float(diag["ky_streamer"][args.save])
    k_c = float(diag["kperp_cutoff"][args.save])
    cutoff_factor = k_c / ky_I
    print(
        f"flucs: {args.run_dir}, t = {float(group['time'][args.save]):.4f}, "
        f"ky_I = {ky_I:.6f}, kc = {k_c:.6f}, unpadded (nz, nx, ny) = {n_unpadded}"
    )

    for key in ("dimensions", "parameters"):
        for name, value in restart_inputs[key].items():
            if not isinstance(value, dict) and inputs[key].get(name) != value:
                print(f"warning: {key}.{name} differs from the restart input")

    with tempfile.TemporaryDirectory() as tmp:
        curvy_path = pl.Path(tmp) / "snapshot.nc"
        write_curvy3d_file(curvy_path, fields, inputs, n_unpadded)

        with contextlib.redirect_stdout(None):
            pars = Curvy3DInput(str(curvy_path))

        # Set-I masks: flucs (strict kperp < kc, with a relative tolerance on
        # kc^2) against curvy3d's float comparison
        geometry = m3d._kxy_spectral_geometry(pars)
        kx = 2 * np.pi * np.fft.fftfreq(pars.nx, d=pars.Lx / pars.nx)
        ky = geometry["ky"]
        kperp = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
        curvy_mask = kperp < k_c
        Lx, Ly = inputs["dimensions"]["Lx"], inputs["dimensions"]["Ly"]
        exact_kperp2 = (fft_modes(pars.nx)[:, None] * 2 * np.pi / Lx) ** 2 + (
            np.arange(len(ky))[None, :] * 2 * np.pi / Ly
        ) ** 2
        flucs_mask = exact_kperp2 < k_c**2 * (1 - 1e-6)
        mismatch = np.argwhere(curvy_mask != flucs_mask)
        print(
            f"set-I (kx, ky) masks: {flucs_mask.sum()} modes in flucs, "
            f"{curvy_mask.sum()} in curvy3d, {len(mismatch)} differ"
        )
        if len(mismatch):
            # Shrink curvy3d's cut-off just enough to exclude kperp = kc ties
            cutoff_factor *= 1 - 1e-9
            curvy_mask = kperp < cutoff_factor * ky_I
            print(
                f"   modes on kperp = kc: {mismatch.tolist()}; running curvy3d "
                f"with cutoff_factor * (1 - 1e-9), now "
                f"{np.count_nonzero(curvy_mask != flucs_mask)} differ"
            )

        with contextlib.redirect_stdout(None):
            curvy = m3d.compute_file_nl_exchange(
                str(curvy_path), ky_I, cutoff_factor=cutoff_factor
            )

    # flucs data reduced with curvy3d's reducers, times 2 for W
    nz, nx = fields.shape[1:3]
    iz = fft_modes(n_unpadded[0]) % nz
    ix = fft_modes(n_unpadded[1]) % nx
    half_ny_u = n_unpadded[2] // 2 + 1

    log_bins = hasattr(m3d, "_bin_kxy_log")
    if log_bins:
        kx_edges, ky_edges = m3d._kxy_log_edges(pars)

    def flucs_spectra(name):
        kxky = np.asarray(diag["kxky_spectra"][name][args.save])[ix][:, :half_ny_u]
        kz = np.asarray(diag["kz_spectra"][name][args.save])[iz]
        reduced = m3d._reduce_kxy_spectrum(2 * kxky, pars, geometry)
        spectra = {axis: reduced[f"{axis}_spectrum"] for axis in ("kx", "ky", "kperp")}
        spectra["kz"] = m3d._reduce_kz_spectrum(2 * kz, pars)[1]
        if log_bins:
            spectra["kxky"] = m3d._bin_kxy_log(
                2 * kxky, pars, geometry, kx_edges, ky_edges
            )
        return spectra

    # curvy3d versions differ in which spectra they store
    axes = [
        axis
        for axis in (*AXES, "kxky")
        if f"{TERMS[0]}_{axis}" in curvy or f"{ENERGIES['W_I']}_{axis}" in curvy
    ]
    print(
        "\nmax |2 flucs - curvy3d| / max |curvy3d| per spectrum, with "
        "max |curvy3d Pi_sf| for the exchange terms ('-': not stored)"
    )
    print(f"{'term':40s}" + "".join(f"{axis:>11s}" for axis in axes) + "   total(ky)")
    worst = 0.0
    for name, curvy_name in [*((t, t) for t in TERMS), *ENERGIES.items()]:
        mine = flucs_spectra(name)
        row = []
        for axis in axes:
            if f"{curvy_name}_{axis}" not in curvy:
                row.append("-")
                continue
            reference = np.asarray(curvy[f"{curvy_name}_{axis}"])
            # 1-D spectra are stored per save, kxky as a time average
            reference = reference if axis == "kxky" else reference[0]
            # Exchange terms are compared on the scale of Pi_sf, since some of
            # them vanish identically (e.g. nl_streamer_self when set I is
            # only kz = +-1)
            scale_name = TERMS[0] if name in TERMS else curvy_name
            scale_reference = np.asarray(curvy[f"{scale_name}_{axis}"])
            scale = np.max(np.abs(scale_reference))
            error = np.max(np.abs(mine[axis] - reference)) / scale if scale else 0.0
            worst = max(worst, error)
            row.append(f"{error:.2e}")
        dky = 2 * np.pi / pars.Ly
        total_curvy = np.sum(curvy[f"{curvy_name}_ky"][0]) * dky
        total_flucs = np.sum(mine["ky"]) * dky
        print(
            f"{name:40s}"
            + "".join(f"{e:>11s}" for e in row)
            + f"   {total_flucs: .6e} vs {total_curvy: .6e}"
        )

    # Scalars saved by flucs against the curvy3d totals
    print("\nflucs scalars x 2 against curvy3d totals")
    for name, curvy_name in [*((t, t) for t in TERMS), *ENERGIES.items()]:
        total_curvy = np.sum(curvy[f"{curvy_name}_kx"][0]) * 2 * np.pi / pars.Lx
        scalar = 2 * float(diag[name][args.save])
        print(f"   {name:40s} {scalar: .10e} {total_curvy: .10e}")

    print(f"\nWorst relative spectral difference: {worst:.3e}")
    dataset.close()


if __name__ == "__main__":
    main()
