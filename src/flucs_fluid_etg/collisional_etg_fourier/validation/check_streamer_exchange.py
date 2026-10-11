"""Self-consistency checks of the streamer_exchange diagnostic.

Usage::

    python check_streamer_exchange.py RUN_DIR [--group N] [--budget-only]

RUN_DIR must hold an output file with the streamer_exchange diagnostic, saved
with the kxkykz contraction. If the same file also holds spectral_free_energy
with the kxkykz contraction, check 1 compares against its dWdt_nonlinear.
Saving every step makes check 4 accurate to the time-stepping order.

1. Splits add up: sum over (d, a, b) of T_{d,ab}(k) equals dWdt_nonlinear(k)
   mode by mode.
2. Identities, summed over k: Pi_sf + Pi_fs = 0, leak + mirror = 0, and the
   Batchelor, small-flow-on-streamer and self terms vanish.
3. Contractions: every saved contraction (kx, ky, kz, kperp, kzkperp, kxky)
   equals the same reduction of the kxkykz data done here with numpy, and the
   scalars equal their sums. This guards the ky > 0 conjugate weighting.
4. Budget: dW_d/dt from centred differences of W_d equals
   inj_d + coll_d + hyper_d + sum_ab T_{d,ab}, for d = I, II.
"""

import argparse
import pathlib as pl

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
DESTINATION = {
    "nl_streamer_to_small": "II",
    "nl_small_to_streamer": "I",
    "nl_batchelor": "II",
    "nl_small_self": "II",
    "nl_streamer_self": "I",
    "nl_streamer_self_to_small": "II",
    "nl_small_to_streamer_via_streamer_flow": "I",
    "nl_small_flow_on_streamer": "I",
}


def find_group(run_dir, group=None):
    """Returns the netCDF group holding streamer_exchange."""
    for path in sorted(pl.Path(run_dir).glob("output*.nc")):
        dataset = nc.Dataset(path)
        for name, sub in dataset.groups.items():
            if group is not None and name != str(group):
                continue
            if "streamer_exchange" in sub.groups:
                return dataset, sub
        dataset.close()
    raise FileNotFoundError(f"No streamer_exchange output in {run_dir}")


def ky_weight(n_ky):
    weight = np.full(n_ky, 2.0)
    weight[0] = 1.0
    return weight


def reduce_like_flucs(full, spectrum, kx, ky, kperp_bins):
    """Reduces (kz, kx, ky) half-grid data like FourierReductions: kx, kz, kperp
    and kzkperp count the conjugate (ky < 0) modes, ky and kxky do not."""
    weighted = full * ky_weight(len(ky))[None, None, :]
    match spectrum:
        case "kz":
            return weighted.sum(axis=(1, 2))
        case "kx":
            return weighted.sum(axis=(0, 2))
        case "ky":
            # Middle-axis reductions do not double ky > 0
            return full.sum(axis=(0, 1))
        case "kxky":
            return full.sum(axis=0)
        case "kperp" | "kzkperp":
            # Uniform half-open shells from kperp_bins[0], with the bin index
            # evaluated exactly as in the CUDA shell sum. Many modes sit on
            # shell edges, so the arithmetic matters.
            nk = len(kperp_bins)
            kmin = kperp_bins[0]
            kmax = kmin + nk * (kperp_bins[1] - kperp_bins[0])
            kperp = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
            index = ((kperp - kmin) * (nk / (kmax - kmin))).astype(int)
            # Each ky > 0 mode is joined by the conjugate of its partner at
            # (-kz, -kx, ky), which is the mode at (kz, kx, -ky)
            nz, nx = full.shape[:2]
            partner = full[(-np.arange(nz)) % nz][:, (-np.arange(nx)) % nx]
            shell_data = full.copy()
            shell_data[:, :, 1:] += partner[:, :, 1:]
            out = np.zeros((nz, nk))
            valid = index < nk
            for iz in range(nz):
                np.add.at(out[iz], index[valid], shell_data[iz][valid])
            return out if spectrum == "kzkperp" else out.sum(axis=0)
    raise ValueError(spectrum)


def relative(a, b, scale=None):
    scale = np.max(np.abs(b)) if scale is None else scale
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / scale if scale else 0.0


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir")
    parser.add_argument("--group", default=None)
    parser.add_argument("--budget-only", action="store_true")
    args = parser.parse_args()

    dataset, group = find_group(args.run_dir, args.group)
    diag = group["streamer_exchange"]
    time = np.asarray(group["time"][:])

    def scalar(name):
        return np.asarray(diag[name][:], dtype=float)

    print(f"{dataset.filepath()}, group {group.name}: {len(time)} saves")
    print(
        f"ky_I = {scalar('ky_streamer')[-1]:.4f}, kc = {scalar('kperp_cutoff')[-1]:.4f}, "
        f"iky = {int(scalar('iky_streamer')[-1])}"
    )

    # Skip the t = 0 save: there the previous fields are zero
    saves = slice(1, None)
    scale = np.max(np.abs(scalar("nl_streamer_to_small")[saves]))
    results = {}

    if not args.budget_only:
        # 1. Splits add up
        full = "kxkykz_spectra/"
        if "spectral_free_energy" in group.groups:
            reference = group["spectral_free_energy"][full + "dWdt_nonlinear"][:]
            total = sum(np.asarray(diag[full + term][:]) for term in TERMS)
            error = relative(total[saves], reference[saves])
            results["1 splits add up (max |sum - dWdt_nonlinear| / max |.|)"] = error
        else:
            print("1. skipped: no spectral_free_energy kxkykz output")

        # 2. Identities
        identities = {
            "Pi_sf + Pi_fs": scalar("nl_streamer_to_small")
            + scalar("nl_small_to_streamer"),
            "leak + mirror": scalar("nl_streamer_self_to_small")
            + scalar("nl_small_to_streamer_via_streamer_flow"),
            "Batchelor": scalar("nl_batchelor"),
            "small flow on streamer": scalar("nl_small_flow_on_streamer"),
            "small self": scalar("nl_small_self"),
            "streamer self": scalar("nl_streamer_self"),
        }
        for name, value in identities.items():
            results[f"2 {name} (max |.| / max |Pi_sf|)"] = (
                np.max(np.abs(value[saves])) / scale
            )

        # 3. Contractions against numpy reductions of kxkykz
        # Wavenumbers as in kx_from_ikx and ky_from_iky
        inputs = toml.loads(str(group["input_file"][...]))
        nx, n_ky = len(diag["kx"]), len(diag["ky"])
        kx = (2.0 * np.pi / inputs["dimensions"]["Lx"]) * np.rint(
            np.fft.fftfreq(nx) * nx
        )
        ky = (2.0 * np.pi / inputs["dimensions"]["Ly"]) * np.arange(n_ky)
        kperp_bins = np.asarray(diag["kperp"][:]) if "kperp" in diag.variables else None
        worst = {}
        for name in (*TERMS, "W_I", "W_II"):
            # Exchange terms are compared on the scale of Pi_sf, since several
            # of them vanish
            reference = "nl_streamer_to_small" if name in TERMS else name
            reference_data = np.asarray(diag[full + reference][:])[saves]
            full_data = np.asarray(diag[full + name][:])[saves]
            for spectrum in ("kx", "ky", "kz", "kperp", "kzkperp", "kxky"):
                subgroup = f"{spectrum}_spectra"
                if subgroup not in diag.groups:
                    continue
                saved = np.asarray(diag[subgroup][name][:])[saves]
                mine, mine_reference = (
                    np.stack(
                        [
                            reduce_like_flucs(f, spectrum, kx, ky, kperp_bins)
                            for f in data
                        ]
                    )
                    for data in (full_data, reference_data)
                )
                error = relative(saved, mine, np.max(np.abs(mine_reference)))
                worst[spectrum] = max(worst.get(spectrum, 0.0), error)
            weight = ky_weight(len(ky))
            summed = (full_data * weight).sum(axis=(1, 2, 3))
            summed_reference = (reference_data * weight).sum(axis=(1, 2, 3))
            error = relative(
                scalar(name)[saves], summed, np.max(np.abs(summed_reference))
            )
            worst["scalar"] = max(worst.get("scalar", 0.0), error)
        for spectrum, error in worst.items():
            results[f"3 {spectrum} contraction vs numpy (max relative)"] = error

    # 4. Budget closure with centred differences
    if len(time) >= 3:
        print("\n4. Budget closure, dW/dt (centred) vs sum of terms")
        for part in ("I", "II"):
            W = scalar(f"W_{part}")
            rates = (
                scalar(f"dWdt_inj_{part}")
                + scalar(f"dWdt_coll_{part}")
                + scalar(f"dWdt_hyperdissipation_{part}")
                + sum(scalar(t) for t in TERMS if DESTINATION[t] == part)
            )
            dWdt = (W[2:] - W[:-2]) / (time[2:] - time[:-2])
            residual = dWdt - rates[1:-1]
            error = np.max(np.abs(residual)) / np.max(np.abs(rates[1:-1]))
            results[f"4 budget {part} (max |dW/dt - terms| / max |terms|)"] = error
            print(
                f"   {part}: dW/dt = {dWdt[-1]:+.6e}, terms = {rates[-2]:+.6e}, "
                f"inj = {scalar(f'dWdt_inj_{part}')[-2]:+.4e}, "
                f"coll = {scalar(f'dWdt_coll_{part}')[-2]:+.4e}, "
                f"hyper = {scalar(f'dWdt_hyperdissipation_{part}')[-2]:+.4e}, "
                f"NL = {rates[-2] - scalar(f'dWdt_inj_{part}')[-2] - scalar(f'dWdt_coll_{part}')[-2] - scalar(f'dWdt_hyperdissipation_{part}')[-2]:+.4e}"
            )

    print("\nPi_sf (last save) =", f"{scalar('nl_streamer_to_small')[-1]:.6e}")
    print("Results:")
    for name, value in results.items():
        print(f"   {name}: {value:.3e}")

    dataset.close()


if __name__ == "__main__":
    main()
