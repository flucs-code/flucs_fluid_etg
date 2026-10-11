"""Streamer runs that ramp up the scale separation in stages.

Each kappaN gets a chain of stages. Stage 0 starts from the predicted streamer
([parameters.streamer]); every later stage starts from the previous stage's
restart file on a finer grid with a smaller nu_perp. Stage s targets the
separation R_s = k_d / ky_I between the dissipation scale
k_d = (dT_I ky_I / nu_perp)^(1/3) and the streamer, so

    nu_perp = dT_I / (R_s^3 ky_I^2),

with ky_I and dT_I = C_T gamma_lin(ky_I) / ky_I predicted exactly as the run
does (ky_I = ky_over_kpeak * ky_peak, snapped to a box harmonic). The grid
resolves kmax = kmax_over_kd * k_d in kx and ky. The fuzz has
kz ~ kperp^(1/2), so the parallel grid resolves kz_max = c_z kmax^(1/2), with
c_z = 1 from the curvy3d grids (nx, nz) = (323, 47) and (863, 71) at
Lx = Lz = 20, which both have kz_max = (1.0 +- 0.05) kmax^(1/2); nz is at
least NZ_MIN, and --nz-unpadded > 0 fixes it instead.

Usage (in an environment with flucs and flucs_fluid_etg installed)::

    python streamer_ladder.py plan
    python streamer_ladder.py write RUNS_DIR
    python streamer_ladder.py submit RUNS_DIR --gpus 2080,4090,a100,h200

``plan`` prints ky_I, dT_I, nu_perp, the grid and an estimate of the GPU
memory of every stage. ``write`` creates RUNS_DIR/kappaN_<value>/stage_<s>/
with input.toml and job.sh. ``submit`` submits every chain with sbatch, each
stage depending on the previous one (afterok); --gpus gives one GPU type per
stage (the last is reused), --dry-run prints the commands only. The job
script reuses the #SBATCH header of SUBMIT_DIR/submit_<gpu>.cmd, copies the
previous stage's restart.nc in if the stage has none, and runs flucs.
Every stage uses restart_if_exists, so a resubmitted stage resumes from its
own restart file.

The defaults (box, stages, run times, output) are the constants below and
the command-line options.
"""

import argparse
import contextlib
import io
import math
import pathlib as pl
import subprocess
import tempfile

import numpy as np

KAPPA_N_CRIT = 0.5641905229536246
KAPPA_N = (0.55, 0.54, 0.53, 0.52, 0.51, 0.50, 0.475, 0.45)
SEPARATIONS = (10.0, 25.0, 50.0, 100.0)  # R_s = k_d / ky_I per stage

LX = LY = 15.0
LZ = 20.0
KZ_OVER_SQRT_KMAX = 1.0  # c_z in kz_max = c_z kmax^(1/2)
NZ_MIN = 15
NZ_UNPADDED = -1  # > 0 fixes nz in every stage
KMAX_OVER_KD = 0.6  # curvy3d kappaN = 0.45, nu = 1e-6, nx = 323 had 0.52
KY_OVER_KPEAK = 0.85
C_T = 13.0

# Run time of each stage in units of 1/gamma_lin(ky_I): stage 0 saturates
# from the initial condition, later stages re-adjust after nu_perp drops
STAGE_TIME_GAMMA = (40.0, 15.0)

# Steps between saves at the stage-0 time step, scaled with dt_max0/dt_max
# so that saves stay roughly evenly spaced in time
SAVE_STEPS_0D = 100
SAVE_STEPS_SPECTRA = 2000
SAVE_STEPS_REALSPACE = 20000
RESTART_STEPS = 5000

# The [N/N] Pade propagator of the linear terms is accurate only while
# dt * (largest linear damping rate) stays below about 3.5 (checked with
# flucs' linear-propagator report). The largest rate is the parallel damping
# at kz_max, so every stage gets dt_max = min(DT_MAX, PADE_LIMIT / rate).
# Hyperdissipation is applied exactly and does not enter.
DT_MAX = 0.05
PADE_LIMIT = 3.0

# Wall time estimate: seconds per padded grid point per RK4 step, measured
# as 1.0e-8 on an MX550 and scaled by memory bandwidth (about 2x uncertain)
SECONDS_PER_POINT_STEP = {"a100": 5e-10, "h200": 3e-10, "rtx6000": 1e-9, "4090": 1e-9}

# GPU memory estimate, bytes per padded grid point (single precision) and
# a fixed overhead, from flucs runs with these diagnostics
BYTES_PER_POINT = 120
OVERHEAD_GB = 0.5

SUBMIT_DIR = "/mnt/users/turica/source/flucs_submit_cmd"


def odd_unpadded(kmax, L):
    """Smallest odd unpadded n whose largest wavenumber reaches kmax."""
    return 2 * math.ceil(kmax * L / (2.0 * math.pi)) + 1


def padded(n_unpadded):
    """flucs' padded size for two-thirds dealiasing: 3-smooth >= 3 half_n."""
    target = 3 * (n_unpadded // 2 + 1)
    n = target
    while True:
        m = n
        for p in (2, 3):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def streamer_prediction(kappaN, args):
    """ky_I, its index, gamma_lin and dT_I, from the flucs system itself."""
    from flucs.input import FlucsInput

    toml = f"""
[setup]
solver = "FourierSolver"
system = "CollisionalETGFourier"
[dimensions]
Lx = {args.Lx}
Ly = {args.Ly}
Lz = {args.Lz}
[dealiasing]
method = "two-thirds"
nx_unpadded = 2049
ny_unpadded = 2049
nz_unpadded = 3
[parameters]
kappaT = 1.0
kappaN = {kappaN}
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = pl.Path(tmp) / "input.toml"
        path.write_text(toml)
        with contextlib.redirect_stdout(io.StringIO()):
            _, system = FlucsInput(path).create_solver_system()

    ky_marg = system._find_marginal_ky()
    ky_peak = system._find_peak_ky(ky_marg)
    dky = 2.0 * np.pi / args.Ly
    iky = int(np.rint(args.ky_over_kpeak * ky_peak / dky))
    ky = iky * dky
    kz = 2.0 * np.pi / args.Lz
    gamma = float(system.compute_most_unstable_mode_cpu(kz, 0.0, ky)[0].imag)

    def max_damping(n_z):
        """Largest linear damping rate on a grid with n_z unpadded kz modes."""
        kz_max = (n_z // 2) * 2.0 * np.pi / args.Lz
        ky_scan = np.linspace(0.0, 4.0 * ky, 200)
        matrix = np.moveaxis(
            system.compute_linear_matrix_cpu(kz_max, 0.0, ky_scan), (0, 1), (-2, -1)
        )
        # dfields/dt = -M fields: the damping rates are Re(eigenvalues of M)
        return float(np.max(np.linalg.eigvals(matrix).real))

    return {
        "max_damping": max_damping,
        "ky_marg": ky_marg,
        "ky_peak": ky_peak,
        "iky": iky,
        "ky": ky,
        "gamma": gamma,
        "dT": args.C_T * gamma / ky,
    }


def plan(args):
    """One record per (kappaN, stage)."""
    records = []
    for kappaN in args.kappaN:
        p = streamer_prediction(kappaN, args)
        n0 = None
        dt0 = None
        for stage, R in enumerate(args.separations):
            nu = p["dT"] / (R**3 * p["ky"] ** 2)
            k_d = R * p["ky"]
            kmax = args.kmax_over_kd * k_d
            n = odd_unpadded(kmax, args.Lx)
            if args.nz_unpadded > 0:
                n_z = args.nz_unpadded
            else:
                kz_max = args.kz_over_sqrt_kmax * np.sqrt(kmax)
                n_z = max(odd_unpadded(kz_max, args.Lz), NZ_MIN)
            n0 = n0 or n
            points = padded(n) ** 2 * padded(n_z)
            dt_max = min(DT_MAX, PADE_LIMIT / p["max_damping"](n_z))
            dt0 = dt0 or dt_max
            time_gamma = args.stage_time_gamma[
                min(stage, len(args.stage_time_gamma) - 1)
            ]
            records.append(
                {
                    "kappaN": kappaN,
                    "Lambda": KAPPA_N_CRIT - kappaN,
                    "stage": stage,
                    "R": R,
                    "nu": nu,
                    "k_d": k_d,
                    "n": n,
                    "nz": n_z,
                    "kmax_over_kd": (n // 2) * 2.0 * np.pi / args.Lx / k_d,
                    "memory_GB": OVERHEAD_GB + BYTES_PER_POINT * points / 1e9,
                    "tfinal": time_gamma / p["gamma"],
                    "steps": time_gamma / p["gamma"] / dt_max,
                    "grid_scale": n / n0,
                    "dt_max": dt_max,
                    "dt_scale": dt0 / dt_max,
                    "points": points,
                    **{key: value for key, value in p.items() if key != "max_damping"},
                }
            )
    return records


def print_plan(records):
    print(
        f"{'kappaN':>7}{'Lambda':>8}{'stage':>6}{'ky_I':>7}{'iky':>5}"
        f"{'gamma':>9}{'dT_I':>8}{'R':>6}{'nu_perp':>10}{'k_d':>8}"
        f"{'n_xy':>6}{'n_z':>5}{'kmax/k_d':>9}{'mem GB':>8}{'tfinal':>9}"
        f"{'dt_max':>9}{'steps':>9}{'A100 h':>8}"
    )
    for r in records:
        print(
            f"{r['kappaN']:7.3f}{r['Lambda']:8.3f}{r['stage']:6d}{r['ky']:7.3f}"
            f"{r['iky']:5d}{r['gamma']:9.4f}{r['dT']:8.4f}{r['R']:6.0f}"
            f"{r['nu']:10.2e}{r['k_d']:8.1f}{r['n']:6d}{r['nz']:5d}"
            f"{r['kmax_over_kd']:9.2f}{r['memory_GB']:8.1f}{r['tfinal']:9.0f}"
            f"{r['dt_max']:9.1e}{r['steps']:9.1e}"
            f"{r['steps'] * r['points'] * SECONDS_PER_POINT_STEP['a100'] / 3600:8.1f}"
        )


def input_toml(r, args):
    scale = max(1, round(r["dt_scale"]))
    return f"""# kappaN = {r["kappaN"]}, stage {r["stage"]}: k_d/ky_I = {r["R"]:g}
# Predicted ky_I = {r["ky"]:.4f} (iky {r["iky"]}), gamma_lin = {r["gamma"]:.4e},
# dT_I = {r["dT"]:.4e}, k_d = {r["k_d"]:.2f}; written by streamer_ladder.py

[setup]
solver = "FourierSolver"
system = "CollisionalETGFourier"
linear = false
precision = "single"

[restart]
write_restart_file = true
restart_if_exists = true
write_steps = {RESTART_STEPS * scale}

[dimensions]
Lx = {args.Lx}
Ly = {args.Ly}
Lz = {args.Lz}

[dealiasing]
method = "two-thirds"
nx_unpadded = {r["n"]}
ny_unpadded = {r["n"]}
nz_unpadded = {r["nz"]}

[time]
dt_max = {r["dt_max"]:.3e}
max_cfl = 0.8
tfinal = {r["tfinal"]:.1f}

[timestepping]
method = "rk4"

[parameters]
kappaT = 1.0
kappaN = {r["kappaN"]}

[hyperdissipation]
kperp = {r["nu"]:.6e}
kperp_power = 2
kperp_normalised = false

# Used for the initial condition of stage 0 only (later stages restart); it
# also fixes the streamer ky of the streamer_exchange diagnostic
[parameters.streamer]
mode = "free"
ky_over_kpeak = {args.ky_over_kpeak}
cutoff_factor = 2.0
amplitude_mode = "predicted"
C_T = {args.C_T}
kz_structure = "both"
kx_width = 0.0
seed_mode = "predicted"
seed_kz_power = -3.0
seed_kz0_fraction = 0.6

[output]
write_steps = {50 * SAVE_STEPS_0D * scale}

[output.0d]
save_steps = {SAVE_STEPS_0D * scale}
diags = [
    {{name = "heatflux"}},
    {{name = "free_energy"}},
    {{name = "streamer_exchange", options = {{spectra = []}}}},
]

[output.spectra]
save_steps = {SAVE_STEPS_SPECTRA * scale}
diags = [
    {{name = "streamer_exchange", options = {{spectra = ["kx", "ky", "kz", "kperp", "kxky"]}}}},
    {{name = "spectra", options = {{spectra = ["kx", "ky", "kz", "kperp", "kxky"]}}}},
    {{name = "spectral_free_energy", options = {{spectra = ["kx", "ky", "kz", "kperp", "kxky"]}}}},
]

# Real-space snapshots of (phi, dT) in the planes z = 0 and x = 0
[output.realspace]
save_steps = {SAVE_STEPS_REALSPACE * scale}
diags = [
    {{name = "realspace_data", options = {{locations = [":, 0, :, :", ":, :, 0, :"]}}}},
]
"""


def stage_dir(root, r):
    return pl.Path(root) / f"kappaN_{r['kappaN']:.3f}" / f"stage_{r['stage']}"


def job_script(gpu, args):
    """#SBATCH header of the user's submit_<gpu>.cmd, then copy and run."""
    cmd = pl.Path(args.submit_dir) / f"submit_{gpu}.cmd"
    lines = cmd.read_text().splitlines()
    body = [line for line in lines if line.strip() != "flucs"]
    body += [
        "",
        "# Start from the previous stage unless this stage has a restart file",
        "if [ ! -f restart.nc ] && [ -f ../PREVIOUS/restart.nc ]; then",
        "    cp ../PREVIOUS/restart.nc restart.nc",
        '    echo "Copied ../PREVIOUS/restart.nc"',
        "fi",
        "",
        "flucs",
        "",
    ]
    return "\n".join(body)


def write(args):
    records = plan(args)
    print_plan(records)
    for r in records:
        directory = stage_dir(args.runs_dir, r)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "input.toml").write_text(input_toml(r, args))
    print(f"\nWrote {len(records)} input files under {args.runs_dir}")


def submit(args):
    gpus = args.gpus.split(",")
    root = pl.Path(args.runs_dir)
    for chain in sorted(root.glob("kappaN_*")):
        stages = sorted(chain.glob("stage_*"), key=lambda p: int(p.name.split("_")[1]))
        job_id = None
        for stage in stages:
            index = int(stage.name.split("_")[1])
            gpu = gpus[min(index, len(gpus) - 1)]
            script = job_script(gpu, args).replace(
                "PREVIOUS", f"stage_{index - 1}" if index > 0 else "NONE"
            )
            (stage / "job.sh").write_text(script)

            command = ["sbatch", "--parsable", f"--chdir={stage}"]
            if job_id is not None:
                command.append(f"--dependency=afterok:{job_id}")
            command.append(str(stage / "job.sh"))

            if args.dry_run:
                print(" ".join(command))
                job_id = f"<{chain.name}/{stage.name}>"
                continue

            result = subprocess.run(command, capture_output=True, text=True, check=True)
            job_id = result.stdout.strip().split(";")[0]
            print(f"{chain.name}/{stage.name} on {gpu}: job {job_id}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("plan", "write", "submit"))
    parser.add_argument("runs_dir", nargs="?", default="streamer_ladder_runs")
    parser.add_argument("--kappaN", type=float, nargs="+", default=list(KAPPA_N))
    parser.add_argument(
        "--separations", type=float, nargs="+", default=list(SEPARATIONS)
    )
    parser.add_argument("--Lx", type=float, default=LX)
    parser.add_argument("--Ly", type=float, default=LY)
    parser.add_argument("--Lz", type=float, default=LZ)
    parser.add_argument("--nz-unpadded", type=int, default=NZ_UNPADDED)
    parser.add_argument("--kz-over-sqrt-kmax", type=float, default=KZ_OVER_SQRT_KMAX)
    parser.add_argument("--kmax-over-kd", type=float, default=KMAX_OVER_KD)
    parser.add_argument("--ky-over-kpeak", type=float, default=KY_OVER_KPEAK)
    parser.add_argument("--C-T", type=float, default=C_T)
    parser.add_argument(
        "--stage-time-gamma", type=float, nargs="+", default=list(STAGE_TIME_GAMMA)
    )
    parser.add_argument("--gpus", default="4090,4090,a100,h200")
    parser.add_argument("--submit-dir", default=SUBMIT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    match args.command:
        case "plan":
            print_plan(plan(args))
        case "write":
            write(args)
        case "submit":
            submit(args)


if __name__ == "__main__":
    main()
