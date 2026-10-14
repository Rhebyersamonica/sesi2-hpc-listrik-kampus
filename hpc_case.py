"""
Studi Kasus: Menjembatani Infrastruktur HPC untuk Masalah Lokal
Berdasarkan webinar "Bridging Global Infrastructure to Address Local Challenge with HPC"
(Dr. Worawan Diaz Carballo, Thammasat University, 5 Oktober 2026).

Masalah lokal (terinspirasi proyek "Twin-B" di materi): tagihan listrik kampus.
Kebijakan penggunaan AC (jam hidup, jam mati, suhu setelan) diuji lewat simulasi berbasis agen
(Monte Carlo), lalu dijalankan PARALEL agar terlihat manfaat komputasi kinerja tinggi.

Alur (sesuai materi): masalah lokal -> workflow -> komputasi paralel -> hasil -> keputusan manusia.
Data SINTETIS. Jalankan:  python hpc_case.py --workers 4 --replicates 20
"""
# ----- SETUP
import argparse, os, time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT = "output"
os.makedirs(OUT, exist_ok=True)
TARIF_RP_PER_KWH = 1444.70      # asumsi tarif listrik (ubah sesuai kebutuhan)
N_CLASS, N_LIB, N_LAB = 30, 4, 6  # ruang kelas, perpustakaan, laboratorium
N_ROOMS = N_CLASS + N_LIB + N_LAB
DAYS = 30


# ----- 1. MINI-LINPACK
def mini_linpack(sizes=(500, 1000, 2000, 3000), repeats=3):
    """Mengukur FLOPS dengan menyelesaikan Ax=b (ide benchmark LINPACK/TOP500 di materi).
    Operasi ~ (2/3)n^3 + 2n^2 floating point operation."""
    rows = []
    for dtype in (np.float64, np.float32):
        for n in sizes:
            rng = np.random.default_rng(0)
            A = rng.random((n, n)).astype(dtype) + n * np.eye(n, dtype=dtype)
            b = rng.random(n).astype(dtype)
            best = min(_time(lambda: np.linalg.solve(A, b)) for _ in range(repeats))
            flop = (2 / 3) * n**3 + 2 * n**2
            rows.append({"dtype": np.dtype(dtype).name, "n": n, "waktu_s": best,
                         "GFLOPS": flop / best / 1e9})
    return pd.DataFrame(rows)


def _time(fn):
    t = time.perf_counter(); fn(); return time.perf_counter() - t


# ----- 2. SIMULASI BERBASIS AGEN
def simulate(args):
    """Satu replikasi simulasi 30 hari per jam.
    policy = (jam_on, jam_off, setpoint, perpus_tutup_akhir_pekan). Lab WAJIB tetap menyala 24 jam
    (constraint dari materi: listrik lab biologi tidak boleh dimatikan)."""
    (on, off, setpoint, lib_closed), seed = args
    rng = np.random.default_rng(seed)
    base_kw = rng.uniform(1.5, 3.5, N_ROOMS)              # daya AC tiap ruang
    idx = np.arange(N_ROOMS)
    is_lib = (idx >= N_CLASS) & (idx < N_CLASS + N_LIB)
    is_lab = idx >= N_CLASS + N_LIB
    kwh = occ_hours = blocked = penalty = 0.0
    for d in range(DAYS):
        weekend = (d % 7) in (5, 6)
        for h in range(24):
            t_out = 29 + 5 * np.sin((h - 9) / 24 * 2 * np.pi) + rng.normal(0, 1)
            p = np.full(N_ROOMS, 0.02)
            if 7 <= h <= 18 and not weekend:
                p[:N_CLASS] = 0.55
            lib_open = (8 <= h <= 20) if not weekend else (8 <= h <= 16 and not lib_closed)
            p[is_lib] = 0.5 if lib_open else 0.0
            p[is_lab] = 0.5 if 8 <= h <= 18 else 0.3
            occ = rng.random(N_ROOMS) < p
            allowed = np.full(N_ROOMS, on <= h < off)
            allowed[is_lab] = True
            if weekend and lib_closed:
                allowed[is_lib] = False
            ac = allowed & occ
            kwh += float((base_kw[ac] * max(t_out - setpoint, 0) / 8).sum())
            occ_hours += occ.sum()
            blocked += (occ & ~allowed & (t_out > 28)).sum()
            if setpoint >= 27:
                penalty += 0.5 * ac.sum()
    return {"kwh": kwh, "discomfort_rate": (blocked + penalty) / max(occ_hours, 1)}


def policy_grid():
    return [(on, off, sp, True) for on in (7, 8, 9) for off in (16, 18, 20) for sp in (22, 24, 26, 27)]


BASELINE = (0, 24, 22, False)   # AC boleh menyala kapan saja, 22 C, perpustakaan buka akhir pekan


# ----- 3. KOMPUTASI PARALEL
def run_all(policies, replicates, workers):
    jobs = [(p, 1000 * i + r) for i, p in enumerate(policies) for r in range(replicates)]
    t0 = time.perf_counter()
    if workers == 1:
        res = [simulate(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(simulate, jobs, chunksize=max(1, len(jobs) // (workers * 4))))
    elapsed = time.perf_counter() - t0
    df = pd.DataFrame(res)
    df["policy"] = [j[0] for j in jobs]
    return df, elapsed


def scaling_test(policies, replicates=5, max_workers=None):
    """Strong scaling: pekerjaan tetap, jumlah worker bertambah."""
    max_workers = max_workers or os.cpu_count()
    ws = sorted({1, 2, 4, 8, 16, max_workers} & set(range(1, max_workers + 1)))
    rows, t1 = [], None
    for w in ws:
        _, t = run_all(policies, replicates, w)
        t1 = t1 or t
        rows.append({"workers": w, "waktu_s": t, "speedup": t1 / t, "efisiensi": t1 / t / w})
    df = pd.DataFrame(rows)
    plt.figure(figsize=(5.5, 4))
    plt.plot(df.workers, df.speedup, "o-", label="Aktual")
    plt.plot(df.workers, df.workers, "--", label="Ideal")
    plt.xlabel("Jumlah worker"); plt.ylabel("Speedup"); plt.legend(); plt.title("Strong scaling")
    plt.tight_layout(); plt.savefig(f"{OUT}/02_scaling.png", dpi=130); plt.close()
    return df


# ----- 4. ANALISIS & KEPUTUSAN
def analyse(df, base_df, max_discomfort=0.10):
    base_kwh = base_df.kwh.mean()
    agg = df.groupby("policy").agg(kwh=("kwh", "mean"), kwh_std=("kwh", "std"),
                                   discomfort_rate=("discomfort_rate", "mean")).reset_index()
    agg["hemat_%"] = (1 - agg.kwh / base_kwh) * 100
    agg["hemat_rp_per_bulan"] = (base_kwh - agg.kwh) * TARIF_RP_PER_KWH
    agg[["jam_on", "jam_off", "setpoint", "_"]] = pd.DataFrame(agg.policy.tolist())
    agg = agg.drop(columns=["policy", "_"])
    ok = agg[agg.discomfort_rate <= max_discomfort].sort_values("hemat_%", ascending=False)

    plt.figure(figsize=(7, 4.5))
    sc = plt.scatter(agg.discomfort_rate * 100, agg["hemat_%"], c=agg.setpoint, cmap="viridis")
    plt.colorbar(sc, label="Setpoint AC (C)")
    plt.axvline(max_discomfort * 100, color="red", ls="--", label=f"Batas ketidaknyamanan {max_discomfort:.0%}")
    plt.xlabel("Tingkat ketidaknyamanan (%)"); plt.ylabel("Penghematan listrik (%)")
    plt.title("Trade-off hemat energi vs kenyamanan"); plt.legend(); plt.tight_layout()
    plt.savefig(f"{OUT}/03_tradeoff.png", dpi=130); plt.close()
    agg.round(2).to_csv(f"{OUT}/hasil_kebijakan.csv", index=False)
    return agg, ok, base_kwh


# ----- 5. SKRIP SLURM
def write_sbatch(path="job.sbatch"):
    """Contoh skrip batch SLURM, seperti yang ditunjukkan di materi (minta resource, submit job)."""
    s = """#!/bin/bash
#SBATCH -J twin-listrik-kampus
#SBATCH -N 1                      # jumlah node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16        # jumlah core yang diminta
#SBATCH -t 00:30:00               # batas waktu
#SBATCH -p compute                # nama partisi (sesuaikan dengan cluster)
#SBATCH -o slurm-%j.out

# module load Python              # sesuaikan dengan cluster
python hpc_case.py --workers $SLURM_CPUS_PER_TASK --replicates 200
"""
    open(path, "w").write(s)
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--replicates", type=int, default=20)
    ap.add_argument("--skip-scaling", action="store_true")
    a = ap.parse_args()
    pd.set_option("display.width", 160, "display.max_columns", 20)

    print("== 1) Mini-LINPACK ==");  print(mini_linpack().round(2))
    pol = policy_grid()
    if not a.skip_scaling:
        print("\n== 2) Strong scaling =="); print(scaling_test(pol[:12]).round(2))
    print(f"\n== 3) Simulasi {len(pol)} kebijakan x {a.replicates} replikasi, {a.workers} worker ==")
    df, t = run_all(pol, a.replicates, a.workers)
    base, _ = run_all([BASELINE], a.replicates, a.workers)
    print(f"Selesai dalam {t:.1f} s")
    agg, ok, base_kwh = analyse(df, base)
    print(f"Baseline: {base_kwh:,.0f} kWh/bulan\n")
    print(ok.head(5).round(2))
    write_sbatch()
