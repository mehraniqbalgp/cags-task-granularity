#!/usr/bin/env python3
"""Run every experiment reported in the paper and append rows to results/raw.csv.

usage: python3 bench/run_all.py [--quick]
--quick uses 1 repetition and fewer thread counts (a smoke test, not for the paper).
"""
import os, platform, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "cagran")
OUT = os.path.join(ROOT, "results")
QUICK = "--quick" in sys.argv

SIZES = {"fib": 42, "nqueens": 13, "msort": 1 << 24, "loop": 1 << 22}
PS = [1, 12] if QUICK else [1, 2, 4, 6, 12]
REPS = 1 if QUICK else 5
SWEEP_REPS = 1 if QUICK else 3
ORACLE = {  # fixed cutoffs swept to find the best one per P (oracle baseline)
    "fib": list(range(16, 41, 2)),
    "nqueens": list(range(2, 13)),
    "msort": [1 << e for e in range(7, 24, 2)],
    "loop": [1 << e for e in range(6, 21, 2)],
}
LOOP_BASELINES = ["static", "dynamic", "guided", "tl_default"]
ETAS = [0.5, 0.8, 0.9, 0.95, 0.99]
HEADER = "kernel,variant,P,rep,wall_s,cpu_s,tasks,param,h_us,theta_us,select_s,ok,experiment,eta,size\n"


def env_for(P, eta=None):
    e = dict(os.environ)
    # one thread per physical core up to the core count, then fill SMT siblings
    cores = os.cpu_count() // 2 or 1
    e["OMP_PLACES"], e["OMP_PROC_BIND"] = ("cores", "spread") if P <= cores else ("threads", "close")
    if eta is not None:
        e["CAGS_ETA"] = str(eta)
    return e


def run(f, exp, kernel, variant, P, reps, size, param=None, eta=0.9):
    cmd = [BIN, kernel, variant, str(P), str(reps), str(size)] + ([str(param)] if param is not None else [])
    r = subprocess.run(cmd, env=env_for(P, eta), capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}\n{r.stderr}")
    for line in r.stdout.strip().splitlines():
        f.write(f"{line},{exp},{eta},{size}\n")
    f.flush()


def main():
    os.makedirs(OUT, exist_ok=True)
    subprocess.run(["make", "-s", "-C", ROOT], check=True)
    raw = os.path.join(OUT, "raw_quick.csv" if QUICK else "raw.csv")
    with open(os.path.join(OUT, "environment.txt"), "w") as f:
        f.write(f"date: {time.strftime('%Y-%m-%d %H:%M:%S %z')}\n")
        f.write(f"platform: {platform.platform()}\n")
        f.write(subprocess.run(["lscpu"], capture_output=True, text=True).stdout)
        f.write(subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0] + "\n")
    t0 = time.time()
    with open(raw, "w") as f:
        f.write(HEADER)
        for k, n in SIZES.items():
            print(f"[{time.time()-t0:6.0f}s] {k}: sequential", flush=True)
            run(f, "main", k, "seq", 1, REPS, n)
            for P in PS:
                print(f"[{time.time()-t0:6.0f}s] {k}: P={P}", flush=True)
                run(f, "main", k, "cags", P, REPS, n)
                for v in (LOOP_BASELINES if k == "loop" else ["depth"]):
                    run(f, "main", k, v, P, REPS, n)
                for c in ORACLE[k]:
                    run(f, "oracle", k, "fixed", P, SWEEP_REPS, n, c)
            Pmax = PS[-1]
            print(f"[{time.time()-t0:6.0f}s] {k}: eta sensitivity and fine tasks", flush=True)
            for eta in ETAS:
                run(f, "eta", k, "cags", Pmax, REPS, n, eta=eta)
            # every-call tasking at the main size is prohibitive for fib (~8e8 tasks): use fib(34)
            nf = 34 if k == "fib" else n
            if nf != n:
                run(f, "fine", k, "seq", 1, SWEEP_REPS, nf)
                run(f, "fine", k, "cags", Pmax, SWEEP_REPS, nf)
            run(f, "fine", k, "fine", Pmax, SWEEP_REPS, nf)
    print(f"done in {time.time()-t0:.0f}s -> {raw}")


if __name__ == "__main__":
    main()
