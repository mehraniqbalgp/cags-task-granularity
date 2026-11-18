#!/usr/bin/env python3
"""Summarize results/raw.csv into summary CSVs, LaTeX tables/figures and PNG plots.

usage: python3 bench/analyze.py [results/raw.csv]
Statistic: median over repetitions. "Oracle" = best fixed cutoff found by an exhaustive sweep
at each P (its search cost is NOT included, and taking the minimum of noisy medians favours it).
"""
import csv, os, statistics as st, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "raw.csv")
OUT = os.path.join(ROOT, "results")
TEX = os.path.join(OUT, "latex")
os.makedirs(TEX, exist_ok=True)

KERNELS = ["fib", "nqueens", "msort", "loop"]
KNAME = {"fib": "Fib(42)", "nqueens": "N-Queens(13)", "msort": "Mergesort($2^{24}$)", "loop": "Irregular loop($2^{22}$)"}
BASE = {"fib": ["depth"], "nqueens": ["depth"], "msort": ["depth"], "loop": ["static", "dynamic", "guided", "tl_default"]}
VNAME = {"depth": "depth cutoff", "static": "static", "dynamic": "dynamic,1", "guided": "guided", "tl_default": "taskloop default"}

rows = list(csv.DictReader(open(RAW)))
assert rows and all(r["ok"] == "1" for r in rows), "missing data or a correctness failure"
med = lambda xs: st.median(xs)
groups = defaultdict(list)
for r in rows:
    groups[(r["experiment"], r["kernel"], r["variant"], int(r["P"]), r["param"] if r["variant"] == "fixed" else "", r["eta"], r["size"])].append(r)


def agg(exp, k, v, P, param="", eta="0.9", size=None):
    rs = [r for key, g in groups.items() for r in g
          if key[:5] == (exp, k, v, P, param) and key[5] == eta and (size is None or key[6] == str(size))]
    if not rs:
        return None
    f = lambda c: med([float(r[c]) for r in rs])
    return dict(wall=f("wall_s"), cpu=f("cpu_s"), tasks=med([int(r["tasks"]) for r in rs]),
                param=med([int(r["param"]) for r in rs]), h=f("h_us"), theta=f("theta_us"), sel=f("select_s"), n=len(rs))


SIZES_ = {r["kernel"]: r["size"] for r in rows if r["experiment"] == "main"}
PS = sorted({int(r["P"]) for r in rows if r["experiment"] == "main"})
PMAX = PS[-1]
seq = {k: agg("main", k, "seq", 1) for k in KERNELS}


def oracle(k, P):
    cuts = sorted({key[4] for key in groups if key[:4] == ("oracle", k, "fixed", P)}, key=int)
    res = [(agg("oracle", k, "fixed", P, c), int(c)) for c in cuts]
    best = min(res, key=lambda x: x[0]["wall"])
    return best[0], best[1], res


# ---------------------------------------------------------------- summary.csv
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["kernel", "variant", "P", "median_wall_s", "median_cpu_s", "speedup_vs_seq", "cpu_inflation", "param", "h_us", "select_ms", "reps"])
    for k in KERNELS:
        for P in PS:
            for v in ["cags"] + BASE[k]:
                a = agg("main", k, v, P)
                w.writerow([k, v, P, f"{a['wall']:.6f}", f"{a['cpu']:.6f}", f"{seq[k]['wall']/a['wall']:.3f}",
                            f"{a['cpu']/seq[k]['cpu']:.3f}", a["param"], f"{a['h']:.3f}", f"{a['sel']*1e3:.2f}", a["n"]])
            o, c, _ = oracle(k, P)
            w.writerow([k, f"oracle(cut={c})", P, f"{o['wall']:.6f}", f"{o['cpu']:.6f}", f"{seq[k]['wall']/o['wall']:.3f}",
                        f"{o['cpu']/seq[k]['cpu']:.3f}", c, "", "", o["n"]])

# ------------------------------------------------------------ main table (P max)
lines = []
summary = {}
for k in KERNELS:
    c = agg("main", k, "cags", PMAX)
    bs = [(agg("main", k, v, PMAX), v) for v in BASE[k]]
    b, bv = min(bs, key=lambda x: x[0]["wall"])
    o, oc, _ = oracle(k, PMAX)
    s = seq[k]["wall"]
    summary[k] = dict(cags=c, base=b, bv=bv, orc=o, oc=oc, seq=seq[k])
    lines.append(f"{KNAME[k]} & {s:.3f} & {c['wall']:.3f} ({s/c['wall']:.2f}$\\times$) & {VNAME[bv]}: {b['wall']:.3f} ({s/b['wall']:.2f}$\\times$)"
                 f" & {o['wall']:.3f} & {c['wall']/o['wall']:.2f} & {c['cpu']/seq[k]['cpu']:.2f} / {b['cpu']/seq[k]['cpu']:.2f} & {c['sel']*1e3:.1f} \\\\")
open(os.path.join(TEX, "table_main.tex"), "w").write("\n".join(lines) + "\n")

# --------------------------------------------------------- per-P speedup table
lines = []
for k in KERNELS:
    s = seq[k]["wall"]
    cells = []
    for P in PS:
        c = agg("main", k, "cags", P)
        b = min((agg("main", k, v, P) for v in BASE[k]), key=lambda a: a["wall"])
        o, _, _ = oracle(k, P)
        cells.append(f"{s/c['wall']:.2f} / {s/b['wall']:.2f} / {s/o['wall']:.2f}")
    lines.append(f"{KNAME[k]} & " + " & ".join(cells) + " \\\\")
open(os.path.join(TEX, "table_scaling.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------- one-worker inflation
lines = []
for k in KERNELS:
    s = seq[k]["wall"]
    c = agg("main", k, "cags", 1)
    vb = BASE[k][0] if k != "loop" else "tl_default"
    b = agg("main", k, vb, 1)
    lines.append(f"{KNAME[k]} & {c['wall']/s - 1:+.3f} & {VNAME[vb]}: {b['wall']/s - 1:+.3f} & {c['param']} \\\\")
open(os.path.join(TEX, "table_oneworker.tex"), "w").write("\n".join(lines) + "\n")

# ----------------------------------------------------------- eta sensitivity
etas = sorted({r["eta"] for r in rows if r["experiment"] == "eta"}, key=float)
lines = []
for k in KERNELS:
    s = seq[k]["wall"]
    sp, cut = [], []
    for e in etas:
        a = agg("eta", k, "cags", PMAX, eta=e)
        sp.append(f"{s/a['wall']:.2f}"); cut.append(str(a["param"]))
    lines.append(f"{KNAME[k]} & " + " & ".join(sp) + " \\\\")
    lines.append("\\quad chosen cutoff & " + " & ".join(f"\\textit{{{c}}}" for c in cut) + " \\\\")
open(os.path.join(TEX, "table_eta.tex"), "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------- every-call tasking
lines = []
for k in KERNELS:
    fr = [r for r in rows if r["experiment"] == "fine" and r["kernel"] == k]
    size = fr[0]["size"]
    fine = agg("fine", k, "fine", PMAX, size=size)
    sq = agg("fine", k, "seq", 1, size=size) or seq[k]
    cg = agg("fine", k, "cags", PMAX, size=size) or agg("main", k, "cags", PMAX)
    name = "Fib(34)" if k == "fib" else KNAME[k]
    tasks = f"{fine['tasks']:,}" if fine["tasks"] >= 0 else "$\\approx${:,}".format(int(size))
    lines.append(f"{name} & {sq['wall']:.3f} & {fine['wall']:.3f} & {tasks} & {(lambda v: f'{v:.4f}' if v < 0.01 else f'{v:.3f}')(sq['wall']/fine['wall'])} & {sq['wall']/cg['wall']:.2f} \\\\")
open(os.path.join(TEX, "table_fine.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------- Theorem 9 check: N_t h / W <= (1-eta)/eta
lines = []
for k in ["fib", "msort", "nqueens"]:
    W = seq[k]["cpu"]
    cells = []
    for P in PS:
        # worst repetition, so the check is conservative
        rs = groups[("main", k, "cags", P, "", "0.9", str(SIZES_[k]))]
        cells.append(f"{max(int(r['tasks']) * float(r['h_us']) * 1e-6 for r in rs) / W:.3f}")
    lines.append(f"{KNAME[k]} & " + " & ".join(cells) + " \\\\")
open(os.path.join(TEX, "table_bound.tex"), "w").write("\n".join(lines) + "\n")

# --------------------------------------------- pgfplots: sweep at P max and scaling
XLAB = {"fib": "cutoff $s$ (Fib argument)", "nqueens": "rows left $s$", "msort": "cutoff (elements)", "loop": "grainsize (iterations)"}
LOGX = {"fib": False, "nqueens": False, "msort": True, "loop": True}
fig = []
for k in KERNELS:
    _, _, res = oracle(k, PMAX)
    c = summary[k]["cags"]
    pts = " ".join(f"({cut},{a['wall']:.4f})" for a, cut in res)
    ax = "semilogxaxis" if LOGX[k] else "axis"
    fig.append(f"""\\begin{{tikzpicture}}
\\begin{{{ax}}}[width=0.255\\textwidth,height=3.6cm,title={{\\scriptsize {KNAME[k]}}},xlabel={{\\scriptsize {XLAB[k]}}},
 ylabel={{\\scriptsize time (s)}},tick label style={{font=\\tiny}},grid=major,ymin=0{',log basis x=2' if LOGX[k] else ''}]
\\addplot[black,mark=*,mark size=1pt] coordinates {{{pts}}};
\\addplot[only marks,mark=square*,mark size=2.2pt,red!70!black] coordinates {{({c['param']},{c['wall']:.4f})}};
\\addplot[gray,dashed,domain={res[0][1]}:{res[-1][1]}] {{{summary[k]['seq']['wall']:.4f}}};
\\end{{{ax}}}
\\end{{tikzpicture}}""")
open(os.path.join(TEX, "fig_sweep.tex"), "w").write("\\hfill\n".join(fig) + "\n")

fig = []
for k in KERNELS:
    s = seq[k]["wall"]
    sp = lambda f: " ".join(f"({P},{s/f(P):.3f})" for P in PS)
    cs = sp(lambda P: agg("main", k, "cags", P)["wall"])
    bs = sp(lambda P: min(agg("main", k, v, P)["wall"] for v in BASE[k]))
    os_ = sp(lambda P: oracle(k, P)[0]["wall"])
    fig.append(f"""\\begin{{tikzpicture}}
\\begin{{axis}}[width=0.255\\textwidth,height=3.6cm,title={{\\scriptsize {KNAME[k]}}},xlabel={{\\scriptsize threads $P$}},ylabel={{\\scriptsize speedup}},
 xtick={{{','.join(map(str, PS))}}},tick label style={{font=\\tiny}},grid=major,ymin=0]
\\addplot[red!70!black,thick,mark=square*,mark size=1.2pt] coordinates {{{cs}}};
\\addplot[blue!70!black,mark=triangle*,mark size=1.4pt] coordinates {{{bs}}};
\\addplot[black,dashed,mark=o,mark size=1.2pt] coordinates {{{os_}}};
\\end{{axis}}
\\end{{tikzpicture}}""")
open(os.path.join(TEX, "fig_scaling.tex"), "w").write("\\hfill\n".join(fig) + "\n")

# -------------------------------------------------------------- PNG for README
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figp, axs = plt.subplots(1, 4, figsize=(14, 3.2))
    for ax, k in zip(axs, KERNELS):
        s = seq[k]["wall"]
        ax.plot(PS, [s / agg("main", k, "cags", P)["wall"] for P in PS], "s-", color="firebrick", label="CAGS (ours)")
        ax.plot(PS, [s / min(agg("main", k, v, P)["wall"] for v in BASE[k]) for P in PS], "^-", color="navy", label="best heuristic")
        ax.plot(PS, [s / oracle(k, P)[0]["wall"] for P in PS], "o--", color="black", label="oracle sweep")
        ax.set_title(KNAME[k].replace("$2^{24}$", "2^24").replace("$2^{22}$", "2^22"))
        ax.set_xlabel("threads P"); ax.set_xticks(PS); ax.grid(alpha=.3)
    axs[0].set_ylabel("speedup vs sequential"); axs[0].legend(fontsize=8)
    figp.tight_layout()
    os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
    figp.savefig(os.path.join(OUT, "figures", "speedup.png"), dpi=130)
except ImportError:
    pass

for k in KERNELS:
    S = summary[k]
    print(f"{k:8s} seq={S['seq']['wall']:.3f} cags={S['cags']['wall']:.3f}(cut {S['cags']['param']}, h={S['cags']['h']:.1f}us, sel={S['cags']['sel']*1e3:.1f}ms)"
          f" best_base={S['bv']}:{S['base']['wall']:.3f} oracle={S['orc']['wall']:.3f}(cut {S['oc']}) cags/oracle={S['cags']['wall']/S['orc']['wall']:.2f}"
          f" cpu_infl cags={S['cags']['cpu']/S['seq']['cpu']:.2f} base={S['base']['cpu']/S['seq']['cpu']:.2f}")
