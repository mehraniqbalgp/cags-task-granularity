# Proofs

Complete derivations for every analytical statement in the paper, plus the guarantees of the
Cost-Aware Granularity Selection (CAGS) rule implemented in `src/cagran.c`.
Equation numbers refer to the paper. Section A restates the survey's models; Section B proves the new results.
Numbering in the paper's Appendix: Lemma 8 = Lemma 1, Theorem 9 = Theorem 1, Corollary 10 = Corollary 1,
Proposition 11 = Proposition 1, Proposition 12 = Proposition 2.

Notation: `W` useful sequential work (seconds on one worker), `D` span (longest dependency path),
`P` identical workers, `N` tasks, `g` useful work per task, `h` per-task management cost (CPU seconds),
`eta` target efficiency with `0 < eta < 1`, `mu` scheduler decisions per second.

---

## A. Survey models

### Proposition 1 (work-span lower bound, Eq. 3)
Ignoring overhead, every schedule satisfies `T_P >= max(W/P, D)`.

*Proof.* In time `T_P`, `P` workers deliver at most `P T_P` units of work, and all `W` units must be done,
so `P T_P >= W`. The `D` units on the longest dependency path must run one after another, because each
starts only after its predecessor finishes, so `T_P >= D`. Both hold, hence the maximum. ∎

### Proposition 2 (amortization, Eqs. 4–6)
`N` independent equal tasks, `P | N`, overhead spread evenly, no transfer or contention.
Then `T_P = N(g+h)/P`, `S_P = Pg/(g+h)`, `E_P = g/(g+h)`, and `E_P >= eta  <=>  g >= eta h/(1-eta)`.

*Proof.* Each worker runs `N/P` tasks of cost `g+h`, so `T_P = N(g+h)/P`, while `T_seq = Ng`.
`S_P = T_seq/T_P = Pg/(g+h)` and `E_P = S_P/P = g/(g+h)`.
Since `g+h > 0`: `g/(g+h) >= eta <=> g >= eta g + eta h <=> g(1-eta) >= eta h <=> g >= eta h/(1-eta)`. ∎

Example: `eta = 0.9` gives `g >= 9h`.

### Proposition 3 (scheduler capacity, Eqs. 7–8)
If one service makes every scheduling decision at rate at most `mu`, then `T_P >= N/mu`.
Keeping `P` workers busy in steady state requires `P/(g+h) <= mu`.

*Proof.* All `N` decisions happen before the run ends, and they take at least `N/mu` seconds.
In steady state each busy worker finishes a task every `g+h` seconds, so the workers ask for `P/(g+h)` decisions per second.
The service supplies at most `mu`, so continuous supply requires `P/(g+h) <= mu`. ∎

### Proposition 4 (feasible interval, Eq. 9)
For a slack factor `k >= 1`, a grain `g` with `eta h/(1-eta) <= g <= W/(kP)` exists if and only if
`W >= kP eta h/(1-eta)`.

*Proof.* A closed interval `[a, b]` is nonempty if and only if `a <= b`. Multiply both sides by `kP > 0`. ∎

The upper end follows from requiring `N = W/g >= kP` tasks.

### Proposition 5 (break-even, Eq. 10)
Write the elapsed time of decomposition `x` as `T_x = U + R_x + O_x`, where `U` is useful-work time (the same for both),
`R_x` is delay from imbalance and exposed dependencies, and `O_x` is all other nonuseful time.
Then `T_f < T_c  <=>  R_c - R_f > O_f - O_c`.

*Proof.* `T_f < T_c <=> U + R_f + O_f < U + R_c + O_c <=> R_c - R_f > O_f - O_c`. ∎

### Proposition 6 (CPU ratio, Eq. 11)
Under Proposition 2, `C_P / C_seq = 1 + h/g`.

*Proof.* `C_P = N(g+h)` and `C_seq = Ng`. Divide. ∎

### Proposition 7 (execution waves, Eq. 13)
`N` equal tasks of cost `g+h`, all ready, any greedy scheduler (no worker idles while a task waits).
Then `T_P = ceil(N/P)(g+h)`.

*Proof.* At time 0, `min(N, P)` tasks start. Since all tasks take the same time, they finish together, and the next `min(remaining, P)` start at once.
This repeats in waves of length `g+h`, and the number of waves is the smallest `w` with `wP >= N`, which is `ceil(N/P)`. ∎

`analytical_examples.py` in the paper source reproduces Table III from this formula.

---

## B. Guarantees of Cost-Aware Granularity Selection (CAGS)

### Model assumptions

Implementation note: `src/cagran.c` samples `t(s)` (and the loop's mean iteration cost) *before* measuring `h`, on a warmed-up core. Sampling right after the overhead burst, while idle OpenMP threads still spin on SMT siblings, inflates `t(s)` and can violate the bound in practice (the paper's Table IX discusses this).

- (A1) Each created task costs `h` CPU-seconds of management, charged to some worker. `h` is measured at run time (`calibrate_h`) as `h = P T_empty / N_empty`, which is Proposition 2 with `g = 0`.
- (A2) For the recursive kernels, the sequential cost `t(s)` of a subproblem of size `s` is nondecreasing in `s`, and the sampled `t(s)` is accurate.
- (A3) In the recursive kernels a node either runs its whole subtree sequentially, or spawns **all** of its `b >= 2` children as tasks and waits for them. Fork-join with `taskwait` is fully strict.

### CAGS rule for recursive kernels
`theta = 2 eta h/(1-eta)`. Let `s* = min{ s : t(s) >= theta }`.
A node spawns its children only if every child has size `>= s*`; otherwise it runs sequentially.

### Lemma 8 (task count of a full spawn tree)
If every task that spawns creates at least two tasks, and `L` tasks spawn nothing ("leaf tasks"), then the
total number of tasks, including the root, is `N_t <= 2L - 1`.

*Proof.* The tasks form a rooted tree in which every internal node has at least 2 children.
Count edges two ways: `N_t - 1 = sum of children over internal nodes >= 2 I`, where `I = N_t - L` is the number of internal nodes.
So `N_t - 1 >= 2N_t - 2L`, which gives `N_t <= 2L - 1`. ∎

### Theorem 9 (CPU-efficiency guarantee)
Under (A1)–(A3), CAGS gives total CPU `C <= W/eta`, so the useful fraction of CPU is `W/C >= eta`.

*Proof.* A leaf task is either the root (only when the root is too small to spawn, so `N_t = 1`) or a child created by a spawning parent.
In the second case its size is `>= s*`, so by (A2) it performs at least `t(s*) >= theta` useful work.
Leaf tasks run disjoint subtrees, so `L theta <= W`, which gives `L <= W/theta`.
By Lemma 8, `N_t <= 2L <= 2W/theta`. Total CPU is useful work plus management:
`C = W + N_t h <= W + 2Wh/theta = W + W(1-eta)/eta = W/eta`.
(For `N_t = 1`, `C = W + h`, a single constant cost.) ∎

The factor 2 in `theta` exists only to pay for internal tasks (Lemma 8). With `eta = 0.9`, `theta = 18h`.

### Corollary 10 (elapsed-time bound)
Let `l` be the depth of the task tree. Under a greedy scheduler,
`T_P <= W/(eta P) + D + l h`. Under randomized work stealing the expected time is
`W/(eta P) + O(D + l h)`.

*Proof.* With overhead included, the computation has work `W' = W + N_t h <= W/eta` (Theorem 9) and span
`D' <= D + l h`, since a root-to-leaf path crosses at most `l` task creations.
Graham's (1969) and Brent's (1974) greedy bound gives `T_P <= W'/P + D'`.
For fully strict computations (A3), Blumofe and Leiserson (1999) give expected time `W'/P + O(D')`. ∎

### Proposition 11 (loop rule: CPU bound and runtime queue cap)
Consider a taskloop of `N` iterations with mean iteration cost `gbar`, so `W = N gbar`, and grain `G = ceil(g*/gbar)` iterations, with
`g* >= lo = max(eta h/(1-eta), W/(QP))`. With libgomp's `grainsize` semantics (at most `floor(N/G)` tasks):
1. `C <= W/eta`;
2. `N_t <= QP`, so the runtime does not exceed its cap of `Q P` queued tasks. libgomp uses `Q = 64` (`libgomp/taskloop.c`: if `task_count + num_tasks > 64 * nthreads`, every task runs undeferred on the encountering thread).

*Proof.* `G gbar >= g*`, so `N_t <= N/G <= N gbar/g* = W/g*`.
(1) `C = W + N_t h <= W + W h/g* <= W + W(1-eta)/eta = W/eta`, because `g* >= eta h/(1-eta)`.
(2) `N_t <= W/g* <= W/(W/(QP)) = QP`.
Neither step needs equal iteration costs: only the total `W` and the task count are used. ∎

Note: the cap in (2) counts tasks already queued (`task_count`). The taskloop runs alone inside its own `single`, so `task_count = 0` when it starts.

### Proposition 12 (robustness of the log-midpoint choice)
Given estimated bounds `lo <= hi` with `rho = hi/lo`, CAGS picks `g* = sqrt(lo hi)`. Then:
1. `g*` lies in the true feasible interval `[lo', hi']` whenever `lo' <= lo sqrt(rho)` and `hi' >= hi/sqrt(rho)`, that is, whenever each estimate is off by at most a factor `sqrt(rho)` in the harmful direction;
2. `g*` is the unique maximizer of `min(g/lo, hi/g)`, the worst-case multiplicative margin to the two edges.

*Proof.* (1) `lo sqrt(rho) = lo sqrt(hi/lo) = sqrt(lo hi) = g*`, so `lo' <= g*`. Likewise `hi/sqrt(rho) = g*`, so `hi' >= g*`.
(2) `g/lo` increases and `hi/g` decreases in `g`, so their minimum is largest where they are equal: `g/lo = hi/g`, which gives `g = sqrt(lo hi)`. ∎

### Remark (where the guarantees do not apply)
In N-Queens, subtrees with the same number of rows left can differ by orders of magnitude, because many prefixes are dead ends.
That breaks (A2) for individual nodes, and a node may have a single valid child, which breaks Lemma 8's branching assumption.
CAGS still uses the *mean* sampled cost there, so Theorem 9 is not claimed for N-Queens. Its results are reported as empirical only.
