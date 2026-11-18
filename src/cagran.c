/*
 * cagran.c - Cost-Aware task GRANularity selection (CAGS) for OpenMP tasks.
 *
 * Kernels : fib, nqueens, msort (recursive tasks), loop (irregular taskloop)
 * Variants: seq    serial elision, no OpenMP constructs inside the kernel
 *           fine   a task at every recursive call / grainsize(1)
 *           depth  fixed depth cutoff D = ceil(log2(8P))   (recursive kernels)
 *           fixed  size cutoff given on the command line   (used for the oracle sweep)
 *           cags   proposed: measure h, sample t(s), pick the cutoff from the model
 *           static, dynamic, guided, tl_default            (loop kernel baselines)
 *
 * Output: one CSV row per repetition on stdout
 *   kernel,variant,P,rep,wall_s,cpu_s,tasks,param,h_us,theta_us,select_s,ok
 *
 * The CAGS rule (see proofs/PROOFS.md):
 *   recursive kernels : theta = 2*eta/(1-eta) * h ; s* = min{ s : t(s) >= theta }
 *                       children are spawned only if every child has size >= s*.
 *   loop kernel       : lo = max(eta/(1-eta) * h, W/(Q P)) ; hi = W/(k P) ;
 *                       g* = sqrt(lo*hi) if lo <= hi else hi ; grainsize = ceil(g* / gbar).
 * The time to measure h and sample t(s) is included in wall_s and cpu_s.
 */
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#define MAXT 512
#define MAXN 24

static double ETA = 0.9; /* target efficiency, overridable with CAGS_ETA */
static double KSLACK = 8; /* tasks per worker for the upper bound, CAGS_K   */
static double QCAP = 64;  /* runtime queue cap per worker (libgomp: 64), CAGS_QCAP */

/* ------------------------------------------------------------------ utils */
static struct { long n; char pad[56]; } tcount[MAXT];
static inline void count_task(void) { tcount[omp_get_thread_num()].n++; }
static long tasks_total(void) { long s = 0; for (int i = 0; i < MAXT; i++) s += tcount[i].n; return s; }

static double cpu_now(void) {
    struct rusage r;
    getrusage(RUSAGE_SELF, &r);
    return r.ru_utime.tv_sec + 1e-6 * r.ru_utime.tv_usec + r.ru_stime.tv_sec + 1e-6 * r.ru_stime.tv_usec;
}

static int cmpd(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static uint64_t rng = 88172645463325252ULL;
static inline uint64_t xorshift(void) { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng; }

/* ---------------------------------------------------- overhead calibration */
static void empty_tree(int d) {
    if (d == 0) return;
#pragma omp task
    { count_task(); empty_tree(d - 1); }
#pragma omp task
    { count_task(); empty_tree(d - 1); }
#pragma omp taskwait
}

/* h = P * T / N: CPU service per task on P workers, as in T_P = N(g+h)/P with g = 0. */
static double calibrate_h(int P) {
    enum { DEPTH = 10, REPS = 3 };
    const double ntasks = (double)((1L << (DEPTH + 1)) - 2);
    double t[REPS];
    for (int r = 0; r < REPS; r++) {
        double el = 0;
#pragma omp parallel num_threads(P)
#pragma omp single
        {
            double t0 = omp_get_wtime();
            empty_tree(DEPTH);
            el = omp_get_wtime() - t0;
        }
        t[r] = P * el / ntasks;
    }
    qsort(t, REPS, sizeof t[0], cmpd);
    return t[REPS / 2];
}

/* Mean seconds of one call of f(ctx, s); repeats until >= 50 us were timed. */
typedef void (*seqfn)(void *ctx, int s);
static double time_call(seqfn f, void *ctx, int s) {
    long calls = 0;
    double t0 = omp_get_wtime(), el;
    do { f(ctx, s); calls++; el = omp_get_wtime() - t0; } while (el < 50e-6);
    return el / calls;
}

/* Sampled cost profile t(s) for s = lo, lo+1 (or 2lo), ... until t(s) >= TMAX or s >= hi.
   Taken before h is measured, so idle OpenMP threads are not spinning on SMT siblings
   and inflating the samples. */
#define TMAX 1e-3
typedef struct { int n, s[64]; double t[64]; } profile_t;
static void profile(seqfn f, void *ctx, int lo, int hi, int doubling, profile_t *p) {
    p->n = 0;
    for (int s = lo; s < hi && p->n < 64; s = doubling ? 2 * s : s + 1) {
        p->s[p->n] = s;
        p->t[p->n] = time_call(f, ctx, s);
        if (p->t[p->n++] >= TMAX) break;
    }
}
/* s* = smallest sampled s with t(s) >= theta. If the whole problem is cheaper than theta, return hi
   (run sequentially); if theta exceeds TMAX, fall back to the largest sampled size. */
static int pick_cutoff(const profile_t *p, int hi, double theta) {
    for (int i = 0; i < p->n; i++)
        if (p->t[i] >= theta) return p->s[i];
    return p->n && p->t[p->n - 1] >= TMAX ? p->s[p->n - 1] : hi;
}

/* --------------------------------------------------------------- fib */
static volatile long sink;
static long fib_seq(int n) { return n < 2 ? n : fib_seq(n - 1) + fib_seq(n - 2); }
static void fib_sample(void *ctx, int s) { (void)ctx; sink = fib_seq(s); }

/* mode 0: spawn iff both children have size >= cut; mode 1: spawn iff depth < cut */
static long fib_par(int n, int d, int cut, int mode) {
    if (n < 2) return n;
    int spawn = mode ? d < cut : n - 2 >= cut;
    if (!spawn) return fib_seq(n);
    long x, y;
#pragma omp task shared(x)
    { count_task(); x = fib_par(n - 1, d + 1, cut, mode); }
#pragma omp task shared(y)
    { count_task(); y = fib_par(n - 2, d + 1, cut, mode); }
#pragma omp taskwait
    return x + y;
}

/* ------------------------------------------------------------ nqueens */
typedef struct { signed char a[MAXN]; } board_t;
static int nq_ok(const signed char *a, int r, int c) {
    for (int i = 0; i < r; i++) {
        int p = a[i];
        if (p == c || p - c == r - i || c - p == r - i) return 0;
    }
    return 1;
}
static long nq_seq(int n, int r, signed char *a) {
    if (r == n) return 1;
    long s = 0;
    for (int c = 0; c < n; c++)
        if (nq_ok(a, r, c)) { a[r] = (signed char)c; s += nq_seq(n, r + 1, a); }
    return s;
}
/* "size" of a node = rows still to place. Spawn children iff they keep >= cut rows. */
static long nq_par(int n, int r, board_t b, int cut, int mode) {
    if (r == n) return 1;
    int spawn = mode ? r < cut : n - (r + 1) >= cut;
    if (!spawn) return nq_seq(n, r, b.a);
    long cnt[MAXN] = {0};
    for (int c = 0; c < n; c++) {
        if (!nq_ok(b.a, r, c)) continue;
        board_t nb = b;
        nb.a[r] = (signed char)c;
#pragma omp task firstprivate(nb, c) shared(cnt)
        { count_task(); cnt[c] = nq_par(n, r + 1, nb, cut, mode); }
    }
#pragma omp taskwait
    long s = 0;
    for (int c = 0; c < n; c++) s += cnt[c];
    return s;
}
/* Sample mean subtree time with s rows left, over random valid prefixes. */
static int NQ_N;
static void nq_sample(void *ctx, int s) {
    (void)ctx;
    board_t b;
    for (int tries = 0; tries < 64; tries++) {
        int r = 0, depth = NQ_N - s;
        for (; r < depth; r++) {
            int c0 = (int)(xorshift() % NQ_N), c;
            for (c = 0; c < NQ_N; c++) if (nq_ok(b.a, r, (c0 + c) % NQ_N)) break;
            if (c == NQ_N) break;
            b.a[r] = (signed char)((c0 + c) % NQ_N);
        }
        if (r == depth) { sink = nq_seq(NQ_N, r, b.a); return; }
    }
}

/* ------------------------------------------------------------ mergesort */
#define MS_BASE 32
static void insertion(int *a, long n) {
    for (long i = 1; i < n; i++) {
        int v = a[i];
        long j = i - 1;
        while (j >= 0 && a[j] > v) { a[j + 1] = a[j]; j--; }
        a[j + 1] = v;
    }
}
static void merge(int *a, int *tmp, long n, long mid) {
    long i = 0, j = mid, k = 0;
    while (i < mid && j < n) tmp[k++] = a[i] <= a[j] ? a[i++] : a[j++];
    while (i < mid) tmp[k++] = a[i++];
    while (j < n) tmp[k++] = a[j++];
    memcpy(a, tmp, n * sizeof *a);
}
static void ms_seq(int *a, int *tmp, long n) {
    if (n <= MS_BASE) { insertion(a, n); return; }
    ms_seq(a, tmp, n / 2);
    ms_seq(a + n / 2, tmp + n / 2, n - n / 2);
    merge(a, tmp, n, n / 2);
}
static void ms_par(int *a, int *tmp, long n, int d, long cut, int mode) {
    if (n <= MS_BASE) { insertion(a, n); return; }
    int spawn = mode ? d < cut : n / 2 >= cut;
    if (!spawn) { ms_seq(a, tmp, n); return; }
#pragma omp task
    { count_task(); ms_par(a, tmp, n / 2, d + 1, cut, mode); }
#pragma omp task
    { count_task(); ms_par(a + n / 2, tmp + n / 2, n - n / 2, d + 1, cut, mode); }
#pragma omp taskwait
    merge(a, tmp, n, n / 2);
}
static int *MS_SRC;
static void ms_sample(void *ctx, int s) {
    int *buf = ctx;
    memcpy(buf, MS_SRC, (size_t)s * sizeof *buf);
    ms_seq(buf, buf + s, s);
}

/* ------------------------------------------------- irregular loop kernel */
/* Iteration i costs ~ (1 + 255 i / N) dependent multiply-adds: a triangular profile. */
static long LP_N;
static inline double body(long i) {
    long reps = 1 + (255 * i) / LP_N;
    double x = (double)(i & 1023);
    for (long k = 0; k < reps; k++) x = x * 1.0000001 + 1e-7;
    return x;
}
static double loop_check(const double *o) { double s = 0; for (long i = 0; i < LP_N; i++) s += o[i]; return s; }

/* ---------------------------------------------------------------- driver */
typedef struct { double h, theta, select_s; long param; } choice_t;

static choice_t cags_recursive(int P, seqfn f, void *ctx, int lo, int hi, int doubling) {
    choice_t c;
    double t0 = omp_get_wtime();
    profile_t prof;
    profile(f, ctx, lo, hi, doubling, &prof);
    c.h = calibrate_h(P);
    c.theta = 2 * ETA / (1 - ETA) * c.h;
    c.param = pick_cutoff(&prof, hi, c.theta);
    c.select_s = omp_get_wtime() - t0;
    memset(tcount, 0, sizeof tcount); /* calibration tasks are not kernel tasks */
    return c;
}

int main(int argc, char **argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s kernel variant P reps size [param]\n"
                        "  kernel: fib nqueens msort loop\n", argv[0]);
        return 2;
    }
    const char *kernel = argv[1], *variant = argv[2];
    int P = atoi(argv[3]), reps = atoi(argv[4]);
    long size = atol(argv[5]);
    long param = argc > 6 ? atol(argv[6]) : -1;
    if (getenv("CAGS_ETA")) ETA = atof(getenv("CAGS_ETA"));
    if (getenv("CAGS_K")) KSLACK = atof(getenv("CAGS_K"));
    if (getenv("CAGS_QCAP")) QCAP = atof(getenv("CAGS_QCAP"));
    if (P < 1 || P > MAXT || reps < 1 || ETA <= 0 || ETA >= 1) { fprintf(stderr, "bad arguments\n"); return 2; }
    omp_set_dynamic(0);
    int is = !strcmp(variant, "seq"), isfine = !strcmp(variant, "fine"), isdepth = !strcmp(variant, "depth"),
        isfixed = !strcmp(variant, "fixed"), iscags = !strcmp(variant, "cags");
    int depthD = (int)ceil(log2(8.0 * P));

    /* inputs */
    int *src = NULL, *arr = NULL, *tmp = NULL, *sbuf = NULL;
    double *out = NULL, ref = 0;
    long expect = 0;
    if (!strcmp(kernel, "msort")) {
        src = malloc(size * sizeof *src); arr = malloc(size * sizeof *arr); tmp = malloc(size * sizeof *tmp);
        sbuf = malloc(2 * (size_t)size * sizeof *sbuf);
        if (!src || !arr || !tmp || !sbuf) { fprintf(stderr, "out of memory\n"); return 1; }
        for (long i = 0; i < size; i++) src[i] = (int)(xorshift() & 0x7fffffff);
        MS_SRC = src;
    } else if (!strcmp(kernel, "loop")) {
        LP_N = size;
        out = malloc(size * sizeof *out);
        if (!out) { fprintf(stderr, "out of memory\n"); return 1; }
        for (long i = 0; i < size; i++) out[i] = body(i);
        ref = loop_check(out);
    } else if (!strcmp(kernel, "fib")) {
        long a = 0, b = 1;
        for (long i = 0; i < size; i++) { long t = a + b; a = b; b = t; }
        expect = a;
    } else if (!strcmp(kernel, "nqueens")) {
        static const long sol[] = {1, 1, 0, 0, 2, 10, 4, 40, 92, 352, 724, 2680, 14200, 73712, 365596, 2279184, 14772512};
        if (size < 1 || size > 16) { fprintf(stderr, "nqueens size must be 1..16\n"); return 2; }
        NQ_N = (int)size;
        expect = sol[size];
    } else { fprintf(stderr, "unknown kernel\n"); return 2; }

    for (int rep = 0; rep < reps; rep++) {
        if (arr) memcpy(arr, src, size * sizeof *arr);
        if (out) memset(out, 0, size * sizeof *out);
        usleep(100000); /* let idle OpenMP threads stop spinning before the CPU clock is read */
        /* 30 ms busy warm-up (untimed, every variant): the sleep lets a DVFS governor lower the
           clock, which would otherwise inflate the first samples of the timed region */
        for (double tw = omp_get_wtime(); omp_get_wtime() - tw < 0.03;) sink++;
        memset(tcount, 0, sizeof tcount);
        choice_t ch = {0, 0, 0, param};
        long res = 0;
        double c0 = cpu_now(), w0 = omp_get_wtime();

        if (!strcmp(kernel, "fib")) {
            int mode = isdepth;
            if (iscags) ch = cags_recursive(P, fib_sample, NULL, 2, (int)size, 0);
            if (isfine) ch.param = 0;
            if (isdepth) ch.param = depthD;
            if (is) res = fib_seq((int)size);
            else {
#pragma omp parallel num_threads(P)
#pragma omp single
                res = fib_par((int)size, 0, (int)ch.param, mode);
            }
        } else if (!strcmp(kernel, "nqueens")) {
            int mode = isdepth;
            board_t b;
            memset(&b, 0, sizeof b);
            if (iscags) ch = cags_recursive(P, nq_sample, NULL, 1, NQ_N, 0);
            if (isfine) ch.param = 0;
            if (isdepth) ch.param = depthD;
            if (is) res = nq_seq(NQ_N, 0, b.a);
            else {
#pragma omp parallel num_threads(P)
#pragma omp single
                res = nq_par(NQ_N, 0, b, (int)ch.param, mode);
            }
        } else if (!strcmp(kernel, "msort")) {
            int mode = isdepth;
            if (iscags) ch = cags_recursive(P, ms_sample, sbuf, 64, (int)(size / 2), 1);
            if (isfine) ch.param = MS_BASE + 1;
            if (isdepth) ch.param = depthD;
            if (is) ms_seq(arr, tmp, size);
            else {
#pragma omp parallel num_threads(P)
#pragma omp single
                ms_par(arr, tmp, size, 0, ch.param, mode);
            }
        } else { /* loop */
            long N = size;
            if (iscags) {
                double t0 = omp_get_wtime();
                /* mean iteration cost from 4096 evenly spaced iterations (sampled before h, see profile()) */
                long stride = N / 4096 > 0 ? N / 4096 : 1, cnt = 0;
                double s0 = omp_get_wtime(), acc = 0;
                for (long i = 0; i < N; i += stride, cnt++) acc += body(i);
                sink = (long)acc;
                double gbar = (omp_get_wtime() - s0) / cnt, W = gbar * N;
                ch.h = calibrate_h(P);
                /* lo: amortize h (Eq. threshold) and stay under the runtime's task-queue cap */
                double lo = fmax(ETA / (1 - ETA) * ch.h, W / (QCAP * P)), hi = W / (KSLACK * P);
                /* log-midpoint of the feasible interval: tolerates a factor sqrt(hi/lo)
                   misestimate of h or W in either direction (Proposition 12, proofs/PROOFS.md) */
                ch.theta = lo <= hi ? sqrt(lo * hi) : hi;
                ch.param = (long)ceil(ch.theta / gbar);
                if (ch.param < 1) ch.param = 1;
                ch.select_s = omp_get_wtime() - t0;
                memset(tcount, 0, sizeof tcount);
            }
            if (isfine) ch.param = 1;
            if (is) for (long i = 0; i < N; i++) out[i] = body(i);
            else if (!strcmp(variant, "static")) {
#pragma omp parallel for schedule(static) num_threads(P)
                for (long i = 0; i < N; i++) out[i] = body(i);
            } else if (!strcmp(variant, "dynamic")) {
#pragma omp parallel for schedule(dynamic, 1) num_threads(P)
                for (long i = 0; i < N; i++) out[i] = body(i);
            } else if (!strcmp(variant, "guided")) {
#pragma omp parallel for schedule(guided) num_threads(P)
                for (long i = 0; i < N; i++) out[i] = body(i);
            } else if (!strcmp(variant, "tl_default")) {
#pragma omp parallel num_threads(P)
#pragma omp single
#pragma omp taskloop
                for (long i = 0; i < N; i++) out[i] = body(i);
            } else if (isfine || isfixed || iscags) {
                long g = ch.param;
#pragma omp parallel num_threads(P)
#pragma omp single
#pragma omp taskloop grainsize(g)
                for (long i = 0; i < N; i++) out[i] = body(i);
                ch.param = g;
            } else { fprintf(stderr, "unknown variant for loop\n"); return 2; }
        }

        double wall = omp_get_wtime() - w0, cpu = cpu_now() - c0;
        int ok;
        if (!strcmp(kernel, "msort")) {
            ok = 1;
            for (long i = 1; i < size; i++) if (arr[i - 1] > arr[i]) { ok = 0; break; }
        } else if (!strcmp(kernel, "loop")) ok = fabs(loop_check(out) - ref) <= 1e-9 * fabs(ref);
        else ok = res == expect;
        long ntask = tasks_total();
        if (!strcmp(kernel, "loop") && !is && strcmp(variant, "static") && strcmp(variant, "dynamic") && strcmp(variant, "guided"))
            ntask = -1; /* taskloop bodies are not instrumented; count is implementation-defined */
        printf("%s,%s,%d,%d,%.6f,%.6f,%ld,%ld,%.4f,%.4f,%.6f,%d\n", kernel, variant, P, rep, wall, cpu, ntask,
               ch.param, ch.h * 1e6, ch.theta * 1e6, ch.select_s, ok);
        fflush(stdout);
        if (!ok) { fprintf(stderr, "CORRECTNESS FAILURE\n"); return 1; }
    }
    return 0;
}
