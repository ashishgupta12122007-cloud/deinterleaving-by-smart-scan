"""
Learned scan scheduler ("SmartScan-ML").

1. LEARN   (offline)  For every emitter instance in the training scenarios the digital twin gives the expected
           recorded-pulse rate over time.  Monte-Carlo placement of random dwell chunks turns that into a
           detection curve  D_j(tau)  = P(>= K pulses | tau seconds of in-band dwell time)  for each library
           emitter j.  The library of curves is the learned prior over "what is out there and how hard is it
           to intercept".
2. PLAN    (offline)  Choose how many seconds tau_c to spend at every candidate centre c by greedy
           marginal-gain maximisation of  sum_j D_j( sum_c ff_j(c) * tau_c ).
3. SEQUENCE           Stride-schedule the chunks so every centre is revisited evenly (low latency) with jittered
           timing (no resonance with beam rotation periods).
4. ADAPT   (online)   Empirical-Bayes update: live detections per frequency bin re-weight the prior curves,
           so empty / already-solved bands are abandoned and productive ones are extended.
"""
import numpy as np
from scipy.stats import poisson
from sim import Twin, Emitter

from sklearn.isotonic import IsotonicRegression

K_DET = 20
K_CHAR = 100


class Calib:
    """maps the twin's expected pulse count -> probability that the REAL data holds >= K pulses for that emitter.
    Isotonic regression on (log10(expected+1)) fitted on the emitters of the training files; removes the twin's optimism."""

    def __init__(self, pred, act, K):
        self.K = K
        x = np.log10(np.asarray(pred, float) + 1.0)
        yv = (np.asarray(act) >= K).astype(float)
        self.iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip").fit(x, yv)

    def __call__(self, lam):
        lam = np.asarray(lam, float)
        return self.iso.predict(np.log10(lam + 1.0).ravel()).reshape(lam.shape)


def fit_calibrations(twins, Ds, sched_fn, ks=(K_DET, K_CHAR)):
    P, A = [], []
    for tw, D in zip(twins, Ds):
        pred, _ = tw.score_schedule(sched_fn(D))
        act = np.bincount(D["y"], minlength=max(tw.ids) + 1)[tw.ids]
        P.append(pred)
        A.append(act)
    P, A = np.concatenate(P), np.concatenate(A)
    return {K: Calib(P, A, K) for K in ks}
TAU_GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2, 1.6, 2.4, 3.2, 4.8, 6.4, 9.6, 12.8, 19.2, 30.0])


class Library:
    """learned prior: per-emitter detection curves for the union of the training scenarios"""

    def __init__(self, twins, chunk=0.1, R=160, seed=0, freq_jitter_mhz=0.0, k_list=(K_DET, K_CHAR), calib=None):
        rng = np.random.default_rng(seed)
        self.chunk = chunk
        self.em = []          # Emitter objects
        self.scen = []
        curves = {K: [] for K in k_list}
        self.k_list = k_list
        for si, tw in enumerate(twins):
            T = tw.T
            nsl = tw.cum.shape[1] - 1
            L = int(round(chunk / tw.dt))
            for j, e in enumerate(tw.em):
                self.em.append(e)
                self.scen.append(si)
                cj = tw.cum[j]
                row = {K: [] for K in k_list}
                for tau in TAU_GRID:
                    m = int(round(tau / chunk))
                    if m == 0:
                        for K in k_list:
                            row[K].append(0.0)
                        continue
                    starts = rng.integers(0, nsl - L, size=(R, m))
                    N = (cj[starts + L] - cj[starts]).sum(1)
                    for K in k_list:
                        if calib is None:
                            row[K].append(float(poisson.sf(K - 1, N).mean()))
                        else:
                            row[K].append(float(calib[K](N).mean()))
                for K in k_list:
                    curves[K].append(row[K])
        self.D = {K: np.maximum.accumulate(np.array(curves[K]), axis=1) for K in k_list}
        self.n_scen = len(twins)
        self.B = twins[0].B
        self.freq_jitter = freq_jitter_mhz
        self.mean_freq = np.array([e.mean_freq for e in self.em])
        self.rng = rng

    # ---------------------------------------------------- utilities
    def ff_matrix(self, centers, jitter=None):
        """(J, C) count factor of every library emitter for every candidate centre"""
        jit = self.freq_jitter if jitter is None else jitter
        F = np.zeros((len(self.em), len(centers)))
        for j, e in enumerate(self.em):
            if jit > 0:
                sh = self.rng.normal(0, jit)
                f_save = e.freqs.copy()
                e.freqs = e.freqs + sh
                F[j] = e.freq_factor(centers, self.B)
                e.freqs = f_save
            else:
                F[j] = e.freq_factor(centers, self.B)
        return F

    def curve_eval(self, K, tau_eff):
        """tau_eff: (J, ...) effective in-band dwell seconds. returns detection probability with same shape"""
        D = self.D[K]
        x = np.clip(tau_eff, 0, TAU_GRID[-1])
        idx = np.clip(np.searchsorted(TAU_GRID, x, side="right") - 1, 0, len(TAU_GRID) - 2)
        x0, x1 = TAU_GRID[idx], TAU_GRID[idx + 1]
        w = (x - x0) / (x1 - x0)
        J = D.shape[0]
        sh = [J] + [1] * (x.ndim - 1)
        rows = np.arange(J).reshape(sh)
        return D[rows, idx] * (1 - w) + D[rows, idx + 1] * w


def plan_allocation(lib, centers, T=30.0, step=0.1, weights=None, kw=None, teff0=None, ff=None):
    """greedy marginal-gain allocation. weights: per-emitter weights (default 1); kw: {K: weight};
    teff0: effective in-band seconds already spent (for re-planning); ff: cached count-factor matrix"""
    kw = kw or {K_DET: 1.0, K_CHAR: 0.5}
    Jn = len(lib.em)
    w = np.ones(Jn) if weights is None else weights
    if ff is None:
        ff = lib.ff_matrix(centers)                   # (J, C)
    tau = np.zeros(len(centers))
    teff = np.zeros(Jn) if teff0 is None else teff0.copy()
    nsteps = int(round(T / step))
    for _ in range(nsteps):
        cand = teff[:, None] + ff * step              # (J, C)
        gain = np.zeros(len(centers))
        for K, kwt in kw.items():
            g = lib.curve_eval(K, cand) - lib.curve_eval(K, teff)[:, None]
            gain += kwt * (w[:, None] * g).sum(0)
        c = int(np.argmax(gain))
        tau[c] += step
        teff += ff[:, c] * step
    return tau, ff


def stride_sequence(centers, tau, step=0.1, seed=0, T=30.0):
    """even, jittered interleaving of chunks -> list of (t0, dur, centre)"""
    rng = np.random.default_rng(seed)
    items = []
    for c, tc in zip(centers, tau):
        n = int(round(tc / step))
        if n <= 0:
            continue
        for k in range(n):
            u = (k + 0.5 + rng.uniform(-0.35, 0.35)) / n
            items.append((u, c))
    items.sort()
    out, t = [], 0.0
    for _, c in items:
        d = min(step, T - t)
        if d <= 1e-9:
            break
        out.append((t, d, float(c)))
        t += d
    return out
