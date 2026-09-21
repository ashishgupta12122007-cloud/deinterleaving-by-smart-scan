"""Scoring helpers: expected-detection curves, time-to-detect, wasted dwell."""
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz

GRID = np.round(np.arange(0.0, 30.0001, 0.1), 3)


def cum_after_dwells(tw, sched):
    cum = np.zeros(len(tw.em))
    ends, cums = [], []
    for (t0, d, c) in sched:
        cum = cum + tw.expected_pulses(t0, t0 + d, c)
        ends.append(t0 + d)
        cums.append(cum.copy())
    return np.array(ends), np.array(cums)


def p_on_grid(tw, sched, calib, K, grid=GRID):
    """(G,E) probability that each emitter has >= K pulses in the REAL data after time t (calibrated, monotone)"""
    ends, cums = cum_after_dwells(tw, sched)
    p = np.maximum.accumulate(calib[K](cums), axis=0)
    idx = np.searchsorted(ends, grid, side="right") - 1
    out = np.zeros((len(grid), cums.shape[1]))
    ok = idx >= 0
    out[ok] = p[idx[ok]]
    return out


def summarize_curve(curve, grid=GRID):
    final = float(curve[-1])
    if final <= 0:
        return dict(final=0.0, t50=np.nan, t80=np.nan, auc=np.nan)
    return dict(final=final,
                t50=float(grid[np.searchsorted(curve, 0.5 * final)]),
                t80=float(grid[np.searchsorted(curve, 0.8 * final)]),
                auc=float(_trapz(curve, grid) / (grid[-1] * final)))


def detect_times(p, grid=GRID, thr=0.5):
    """first time each emitter's calibrated probability crosses thr (None if never)"""
    hit = p >= thr
    first = hit.argmax(0)
    ok = hit.any(0)
    return [float(grid[f]) if o else None for f, o in zip(first, ok)]


def wasted_fraction(tw, sched, reach_max=18500.0):
    """share of dwell time spent on windows that contain no reachable emitter frequency at all"""
    fr = []
    for e in tw.em:
        if e.mean_freq <= reach_max:
            fr.extend(list(np.atleast_1d(e.freqs)))
    fr = np.array(fr)
    tot = w = 0.0
    for (t0, d, c) in sched:
        tot += d
        if not np.any(np.abs(fr - c) <= tw.B):
            w += d
    return w / tot


def real_detection_times(D, ids, Ks=(20, 100)):
    """time (s) at which the K-th real pulse of every emitter arrived (None if never)"""
    X, y = D["X"], D["y"]
    out = {K: [] for K in Ks}
    counts = []
    for k in ids:
        t = np.sort(X[y == k, 0]) * 1e-6
        counts.append(int(len(t)))
        for K in Ks:
            out[K].append(float(t[K - 1]) if len(t) >= K else None)
    return out, counts
