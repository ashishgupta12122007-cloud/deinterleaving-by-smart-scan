"""Emitter-type identification: cluster-level features -> radar function name (RandomForest)."""
import numpy as np
from deinterleave import deinterleave, merge_clusters, cluster_summary


def feat_vec(s):
    pri = s["pri_us"] if np.isfinite(s["pri_us"]) else 0.0
    spread = (s["f_hi"] - s["f_lo"]) / max(s["f_med"], 1.0)
    return [np.log10(max(s["f_med"], 1.0)), np.log10(max(s["pw_med"], 1e-3)), np.log10(pri + 1.0), np.log10(spread + 1e-4), np.log10(s["n"] + 1)]


def dwell_clusters(D, dwell_list, min_true=10):
    """run the deinterleaver on the listed dwell windows of a scenario and label each cluster with the majority true emitter name"""
    X, y, tx = D["X"], D["y"], D["tx"]
    toa = X[:, 0] * 1e-6
    rows = []
    for (t0, d, c) in dwell_list:
        m = (toa >= t0) & (toa < t0 + d)
        if m.sum() < 60:
            continue
        Xd, yd = X[m], y[m]
        lab = merge_clusters(Xd, deinterleave(Xd, eps=2.2, min_pts=4))
        for s in cluster_summary(Xd, lab):
            if s["n"] < 25:
                continue
            mk = lab == s["cluster"]
            ks, cn = np.unique(yd[mk], return_counts=True)
            k = ks[cn.argmax()]
            purity = cn.max() / mk.sum()
            rows.append((feat_vec(s), tx[int(k)]["function"], purity, int(k)))
    return rows


def pick_dwells(D, sched, n=120, min_pulses=100, seed=0):
    """densest half + random half of the dwell windows that hold >= min_pulses real pulses"""
    X = D["X"]
    toa = X[:, 0] * 1e-6
    L = []
    for (t0, d, c) in sched:
        m = int(((toa >= t0) & (toa < t0 + d)).sum())
        if m >= min_pulses:
            L.append((m, (t0, d, c)))
    L.sort(key=lambda r: -r[0])
    top = [r[1] for r in L[: n // 2]]
    rest = [r[1] for r in L[n // 2:]]
    rng = np.random.default_rng(seed)
    sel = rng.choice(len(rest), min(n // 2, len(rest)), replace=False) if rest else []
    return top + [rest[i] for i in sel]
