"""ML deinterleaver: split the superimposed pulse stream of one dwell into per-emitter pulse trains.
Features: AoA (as unit-circle coords), log10(pulse width) and, for narrow-band emitters, frequency.
Two-stage: DBSCAN in (AoA, PW) then split each cluster on frequency modes when the cluster is multi-carrier."""
import numpy as np
from sklearn.cluster import DBSCAN

AOA_SIG = 0.55      # deg, measured AoA noise
PW_SIG = 0.03       # relative (log10) PW spread tolerance


def features(X):
    aoa = np.radians(X[:, 3])
    r = 180.0 / np.pi / AOA_SIG
    return np.column_stack([np.sin(aoa) * r, np.cos(aoa) * r, np.log10(np.maximum(X[:, 2], 1e-3)) / PW_SIG])


def deinterleave(X, eps=2.2, min_pts=6, max_pts=12000, seed=0):
    """X: (n,5) [ToA,f,PW,AoA,Amp]. returns integer labels (-1 = unassigned)"""
    n = len(X)
    if n == 0:
        return np.zeros(0, int)
    idx = np.arange(n)
    if n > max_pts:
        idx = np.random.default_rng(seed).choice(n, max_pts, replace=False)
    F = features(X[idx])
    lab_s = DBSCAN(eps=eps, min_samples=min_pts).fit_predict(F)
    if len(idx) == n:
        return lab_s
    # assign remaining pulses to nearest sampled cluster member
    from sklearn.neighbors import KNeighborsClassifier
    ok = lab_s >= 0
    if ok.sum() < 5:
        return np.full(n, -1)
    knn = KNeighborsClassifier(1).fit(F[ok], lab_s[ok])
    Fa = features(X)
    lab = knn.predict(Fa)
    d, _ = knn.kneighbors(Fa, 1)
    lab[d[:, 0] > eps * 1.5] = -1
    return lab


def cluster_summary(X, lab):
    out = []
    for c in np.unique(lab[lab >= 0]):
        m = lab == c
        t = np.sort(X[m, 0])
        dt = np.diff(t)
        dt = dt[(dt > 0.5) & (dt < 20000)]
        pri = np.nan
        if len(dt) > 5:
            h, e = np.histogram(dt, bins=np.arange(0.5, min(dt.max(), 3000) + 0.5, 0.5))
            if h.sum() > 0:
                # smallest strong peak
                thr = max(3, 0.15 * h.max())
                pk = np.where(h >= thr)[0]
                pri = 0.5 * (e[pk[0]] + e[pk[0] + 1]) if len(pk) else np.nan
        f = X[m, 1]
        out.append(dict(cluster=int(c), n=int(m.sum()), f_med=float(np.median(f)), f_lo=float(np.percentile(f, 2)), f_hi=float(np.percentile(f, 98)),
                        pw_med=float(np.median(X[m, 2])), aoa=float(np.median(X[m, 3])), amp_max=float(X[m, 4].max()), pri_us=float(pri)))
    return out


def score(lab, truth, min_true=10):
    """ARI + emitter recovery: a true emitter is recovered when one cluster holds >=80% of its pulses and it makes up >=90% of that cluster"""
    from sklearn.metrics import adjusted_rand_score
    ok = truth >= 0
    ari = adjusted_rand_score(truth[ok], np.where(lab[ok] < 0, -1 - truth[ok], lab[ok]))
    rec, tot = 0, 0
    for k in np.unique(truth):
        mk = truth == k
        if mk.sum() < min_true:
            continue
        tot += 1
        labs, cnts = np.unique(lab[mk & (lab >= 0)], return_counts=True)
        if len(labs) == 0:
            continue
        c = labs[cnts.argmax()]
        comp = cnts.max() / mk.sum()
        pur = cnts.max() / (lab == c).sum()
        if comp >= 0.8 and pur >= 0.9:
            rec += 1
    return ari, rec, tot


def merge_clusters(X, lab, aoa_tol=1.2, f_tol=6.0):
    """merge over-segmented clusters (e.g. PW sequences) that share bearing and carrier-frequency support"""
    ids = [c for c in np.unique(lab) if c >= 0]
    if len(ids) < 2:
        return lab
    info = {}
    for c in ids:
        m = lab == c
        f = X[m, 1]
        a = np.radians(X[m, 3])
        info[c] = dict(aoa=np.degrees(np.arctan2(np.sin(a).mean(), np.cos(a).mean())),
                       flo=np.percentile(f, 3), fhi=np.percentile(f, 97), n=m.sum())
    parent = {c: c for c in ids}

    def find(c):
        while parent[c] != c:
            parent[c] = parent[parent[c]]
            c = parent[c]
        return c
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            ia, ib = info[a], info[b]
            da = abs((ia["aoa"] - ib["aoa"] + 180) % 360 - 180)
            if da > aoa_tol:
                continue
            ov = min(ia["fhi"], ib["fhi"]) - max(ia["flo"], ib["flo"]) + f_tol
            span = max(min(ia["fhi"] - ia["flo"], ib["fhi"] - ib["flo"]), f_tol)
            if ov >= 0.5 * span:
                parent[find(a)] = find(b)
    out = lab.copy()
    for c in ids:
        out[lab == c] = find(c)
    return out
