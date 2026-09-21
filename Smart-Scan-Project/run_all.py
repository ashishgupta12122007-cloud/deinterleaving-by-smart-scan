#!/usr/bin/env python3
"""
SIH 26055 - Smart Scan strategy for Electronic Warfare: end-to-end pipeline.

    python run_all.py plan   --data DIR --out OUT     # twin validation + leave-one-out ML schedules  (~5 min)
    python run_all.py signal --data DIR --out OUT     # deinterleaving + type identification + lab snapshots (~2 min)
    python run_all.py export --data DIR --out OUT     # writes prototype_data.json, CSV schedules, results.md
    python run_all.py deploy --data DIR --out OUT     # schedule for a NEW scenario, trained on every provided file

DIR must contain config_*.h5 files. Nothing here needs h5py.
"""
import argparse, glob, json, os, pickle, sys, time
import numpy as np
from scipy.stats import spearmanr

from data_io import load_config
from sim import Twin, baseline_schedule
from scheduler import Library, plan_allocation, stride_sequence, fit_calibrations, K_DET, K_CHAR
from evaluate import GRID, p_on_grid, summarize_curve, detect_times, wasted_fraction, real_detection_times
from deinterleave import deinterleave, merge_clusters, cluster_summary, score
from identify import dwell_clusters, pick_dwells, feat_vec

CENTERS = np.arange(250, 18251, 250.0)


def name_of(path):
    return os.path.basename(path).replace("config_", "").replace(".h5", "")


def load_all(data):
    files = sorted(glob.glob(os.path.join(data, "config_*.h5")))
    Ds = {name_of(f): load_config(f) for f in files}
    Tw = {k: Twin(D) for k, D in Ds.items()}
    return Ds, Tw


def save_csv(path, sched):
    with open(path, "w") as f:
        f.write("t_start_s,dwell_s,center_mhz\n")
        for t0, d, c in sched:
            f.write(f"{t0:.3f},{d:.3f},{c:.0f}\n")


# ----------------------------------------------------------------------------------------------- plan
def cmd_plan(a):
    Ds, Tw = load_all(a.data)
    names = list(Ds)
    os.makedirs(a.out, exist_ok=True)
    res = {"validation": {}, "scen": {}}
    # 1) how faithful is the digital twin? (baseline sweep, twin vs real pulse counts per emitter)
    P, A = [], []
    for k in names:
        pred, _ = Tw[k].score_schedule(baseline_schedule(Ds[k]))
        act = np.bincount(Ds[k]["y"], minlength=max(Tw[k].ids) + 1)[Tw[k].ids]
        rho = spearmanr(pred[(act > 0) | (pred > 1)], act[(act > 0) | (pred > 1)]).correlation
        res["validation"][k] = dict(spearman=float(rho), real_det20=int((act >= 20).sum()), real_det100=int((act >= 100).sum()))
        P.append(pred)
        A.append(act)
    P, A = np.concatenate(P), np.concatenate(A)
    m = (A > 0) | (P > 1)
    res["validation"]["all"] = dict(spearman=float(spearmanr(P[m], A[m]).correlation),
                                    log_corr=float(np.corrcoef(np.log10(P[m] + 1), np.log10(A[m] + 1))[0, 1]),
                                    median_ratio=float(np.median(P[(P > 20) & (A > 20)] / A[(P > 20) & (A > 20)])),
                                    n=int(m.sum()))
    print("twin validation:", res["validation"]["all"], flush=True)

    # 2) leave-one-scenario-out: learn on 4 scenarios, plan + score on the 5th
    t0 = time.time()
    for held in names:
        tr = [n for n in names if n != held]
        calib = fit_calibrations([Tw[n] for n in tr], [Ds[n] for n in tr], baseline_schedule)
        lib = Library([Tw[n] for n in tr], chunk=0.1, R=120, seed=1, calib=calib)
        tau, _ = plan_allocation(lib, CENTERS, step=0.1)
        ml = stride_sequence(CENTERS, tau, step=0.1, seed=3)
        base = baseline_schedule(Ds[held])
        tw = Tw[held]
        r = dict(schedules=dict(base=base, ml=ml), tau=tau)
        for pol, sc in (("base", base), ("ml", ml)):
            for K in (K_DET, K_CHAR):
                p = p_on_grid(tw, sc, calib, K)
                r[f"p_{pol}_{K}"] = p
                r[f"curve_{pol}_{K}"] = p.sum(1)
                r[f"sum_{pol}_{K}"] = summarize_curve(p.sum(1))
                r[f"det_{pol}_{K}"] = detect_times(p)
            r[f"wasted_{pol}"] = wasted_fraction(tw, sc)
        rd, counts = real_detection_times(Ds[held], tw.ids)
        r["real_det"] = rd
        r["real_counts"] = counts
        for K in (K_DET, K_CHAR):
            ts = np.array([np.inf if v is None else v for v in rd[K]])
            r[f"curve_real_{K}"] = np.array([(ts <= g).sum() for g in GRID], float)
        res["scen"][held] = r
        save_csv(os.path.join(a.out, f"ml_schedule_scenario_{held}.csv"), ml)
        print(f"[{held}] baseline(real) det20={int((np.array(counts)>=20).sum())} | twin baseline={r['sum_base_20']['final']:.1f} "
              f"| ML={r['sum_ml_20']['final']:.1f} || K100: base {r['sum_base_100']['final']:.1f} -> ML {r['sum_ml_100']['final']:.1f}  ({time.time()-t0:.0f}s)", flush=True)
    pickle.dump(res, open(os.path.join(a.out, "cache_plan.pkl"), "wb"))


# ----------------------------------------------------------------------------------------------- signal
def find_micro(Xd, yd, lab, W=4.0, dyn_db=30.0):
    """window of W microseconds in which the most pulses from different emitters are on at the same instant"""
    s = Xd[:, 0]
    e = s + Xd[:, 2]
    amp = Xd[:, 4]
    order = np.argsort(s)
    best = None
    for i in order:
        w0 = s[i] - 0.3
        w1 = w0 + W
        sel = np.where((s < w1) & (e > w0))[0]
        if len(sel) < 2:
            continue
        top = amp[sel].max()
        sel = sel[amp[sel] >= top - dyn_db]
        labs = set(yd[sel].tolist())
        if len(labs) < 2:
            continue
        # busiest instant: how many pulses (from different emitters) are on simultaneously
        conc = 0
        for t in np.clip(s[sel], w0, w1):
            on = sel[(s[sel] <= t) & (e[sel] > t)]
            conc = max(conc, len(set(yd[on].tolist())))
        sc = conc * 10000 + len(labs) * 100 + len(sel)
        if best is None or sc > best[0]:
            best = (sc, w0, sel, conc)
    if best is None:
        return None
    _, w0, sel, conc = best
    return dict(w0=float(w0), W=W, conc=int(conc),
                pulses=[dict(t=float(s[j] - w0), pw=float(Xd[j, 2]), f=float(Xd[j, 1]), a=float(amp[j]),
                             cl=int(lab[j]), tr=int(yd[j])) for j in sel])


def cmd_signal(a):
    Ds, Tw = load_all(a.data)
    names = list(Ds)
    sig = {"deint": {}, "ident": {}, "snap": {}}
    t0 = time.time()
    # deinterleaving accuracy on the 60 most complex dwell windows of every scenario
    for k, D in Ds.items():
        X, y = D["X"], D["y"]
        toa = X[:, 0] * 1e-6
        cands = []
        for (ta, d, c) in baseline_schedule(D):
            m = (toa >= ta) & (toa < ta + d)
            if m.sum() < 200:
                continue
            ks, cn = np.unique(y[m], return_counts=True)
            cands.append((int((cn >= 10).sum()), int(m.sum()), ta, d, c))
        cands.sort(reverse=True)
        rec = tot = 0
        ari_l = []
        for (nk, n, ta, d, c) in cands[:60]:
            m = (toa >= ta) & (toa < ta + d)
            lab = merge_clusters(X[m], deinterleave(X[m], eps=2.2, min_pts=4))
            ari, r_, t_ = score(lab, y[m])
            ari_l.append(ari)
            rec += r_
            tot += t_
        sig["deint"][k] = dict(ari=float(np.mean(ari_l)), recovered=int(rec), total=int(tot))
        print(f"[{k}] deinterleave: ARI {np.mean(ari_l):.3f}, emitters recovered {rec}/{tot} ({100*rec/tot:.1f}%) {time.time()-t0:.0f}s", flush=True)

    # cluster tables for the type classifier
    from sklearn.ensemble import RandomForestClassifier
    CL = {k: dwell_clusters(D, pick_dwells(D, baseline_schedule(D))) for k, D in Ds.items()}
    accs = {}
    rfs = {}
    for held in names:
        tr = [r for k in names if k != held for r in CL[k] if r[2] >= 0.9]
        te = [r for r in CL[held] if r[2] >= 0.9]
        rf = RandomForestClassifier(300, random_state=0, n_jobs=-1).fit(np.array([r[0] for r in tr]), [r[1] for r in tr])
        rfs[held] = rf
        pred = rf.predict(np.array([r[0] for r in te]))
        pr = rf.predict_proba(np.array([r[0] for r in te]))
        top3 = np.mean([r[1] in rf.classes_[np.argsort(-p)[:3]] for r, p in zip(te, pr)])
        accs[held] = dict(top1=float(np.mean([p == r[1] for p, r in zip(pred, te)])), top3=float(top3), n=len(te))
        print(f"[{held}] type identification top-1 {accs[held]['top1']:.3f} top-3 {top3:.3f} on {len(te)} clusters", flush=True)
    sig["ident"] = accs

    # lab snapshots (real PDWs of the richest dwell windows)
    rng = np.random.default_rng(5)
    for k, D in Ds.items():
        X, y, tx = D["X"], D["y"], D["tx"]
        toa = X[:, 0] * 1e-6
        cands = []
        for (ta, d, c) in baseline_schedule(D):
            m = (toa >= ta) & (toa < ta + d)
            if m.sum() < 300:
                continue
            ks, cn = np.unique(y[m], return_counts=True)
            cands.append((int((cn >= 15).sum()), int(m.sum()), ta, d, c))
        cands.sort(reverse=True)
        snaps, used = [], set()
        for (nk, n, ta, d, c) in cands[:60]:
            if c in used:
                continue
            m = (toa >= ta) & (toa < ta + d)
            Xd = X[m].copy()
            yd = y[m]
            Xd[:, 0] = (Xd[:, 0] - ta * 1e6)                      # us relative to dwell start
            lab = merge_clusters(Xd, deinterleave(Xd, eps=2.2, min_pts=4))
            summ = [s for s in cluster_summary(Xd, lab) if s["n"] >= 8]
            if len(summ) < 5:
                continue
            micro = find_micro(Xd, yd, lab)
            if micro is None or micro["conc"] < 3:
                continue
            clusters = []
            for s in summ:
                mk = lab == s["cluster"]
                ks_, cn_ = np.unique(yd[mk], return_counts=True)
                kk = int(ks_[cn_.argmax()])
                row = dict(id=s["cluster"], n=s["n"], f=s["f_med"], flo=s["f_lo"], fhi=s["f_hi"], pw=s["pw_med"], aoa=s["aoa"],
                           pri=(None if not np.isfinite(s["pri_us"]) else s["pri_us"]), amp=s["amp_max"],
                           truth_idx=kk, truth=tx[kk]["function"], purity=float(cn_.max() / mk.sum()), pred=None)
                if s["n"] >= 25:
                    pr = rfs[k].predict_proba(np.array([feat_vec(s)]))[0]
                    o = np.argsort(-pr)[:3]
                    row["pred"] = [(str(rfs[k].classes_[i]), float(pr[i])) for i in o]
                clusters.append(row)
            # scatter sample (cap per cluster)
            keep = []
            for cid in np.unique(lab):
                idx = np.where(lab == cid)[0]
                cap = 60 if cid < 0 else 220
                if len(idx) > cap:
                    idx = rng.choice(idx, cap, replace=False)
                keep.extend(idx.tolist())
            keep = np.array(sorted(keep))
            pts = [[round(float(Xd[i, 0]) * 1e-3, 3), round(float(Xd[i, 1]), 1), round(float(Xd[i, 2]), 3), round(float(Xd[i, 4]), 1),
                    int(lab[i]), int(yd[i])] for i in keep]
            snaps.append(dict(t0=float(ta), d=float(d), c=float(c), n_pulses=int(m.sum()), n_true=int(nk), clusters=clusters, pts=pts, micro=micro))
            used.add(c)
            if len(snaps) >= 6:
                break
        snaps.sort(key=lambda s: s["c"])
        sig["snap"][k] = snaps
        print(f"[{k}] lab snapshots: {[(s['c'], len(s['clusters'])) for s in snaps]}", flush=True)
    pickle.dump(sig, open(os.path.join(a.out, "cache_signal.pkl"), "wb"))


# ----------------------------------------------------------------------------------------------- export
def r3(v, n=3):
    return None if v is None else round(float(v), n)


def cmd_export(a):
    Ds, Tw = load_all(a.data)
    plan = pickle.load(open(os.path.join(a.out, "cache_plan.pkl"), "rb"))
    sig = pickle.load(open(os.path.join(a.out, "cache_signal.pkl"), "rb"))
    out = dict(validation=plan["validation"], deint=sig["deint"], ident=sig["ident"], scen={})
    rows = []
    for k, D in Ds.items():
        tw = Tw[k]
        r = plan["scen"][k]
        em = []
        for i, e in enumerate(tw.em):
            f = np.atleast_1d(e.freqs)
            em.append(dict(id=e.idx, name=e.name, fmode=e.fmode, pri_mode=e.pri_mode, f=r3(e.mean_freq, 1), flo=r3(f.min(), 1), fhi=r3(f.max(), 1),
                           fl=[r3(x, 1) for x in f][:8], pri=r3(np.mean(e.pris), 2), pw=r3(np.mean(e.pws), 3), scan=e.scan_type, rpm=r3(e.rpm, 3),
                           bw=r3(e.bw, 2), az0=r3(e.az0, 2), p0=[r3(x, 1) for x in e.p0], v=r3(e.v, 3), ang=r3(np.degrees(e.ang), 1),
                           pt=r3(e.pt_dbm, 1), g=r3(e.gain, 1), real_n=r["real_counts"][i],
                           rd20=r3(r["real_det"][K_DET][i]), rd100=r3(r["real_det"][K_CHAR][i]),
                           bd20=r["det_base_20"][i], bd100=r["det_base_100"][i], md20=r["det_ml_20"][i], md100=r["det_ml_100"][i]))
        sc = dict(rx=[float(x) for x in D["rxd"]["start_position_km"]], B=float(tw.B), T=float(tw.T), n_emitters=len(em),
                  emitters=em,
                  base=[[r3(t), r3(d), int(c)] for t, d, c in r["schedules"]["base"]],
                  ml=[[r3(t), r3(d), int(c)] for t, d, c in r["schedules"]["ml"]],
                  curves={f"{pol}_{K}": [round(float(x), 2) for x in r[f"curve_{pol}_{K}"]] for pol in ("base", "ml", "real") for K in (20, 100)},
                  summary={pol: {K: {kk: r3(vv, 3) for kk, vv in r[f"sum_{pol}_{K}"].items()} for K in (20, 100)} for pol in ("base", "ml")},
                  wasted=dict(base=r3(r["wasted_base"]), ml=r3(r["wasted_ml"])),
                  real_final={20: int((np.array(r["real_counts"]) >= 20).sum()), 100: int((np.array(r["real_counts"]) >= 100).sum())},
                  snaps=sig["snap"][k])
        out["scen"][k] = sc
        for K in (20, 100):
            rows.append((k, K, r[f"sum_base_{K}"]["final"], r[f"sum_ml_{K}"]["final"], r[f"sum_base_{K}"]["t80"], r[f"sum_ml_{K}"]["t80"],
                         r[f"sum_base_{K}"]["auc"], r[f"sum_ml_{K}"]["auc"], sc["real_final"][K]))
    json.dump(out, open(os.path.join(a.out, "prototype_data.json"), "w"), separators=(",", ":"))
    # results table
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write("| scenario | K | real baseline | twin baseline | twin ML | gain | t80 base (s) | t80 ML (s) | AUC base | AUC ML |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for k, K, b, m, tb, tm, ab, am, real in rows:
            f.write(f"| {k} | {K} | {real} | {b:.1f} | {m:.1f} | {m-b:+.1f} | {tb:.1f} | {tm:.1f} | {ab:.2f} | {am:.2f} |\n")
        for K in (20, 100):
            rr = [x for x in rows if x[1] == K]
            f.write(f"| mean | {K} | {np.mean([x[8] for x in rr]):.1f} | {np.mean([x[2] for x in rr]):.1f} | {np.mean([x[3] for x in rr]):.1f} | "
                    f"{np.mean([x[3]-x[2] for x in rr]):+.1f} | {np.mean([x[4] for x in rr]):.1f} | {np.mean([x[5] for x in rr]):.1f} | "
                    f"{np.mean([x[6] for x in rr]):.2f} | {np.mean([x[7] for x in rr]):.2f} |\n")
    print(open(os.path.join(a.out, "results.md")).read())
    print("wrote prototype_data.json", os.path.getsize(os.path.join(a.out, "prototype_data.json")) // 1024, "KB")


# ----------------------------------------------------------------------------------------------- deploy
def cmd_deploy(a):
    """schedule for a NEW scenario: learn from every provided file, no held-out."""
    Ds, Tw = load_all(a.data)
    names = list(Ds)
    calib = fit_calibrations([Tw[n] for n in names], [Ds[n] for n in names], baseline_schedule)
    lib = Library([Tw[n] for n in names], chunk=0.1, R=160, seed=1, calib=calib)
    tau, _ = plan_allocation(lib, CENTERS, step=0.1)
    sched = stride_sequence(CENTERS, tau, step=0.1, seed=3)
    os.makedirs(a.out, exist_ok=True)
    save_csv(os.path.join(a.out, "smartscan_schedule_all_scenarios.csv"), sched)
    with open(os.path.join(a.out, "smartscan_allocation_all_scenarios.csv"), "w") as f:
        f.write("center_mhz,seconds\n")
        for c, t in zip(CENTERS, tau):
            if t > 0:
                f.write(f"{c:.0f},{t:.1f}\n")
    print("deployment schedule written:", len(sched), "dwells")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "signal", "export", "deploy"])
    ap.add_argument("--data", default="/mnt/user-data/uploads")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    {"plan": cmd_plan, "signal": cmd_signal, "export": cmd_export, "deploy": cmd_deploy}[a.cmd](a)
