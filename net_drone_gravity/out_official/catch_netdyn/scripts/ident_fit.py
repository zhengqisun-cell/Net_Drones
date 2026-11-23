"""Reduced net model fitted to the identification runs (ident/raw_<set>.npz from ident_runs.py).

Model (per axis, net centre relative to the formation centre e = x_net - x_formation):
    sum of m damped modes  e_i'' + 2 zeta_i w_i e_i' + w_i^2 e_i = -g_i a_f - d_i v_f,   e = sum_i e_i (+ const)
input a_f / v_f: formation-centre acceleration / velocity (measured: mean drone velocity and its derivative).
Horizontal (x, y share the parameters) and vertical fitted separately, per thrust and ball count n.
Fit: variable projection (outer: w_i, zeta_i by least squares; inner: gains g_i, d_i shared, initial conditions and
offset per run and axis, linear). Error: simulated e against the logged one (= net-centre trajectory error), mm.
python ident_fit.py set [set ...]   -> ident/model_fit_<sets>.json / .txt (sets: x1, x5, x5b)"""
import os, sys, json
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import savgol_filter, cont2discrete, lfilter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
DT = 0.01


def load(name):
    z = np.load(os.path.join(OUT, "ident", f"raw_{name}.npz"))
    meta = json.load(open(os.path.join(OUT, "ident", f"raw_{name}_meta.json")))
    runs = []
    for m in meta:
        lab = m["label"]
        d = {k.split("|")[1]: z[k] for k in z.files if k.split("|")[0] == lab}
        d["label"], d["t_move"], d["p2p"] = lab, m["t_move"], m["p2p"]
        runs.append(d)
    return runs


def signals(r, ax, key="net_c"):
    """(a_f, v_f, e) of run r, axis ax (0, 1, 2), on a uniform 10 ms grid."""
    vf = r["vel"].mean(axis=1).astype(float)[:, ax]
    fc = r["pos"].mean(axis=1).astype(float)[:, ax]
    af = savgol_filter(vf, 9, 3, deriv=1, delta=DT)
    e = r[key].astype(float)[:, ax] - fc
    return af, vf, e


def mode_resp(u, w, z):
    """Response of 1 / (s^2 + 2 z w s + w^2) to input u (ZOH, zero initial state)."""
    A = np.array([[0.0, 1.0], [-w * w, -2 * z * w]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0]])
    Ad, Bd, Cd = cont2discrete((A, B, C, np.zeros((1, 1))), DT, method="zoh")[:3]
    # transfer function of the discrete system
    from scipy.signal import ss2tf
    num, den = ss2tf(Ad, Bd, Cd, np.zeros((1, 1)))
    return lfilter(num[0], den, u)


def free_resp(n, w, z):
    t = np.arange(n) * DT
    wd = w * np.sqrt(max(1 - z * z, 1e-6))
    ex = np.exp(-z * w * t)
    return ex * np.cos(wd * t), ex * np.sin(wd * t)


def design(theta, segs, m):
    """Regression matrix for the linear parameters: [g_1..g_m, d_1..d_m] shared, then per segment 2 m ICs + offset."""
    nseg = len(segs)
    rows, ys = [], []
    nlin = 2 * m + nseg * (2 * m + 1)
    for s_i, (af, vf, e) in enumerate(segs):
        n = len(e)
        X = np.zeros((n, nlin))
        for i in range(m):
            w, z = theta[2 * i], theta[2 * i + 1]
            X[:, i] = mode_resp(-af, w, z)
            X[:, m + i] = mode_resp(-vf, w, z)
            c, s = free_resp(n, w, z)
            X[:, 2 * m + s_i * (2 * m + 1) + 2 * i] = c
            X[:, 2 * m + s_i * (2 * m + 1) + 2 * i + 1] = s
        X[:, 2 * m + s_i * (2 * m + 1) + 2 * m] = 1.0
        rows.append(X)
        ys.append(e)
    return np.vstack(rows), np.concatenate(ys)


def fit(segs, m, theta0):
    def res(th):
        X, y = design(th, segs, m)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return X @ c - y
    lo = [2.0, 0.0] * m
    hi = [40.0, 1.5] * m
    s = least_squares(res, theta0, bounds=(lo, hi), x_scale=[5.0, 0.1] * m)
    X, y = design(s.x, segs, m)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return s.x, c[:2 * m], X @ c - y, y


def predict(theta, gd, segs, m):
    """Prediction of held-out segments with the fitted shared parameters (ICs and offset fitted per segment: they are
    the state at the window start, which the planner gets from its own model / a measurement)."""
    errs = []
    for af, vf, e in segs:
        n = len(e)
        forced = np.zeros(n)
        B = [np.ones(n)]
        for i in range(m):
            w, z = theta[2 * i], theta[2 * i + 1]
            forced += gd[i] * mode_resp(-af, w, z) + gd[m + i] * mode_resp(-vf, w, z)
            B += list(free_resp(n, w, z))
        B = np.column_stack(B)
        c, *_ = np.linalg.lstsq(B, e - forced, rcond=None)
        errs.append(forced + B @ c - e)
    return errs


def split(runs, ax_set, key="net_c"):
    out = []
    for r in runs:
        for ax in ax_set:
            out.append((r["label"], ax) + signals(r, ax, key))
    return out


def stats(err):
    e = np.concatenate(err) if isinstance(err, list) else err
    return {"rms_mm": round(1e3 * float(np.sqrt(np.mean(e ** 2))), 2), "max_mm": round(1e3 * float(np.abs(e).max()), 1)}


def main():
    res, lines = {}, []
    for name in (sys.argv[1:] or ["x1", "x5"]):
        if not os.path.exists(os.path.join(OUT, "ident", f"raw_{name}.npz")):
            continue
        runs = load(name)
        ns = sorted({int(r["label"].split("_n")[1].split("_")[0]) for r in runs})
        for n in ns:
            rn = [r for r in runs if f"_n{n}_" in r["label"]]
            for part, axes, kind, key, ms in (("horizontal", (0, 1), ("h", "o"), "net_c", (1, 2)),
                                              ("vertical", (2,), ("v", "o"), "net_c", (1, 2)),
                                              ("horizontal_com", (0, 1), ("h", "o"), "net_com", (1,)),
                                              ("vertical_com", (2,), ("v", "o"), "net_com", (1,))):
                sel = [r for r in rn if r["label"].split("_")[2][0] in kind]
                segs_all = split(sel, axes, key)
                # train: moves along x (heading 0) and the oblique / vertical ones; test: heading 45 (horizontal)
                tr = [s for s in segs_all if not s[0].endswith("_45")]
                te = [s for s in segs_all if s[0].endswith("_45")]
                sig = np.concatenate([s[4] for s in segs_all])
                key = f"{name}_n{n}_{part}"
                res[key] = {"signal_rms_mm": round(1e3 * float(np.std(sig)), 2), "models": {}}
                for m, th0 in [(m_, [7.0, 0.15] if m_ == 1 else [7.0, 0.15, 14.0, 0.1]) for m_ in ms]:
                    try:
                        th, gd, err, y = fit([s[2:] for s in tr], m, th0)
                    except Exception as exc:
                        res[key]["models"][m] = {"error": repr(exc)}
                        continue
                    ent = {"omega": [round(float(th[2 * i]), 3) for i in range(m)],
                           "zeta": [round(float(th[2 * i + 1]), 4) for i in range(m)],
                           "g": [round(float(x), 5) for x in gd[:m]], "d": [round(float(x), 5) for x in gd[m:]],
                           "train": stats(err)}
                    if te:
                        ent["test"] = stats(predict(th, gd, [s[2:] for s in te], m))
                    res[key]["models"][m] = ent
                    lines.append(f"{key:22s} m={m} w {ent['omega']} z {ent['zeta']} g {ent['g']} d {ent['d']} "
                                 f"train {ent['train']} test {ent.get('test')} (signal rms {res[key]['signal_rms_mm']})")
                    print(lines[-1], flush=True)
    tag = "_".join(sys.argv[1:]) or "x1_x5"
    json.dump(res, open(os.path.join(OUT, "ident", f"model_fit_{tag}.json"), "w"), indent=1)
    open(os.path.join(OUT, "ident", f"model_fit_{tag}.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
