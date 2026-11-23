"""Reduced model relative to the planner's reference (the model the planner uses).

The formation is not a perfect servo: its centre lags the reference by up to ~50 mm in a 1 m move (drone attitude /
rotor lag, position loop with poles at -2 rad/s), as large as the net's own swing. So both the formation centre and the
net centre are modelled as offsets from the formation REFERENCE r, driven by the reference acceleration r'' and velocity
r' (the planner's own input):
    d_i'' + 2 zeta_i w_i d_i' + w_i^2 d_i = -g_i r'' - c_i r',   offset = sum_i d_i (+ const per run)
Net centre (net_c - r) and formation centre (mean drone position - r), horizontal (x, y) and vertical, m = 1..3 modes,
per thrust and ball count. Train: heading 0, oblique and vertical moves; test: heading 45 (horizontal).
python ident_fit_ref.py set [set ...]   (x1, x5, x5b) -> ident/model_ref_<sets>.json / .txt"""
import os, sys, json
import numpy as np
from scipy.signal import savgol_filter
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ident_fit as F

OUT = F.OUT


def signals_ref(r, ax, key):
    rc = r["ref_c"].astype(float)[:, ax]
    rv = savgol_filter(rc, 9, 3, deriv=1, delta=F.DT)
    ra = savgol_filter(rc, 9, 3, deriv=2, delta=F.DT)
    X = r["pos"].mean(axis=1).astype(float)[:, ax] if key == "form" else r[key].astype(float)[:, ax]
    return ra, rv, X - rc


def main():
    res, lines = {}, []
    sets = sys.argv[1:] or ["x1", "x5"]
    for name in sets:
        runs = F.load(name)
        ns = sorted({int(r["label"].split("_n")[1].split("_")[0]) for r in runs})
        for n in ns:
            rn = [r for r in runs if f"_n{n}_" in r["label"]]
            for part, axes, kind, key, ms in (("net_h", (0, 1), ("h", "o"), "net_c", (1, 2, 3)),
                                              ("net_v", (2,), ("v", "o"), "net_c", (1, 2)),
                                              ("form_h", (0, 1), ("h", "o"), "form", (1, 2, 3)),
                                              ("form_v", (2,), ("v", "o"), "form", (1, 2))):
                sel = [r for r in rn if r["label"].split("_")[2][0] in kind]
                segs_all = [(r["label"], ax) + signals_ref(r, ax, key) for r in sel for ax in axes]
                tr = [s for s in segs_all if not s[0].endswith("_45")]
                te = [s for s in segs_all if s[0].endswith("_45")]
                kk = f"{name[:2]}_n{n}_{part}"
                sig = np.concatenate([s[4] - s[4].mean() for s in segs_all])
                res[kk] = {"signal_rms_mm": round(1e3 * float(np.std(sig)), 2), "models": {}}
                for m in ms:
                    th0 = [5.5, 0.3, 9.0, 0.2, 3.0, 0.8][:2 * m]
                    try:
                        th, gd, err, y = F.fit([s[2:] for s in tr], m, th0)
                    except Exception as exc:
                        res[kk]["models"][m] = {"error": repr(exc)}
                        continue
                    ent = {"omega": [round(float(th[2 * i]), 4) for i in range(m)],
                           "zeta": [round(float(th[2 * i + 1]), 4) for i in range(m)],
                           "g": [round(float(x), 5) for x in gd[:m]], "d": [round(float(x), 5) for x in gd[m:]],
                           "train": F.stats(err)}
                    if te:
                        ent["test"] = F.stats(F.predict(th, gd, [s[2:] for s in te], m))
                    res[kk]["models"][m] = ent
                    lines.append(f"{kk:16s} m={m} w {ent['omega']} z {ent['zeta']} g {ent['g']} d {ent['d']} "
                                 f"train {ent['train']} test {ent.get('test')} (signal rms {res[kk]['signal_rms_mm']})")
                    print(lines[-1], flush=True)
    tag = "_".join(sets)
    json.dump(res, open(os.path.join(OUT, "ident", f"model_ref_{tag}.json"), "w"), indent=1)
    open(os.path.join(OUT, "ident", f"model_ref_{tag}.txt"), "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
