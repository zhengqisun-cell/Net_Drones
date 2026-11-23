"""Feedforward pilot: 1 m move along x at thrust x5 (x5_T0 settings, old S-curve moves, netdyn plan off), formation
tracking error (formation centre - reference) and readiness with: no feedforward; string-force feedforward (net mass /
4 x predicted net acceleration, plus net drag) with acceleration lead 0 / 40 / 70 / 100 / 140 ms; lead alone; drag
feedforward with and without lead. python ff_pilot.py -> ff_pilot.txt / .json"""
import os, sys, json, tempfile, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nd_runs as R
from nd_runs import N, merge


def main():
    V = {"none": None}
    for ff in (False, True):
        for lead in (0.0, 0.04, 0.07, 0.1, 0.14):
            if ff or lead:
                V[f"string_ff{int(ff)}_lead{lead:g}"] = {"plan": False, "ff": ff, "lead_s": lead, "drag_ff": False}
    V["drag"] = {"plan": False, "drag_ff": True, "lead_s": 0.0}
    V["drag_lead0.07"] = {"plan": False, "drag_ff": True, "lead_s": 0.07}
    tmp = tempfile.mkdtemp(prefix="netdyn_ff_")
    try:
        cfgs = []
        for k, nd in V.items():
            c = merge(R.cfg_for("x5_T0", "base"), {"seed": 101, "p2p": {"d": [1.0, 0, 0]}, "label": k})
            c["netdyn"] = nd
            c["_out"], c["_tag"], c["_save_log"] = tmp, k, True
            cfgs.append(c)
        res = N.run_catch_batch(cfgs, 14, tmp, "ff_pilot")
        rows, L = [], ["1 m move along x, thrust x5, x5_T0 settings (S-curve), seed 101: formation tracking error along x "
                       "(max / rms over the move and the 6 s after it), readiness and net lag"]
        for k, m in zip(V, res):
            p = m["p2p"]
            Lg = np.load(os.path.join(tmp, f"{k}_log.npz"))
            w = Lg["t"] >= p["t_move_s"]
            e = Lg["pos"].mean(1)[w, 0] - Lg["ref"].mean(1)[w, 0]
            r = {"variant": k, "netdyn": V[k], "form_err_max_mm": round(1e3 * float(np.abs(e).max()), 1),
                 "form_err_rms_mm": round(1e3 * float(np.sqrt((e ** 2).mean())), 1),
                 **{q: p[q] for q in ("t_ready_50mm_s", "t_ready_100mm_s", "overshoot_mm", "net_lag_max_mm",
                                      "tilt_max_deg")}, "disc_min_mm": p["disc_min"]["mm"]}
            Lg.close()
            rows.append(r)
            L.append(f"{k:22s} formation error max {r['form_err_max_mm']:6.1f} rms {r['form_err_rms_mm']:5.1f} mm  "
                     f"ready 5 cm {r['t_ready_50mm_s']} s  overshoot {r['overshoot_mm']} mm  net lag {r['net_lag_max_mm']} "
                     f"mm  tilt {r['tilt_max_deg']}  disc {r['disc_min_mm']}")
        open(os.path.join(R.OUT, "ff_pilot.txt"), "w").write("\n".join(L) + "\n")
        json.dump(rows, open(os.path.join(R.OUT, "ff_pilot.json"), "w"), indent=1)
        print("\n".join(L))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
