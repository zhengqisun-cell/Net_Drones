"""Point-to-point benchmark (A): moves 0.5-3 m at 0 / 45 deg from the standby point, with n balls dropped into the net
first (p2p n_drop); variants from nd_runs.candidates of a family (x1_T0 / x5_T0) or settings/<name>.json.
python p2p_bench.py <family> <variant[,variant]> <n[,n]> [tag]   -> p2p/p2p_<tag>.json / .txt"""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nd_runs as R
from nd_runs import N, merge
import ident_runs as IR

DISTS = (0.5, 1.0, 1.5, 2.0, 3.0)
HEADS = (0.0, 45.0)
COLS = ("T_plan_s", "t_ready_100mm_s", "t_ready_50mm_s", "overshoot_mm", "net_lag_max_mm", "net_swing_after_mm",
        "net_to_target_end_mm")


def main():
    fam, vars_, ns = sys.argv[1], sys.argv[2].split(","), [int(x) for x in sys.argv[3].split(",")]
    tag = sys.argv[4] if len(sys.argv) > 4 else fam
    th = R.FAMILIES[fam][0]
    cfgs, keys = [], []
    for v in vars_:
        base = R.cfg_for(fam, v)
        for n in ns:
            for h in HEADS:
                for d in DISTS:
                    c = merge(base, {"seed": 101, "label": f"{v}_n{n}_{d:g}_{h:g}",
                                     "p2p": {"d": [d * np.cos(np.radians(h)), d * np.sin(np.radians(h)), 0.0],
                                             "n_drop": n, "t_after_s": 6.0}})
                    if "netdyn" in base:
                        c["netdyn"] = base["netdyn"]
                    if th == 5:
                        c["h_dn_table"] = IR.H_DN10
                    cfgs.append(c)
                    keys.append((v, n, h, d))
    res = N.run_catch_batch(cfgs, R.drv2.JOBS, None, f"p2p_{tag}")
    rows = []
    for (v, n, h, d), m in zip(keys, res):
        r = {"variant": v, "n": n, "heading_deg": h, "distance_m": d}
        if "error" in m:
            r["error"] = m["error"]
        else:
            r.update(m["p2p"])
            if "netdyn" in m:
                st = m["netdyn"]["stats"]
                r["netdyn"] = st
                if st["err_n"]:
                    r["state_err_model_rms_mm"] = round(1e3 * np.sqrt(st["err_model_ss"] / st["err_n"]), 1)
                    if st["err_kf_ss"]:
                        r["state_err_kf_rms_mm"] = round(1e3 * np.sqrt(st["err_kf_ss"] / st["err_n"]), 1)
        rows.append(r)
    os.makedirs(os.path.join(R.OUT, "p2p"), exist_ok=True)
    json.dump(rows, open(os.path.join(R.OUT, "p2p", f"p2p_{tag}.json"), "w"), indent=1, default=N._json_np)
    L = ["variant   n head dist T_plan ready10 ready5 overshoot lag_max swing_after end_mm  disc guard motor   tilt  "
         "state_err(model/kf, mm rms)"]
    for r in sorted(rows, key=lambda r: (r["n"], r["heading_deg"], r["distance_m"], vars_.index(r["variant"]))):
        if "error" in r:
            L.append(f"{r['variant']} {r['n']} {r['heading_deg']} {r['distance_m']} ERROR")
            continue
        L.append(f"{r['variant']:9s} {r['n']:2d} {r['heading_deg']:4.0f} {r['distance_m']:4.1f} {r['T_plan_s']:6.2f} "
                 f"{r['t_ready_100mm_s']!s:>7} {r['t_ready_50mm_s']!s:>6} {r['overshoot_mm']:9.1f} {r['net_lag_max_mm']:7.1f} "
                 f"{r['net_swing_after_mm']!s:>11} {r['net_to_target_end_mm']:6.1f} {r['disc_min']['mm']:5.1f} "
                 f"{r['guard_min']['mm']:5.1f} {r['motor_max']:.3f} {r['tilt_max_deg']:6.2f}  "
                 f"{r.get('state_err_model_rms_mm', '')}/{r.get('state_err_kf_rms_mm', '')}")
    # means per variant and ball count
    L.append("")
    L.append("means over the 10 moves (5 distances x 2 headings): variant n  T_plan ready10 ready5 overshoot lag_max "
             "swing_after disc_min guard_min motor_max tilt_max")
    for n in ns:
        for v in vars_:
            sel = [r for r in rows if r["variant"] == v and r["n"] == n and "error" not in r]
            if not sel:
                continue
            mean = lambda k: np.mean([r[k] for r in sel if r[k] is not None])
            L.append(f"  {v:9s} {n:2d} {mean('T_plan_s'):6.2f} {mean('t_ready_100mm_s'):6.2f} {mean('t_ready_50mm_s'):6.2f} "
                     f"{mean('overshoot_mm'):8.1f} {mean('net_lag_max_mm'):7.1f} {mean('net_swing_after_mm'):8.1f} "
                     f"{min(r['disc_min']['mm'] for r in sel):6.1f} {min(r['guard_min']['mm'] for r in sel):6.1f} "
                     f"{max(r['motor_max'] for r in sel):.3f} {max(r['tilt_max_deg'] for r in sel):6.2f}")
    open(os.path.join(R.OUT, "p2p", f"p2p_{tag}.txt"), "w").write("\n".join(L) + "\n")
    print("\n".join(L[-(len(ns) * len(vars_) + 2):]))


if __name__ == "__main__":
    main()
