"""Quick point-to-point comparison (pilot): python p2p_quick.py <out_json> <variants_json> [n_balls ...]
variants_json: {"name": [thrust, settings_file_or_null, overrides]}; moves 0.5 / 1 / 2 / 3 m at 0 and 45 deg."""
import sys, os, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ident_runs as IR
import net_sim as N
import drv2


def main():
    out = sys.argv[1]
    var = json.loads(sys.argv[2])
    ns = [int(x) for x in sys.argv[3:]] or [0]
    dists = (0.5, 1.0, 2.0, 3.0)
    cfgs, keys = [], []
    for name, (th, sp, ov) in var.items():
        base = drv2.setting(th, drv2.load_setting(sp) if sp else {})
        base = drv2.merge(base, ov or {})
        for n in ns:
            for h in (0.0, 45.0):
                for d in dists:
                    c = drv2.merge(base, {"seed": 101, "label": f"{name}_n{n}_{d:g}_{h:g}",
                                          "p2p": {"d": [d * np.cos(np.radians(h)), d * np.sin(np.radians(h)), 0.0],
                                                  "n_drop": n, "t_after_s": 6.0}})
                    if th == 5:
                        c["h_dn_table"] = IR.H_DN10
                    for k in ("netdyn",):
                        if k in (ov or {}):
                            c[k] = ov[k]
                    cfgs.append(c)
                    keys.append((name, n, h, d))
    res = N.run_catch_batch(cfgs, drv2.JOBS, None, "p2p_quick")
    rows = []
    for (name, n, h, d), m in zip(keys, res):
        if "error" in m:
            rows.append({"variant": name, "n": n, "heading": h, "dist": d, "error": m["error"]})
            print(name, n, h, d, "ERROR", m["error"])
            continue
        p = m["p2p"]
        rows.append({"variant": name, "n": n, "heading": h, "dist": d, **p, "netdyn": m.get("netdyn")})
    rows.sort(key=lambda r: (r["n"], r["heading"], r["dist"], r["variant"]))
    print("variant              n  head dist  T_plan ready10 ready5 overshoot lag_max end_mm  disc  guard motor  tilt")
    for r in rows:
        if "error" in r:
            continue
        print(f"{r['variant']:20s} {r['n']:2d} {r['heading']:4.0f} {r['dist']:4.1f}  {r['T_plan_s']:5.2f}  "
              f"{r['t_ready_100mm_s']}  {r['t_ready_50mm_s']}  {r['overshoot_mm']:6.1f}  {r['net_lag_max_mm']:6.1f} "
              f"{r['net_to_target_end_mm']:5.1f}  {r['disc_min']['mm']:5.1f} {r['guard_min']['mm']:5.1f} "
              f"{r['motor_max']:.3f} {r['tilt_max_deg']:5.1f}")
    json.dump(rows, open(out, "w"), indent=1, default=N._json_np)


if __name__ == "__main__":
    main()
