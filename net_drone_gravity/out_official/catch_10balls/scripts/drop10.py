"""Drop test with 10 balls (formation hovering, planner off): balls released at rest 1 m above the net centre every 3 s,
a few cm apart. Per state (0..10 balls in the net, mean over the 1 s before the next drop): drone-to-net-centre drop,
sag, net centre height, lowest knot, mean / max tilt, mean / max motor share, min rotor-disc and guard clearance.
python drop10.py [thrust ...]   (default 5 1)"""
import sys, os, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)
import net_sim as N

OUT = os.path.join(REPO, "out_official", "catch_10balls", "drop")


def run(thrust, nb=10, period=3.0):
    base = {**N.CATCH_DEFAULT}
    if thrust != 1:
        base["p_over"] = {"f_max": N.OfficialParams().f_max * thrust, "max_tilt_deg": 40.0}
    h_dn = N.hold_drop()
    p0 = N.catch_params(base)
    zc = p0.z_hover - h_dn
    launches = [{"dt": period * i, "p0": [0.03 * np.cos(2.4 * i), 0.03 * np.sin(2.4 * i), zc + 1.0],
                 "v0": [0.0, 0.0, 0.0], "v_prior": [0.0, 0.0, 0.0]} for i in range(nb)]
    cfg = {**base, "tier": None, "planner_off": True, "standby": [0.0, 0.0, p0.z_hover], "z_net0": zc, "h_dn": h_dn,
           "h_dn_per_ball_m": 0.0, "h_dn_table": [0.0], "return_standby": False, "launches": launches,
           "t_after_last_max_s": 6.0, "label": f"drop10_x{thrust}", "seed": 1, "ff_balls": True}
    tag = f"drop10_x{thrust}"
    m = N.run_catch(cfg, out=OUT, tag=tag, verbose=False)
    L = dict(np.load(os.path.join(OUT, f"{tag}_log.npz")))
    p = N.catch_params(cfg)
    p.n_balls = nb
    t = L["t"]
    tl = [b["t_launch"] for b in m["balls"]]
    ends = tl + [float(t[-1]) + 1e-9]
    phases = [("all", 0.0, float(t[-1]) + 1.0)]
    cl, cls = N.clearance_analysis(p, L, phases)
    ct = cls["t"]
    dmin, gmin = cls["all"].min(1), np.nanmin(cls["guard"], axis=1)
    rows = []
    for n in range(nb + 1):
        w = (t >= ends[n] - 1.0) & (t < ends[n])          # 1 s before the next drop (n balls in the net)
        wc = (ct >= ends[n] - 1.0) & (ct < ends[n])
        in_net = int(sum(L["ball_contact"][w, b].mean() > 0.9 for b in range(nb)))
        rows.append({"balls_dropped": n, "balls_in_net": in_net,
                     "h_dn_m": round(float(L["pos"][w, :, 2].mean() - L["net_c"][w, 2].mean()), 4),
                     "net_centre_z_m": round(float(L["net_c"][w, 2].mean()), 4),
                     "drones_z_m": round(float(L["pos"][w, :, 2].mean()), 4),
                     "sag_m": round(float(L["sag_net"][w].mean()), 4),
                     "lowest_knot_z_m": round(float(L["net_zmin"][w].min()), 4),
                     "lowest_knot_below_drones_m": round(float(L["pos"][w, :, 2].mean() - L["net_zmin"][w].min()), 4),
                     "tilt_mean_deg": round(float(L["tilt"][w].mean()), 2), "tilt_max_deg": round(float(L["tilt"][w].max()), 2),
                     "motor_mean": round(float((L["motor"][w] / p.f_max).mean()), 4),
                     "motor_max": round(float((L["motor"][w] / p.f_max).max()), 4),
                     "tension_mean_gf": round(float(L["tension"][w].mean() / N.G * 1e3), 2),
                     "disc_min_mm": round(float(1e3 * dmin[wc].min()), 1), "guard_min_mm": round(float(1e3 * gmin[wc].min()), 1)})
    h0 = rows[0]["h_dn_m"]
    res = {"thrust": thrust, "period_s": period, "rows": rows,
           "h_dn_table_m": [round(r["h_dn_m"] - h0, 4) for r in rows],
           "whole_run": {"disc_min_mm": round(float(1e3 * dmin[ct >= tl[0]].min()), 1),
                         "guard_min_mm": round(float(1e3 * gmin[ct >= tl[0]].min()), 1),
                         "motor_max": round(float((L["motor"][t >= tl[0]] / p.f_max).max()), 4),
                         "tilt_max_deg": round(float(L["tilt"][t >= tl[0]].max()), 2),
                         "lowest_knot_z_m": round(float(L["net_zmin"][t >= tl[0]].min()), 4)},
           "balls": [{k: b.get(k) for k in ("outcome", "reason", "t_first_contact", "net_contact_s")} for b in m["balls"]],
           "safety_violations": m["safety_violations"]}
    return res


if __name__ == "__main__":
    th = [float(x) for x in sys.argv[1:]] or [5, 1]
    os.makedirs(OUT, exist_ok=True)
    allr = {}
    for x in th:
        x = int(x) if x == int(x) else x
        r = run(x)
        allr[f"x{x}"] = r
        print(f"thrust x{x}: caught {sum(b['outcome'] == 'caught' for b in r['balls'])}/10, violations {r['safety_violations']}")
        print("n  in  h_dn    dh_tab  netc   sag    lowknot below  tilt_m tilt_x motor_m motor_x disc  guard")
        for row, dh in zip(r["rows"], r["h_dn_table_m"]):
            print(f"{row['balls_dropped']:2d} {row['balls_in_net']:2d}  {row['h_dn_m']:.4f} {dh:.4f} {row['net_centre_z_m']:.3f} "
                  f"{row['sag_m']:.3f}  {row['lowest_knot_z_m']:.3f}  {row['lowest_knot_below_drones_m']:.3f}  "
                  f"{row['tilt_mean_deg']:5.1f}  {row['tilt_max_deg']:5.1f}  {row['motor_mean']:.3f}  {row['motor_max']:.3f}  "
                  f"{row['disc_min_mm']:5.1f} {row['guard_min_mm']:5.1f}")
    with open(os.path.join(OUT, "drop10.json"), "w") as f:
        json.dump(allr, f, indent=1, default=N._json_np)
