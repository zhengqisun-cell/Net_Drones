"""Drivers for the relaxed-safety-line study (out_official/catch_speed_relaxed). Runs only net_sim functions."""
import sys, os, json, copy, time
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)
import numpy as np
import net_sim as N

OUT = os.path.join(REPO, "out_official", "catch_speed_relaxed")
X5 = {"p_over": {"f_max": 1.0, "max_tilt_deg": 40.0}, "loaded_accel_factor": 1.0, "standby_accel_factor": 1.0}
GROUPS = {"med_seq2": {"tier": "medium", "launch": "sequence", "interval_s": 2.0},
          "med_seq1": {"tier": "medium", "launch": "sequence", "interval_s": 1.0},
          "hard_seq2": {"tier": "hard", "launch": "sequence", "interval_s": 2.0},
          "med_salvo": {"tier": "medium", "launch": "salvo", "interval_s": 2.0}}
G4 = tuple(GROUPS)
JOBS = 22
# safety-line tiers: motor share line (None: none, saturation only reported), rotor-disc clearance line [mm]
TIERS = {0: {"motor": 0.90, "disc_mm": 30.0}, 1: {"motor": 0.95, "disc_mm": 25.0},
         2: {"motor": 0.98, "disc_mm": 20.0}, 3: {"motor": None, "disc_mm": 15.0}}
MOTOR_TIME_S = 0.10
GUARD_MM = 10.0
# tier-0 recommended settings (catch_speed round): the starting points
REC0 = {1: {"traj": {"jerk": 30.0}, "limits": {"a_v": 1.6}},
        5: {"limits": {"v_h": 3.0, "a_h": 3.0, "v_up": 3.0, "v_down": 2.0, "a_v": 2.0, "a_tot": 3.0},
            "loaded_accel_factor": 1.0, "standby_accel_factor": 1.0, "traj": {"jerk": 20.0, "diag_factor": 0.9}}}


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if k in ("p_over", "limits", "traj") and isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = copy.deepcopy(v)
    return out


def setting(thrust, over=None):
    base = copy.deepcopy(X5) if thrust == 5 else {}
    return merge(base, over)


def trial_rec(s, m):
    sf = m["safety"]
    mo = sf["motor90"]
    return {"seed": s, "n_balls": m["n_balls"], "n_caught": m["n_caught"],
            "n_launched": sum(b["outcome"] != "not_launched" for b in m["balls"]),
            "all_caught": m["all_caught"], "viol_old": m["safety_violations"], "aborted": m["aborted"],
            "abort": sf["control"]["abort"],
            "reasons": [b.get("reason") for b in m["balls"]],
            "disc_min_mm": sf["clearance"]["disc_min_mm"], "guard_min_mm": sf["clearance"]["guard_min_mm"],
            "motor_max": mo["motor_max"], "motor_lines": mo["lines"],
            "sat_total_s": mo["saturated_total_s"], "sat_longest_s": mo["saturated_longest_s"],
            "at_min_thrust_s": mo["at_min_thrust_s"],
            "tilt_max_deg": sf["control"]["tilt_max_deg"],
            "shell_hits": sf["ball_rotor"]["shell_hits"], "ball_disc_gap_min_mm": sf["ball_rotor"]["disc_gap_min_mm"],
            "fell_out": sum(b.get("reason") == "fell_out" for b in m["balls"]),
            "hard_ok": {k: sf[k]["ok"] for k in ("net_low", "separation", "slack", "box")},
            "net_lowest_m": sf["net_low"]["net_lowest_m"], "slack_max_s": sf["slack"]["slack_max_s"]}


def tier_viol(t, k, form="dur", ball="fixed"):
    """Violated items of trial record t at safety tier k. form 'dur': motor line exceeded for > 0.10 s continuously
    (the existing rule's form); 'peak': motor peak above the line. ball 'fixed': the ball-to-rotor-disc gap keeps the
    existing 30 mm rule in every tier (the sim has no rotor downwash, so a 2.7 g ball near a rotor is riskier than
    modelled); 'relaxed': the gap uses the tier's disc line (extra column only)."""
    T = TIERS[k]
    v = []
    if t["aborted"]:
        v.append("control")
    if T["motor"] is not None:
        if form == "peak":
            if t["motor_max"] > T["motor"]:
                v.append("motor")
        elif t["motor_lines"][f"{T['motor']:.2f}"]["longest_s"] > MOTOR_TIME_S + 1e-9:
            v.append("motor")
    if k == 0:      # the existing rule's exact flags (the summary's clearances are rounded to 0.1 mm)
        if "clearance" in t["viol_old"]:
            v.append("disc" if t["disc_min_mm"] < T["disc_mm"] + 0.05 else "guard")
        if t["shell_hits"] > 0:
            v.append("ball_hit")
        elif "ball_rotor" in t["viol_old"]:
            v.append("ball_disc")
    else:
        if t["disc_min_mm"] < T["disc_mm"]:
            v.append("disc")
        if t["guard_min_mm"] < GUARD_MM:
            v.append("guard")
        if t["shell_hits"] > 0:
            v.append("ball_hit")
        elif ball == "fixed":
            if "ball_rotor" in t["viol_old"]:       # exact existing flag (gap < 30 mm)
                v.append("ball_disc")
        elif t["ball_disc_gap_min_mm"] is not None and t["ball_disc_gap_min_mm"] < T["disc_mm"]:
            v.append("ball_disc")
    for kk, ok in t["hard_ok"].items():
        if not ok:
            v.append(kk)
    return v


def agg_group(ms, seeds):
    ms_ok = [(s, m) for s, m in zip(seeds, ms) if "error" not in m]
    a = N.catch_agg([m for _, m in ms_ok])
    a["excluded_not_launched"] = [{"seed": s, "ball": b["ball"] + 1} for s, m in ms_ok for b in m["balls"]
                                  if b["outcome"] == "not_launched"]
    a["errors"] = [s for s, m in zip(seeds, ms) if "error" in m]
    a["n_all_caught"] = int(sum(m["all_caught"] for _, m in ms_ok))
    a["per_trial"] = [trial_rec(s, m) for s, m in ms_ok]
    a["tiers"] = tier_summary(a["per_trial"])
    return a


def tier_summary(pt):
    out = {}
    for k in TIERS:
        for form in ("dur", "peak"):
            if form == "peak" and TIERS[k]["motor"] is None:
                continue
            vs = [tier_viol(t, k, form) for t in pt]
            items = {}
            for v in vs:
                for x in v:
                    items[x] = items.get(x, 0) + 1
            out[f"{k}_{form}"] = {"viol_trials": int(sum(bool(v) for v in vs)), "items": items,
                                  "all_caught_clean": int(sum(t["all_caught"] and not v for t, v in zip(pt, vs)))}
    return out


def catch_groups(settings, seeds, groups=G4, name="groups"):
    """settings: {name: (thrust, overrides)}. Returns {setting: {group: agg}}."""
    cfgs, keys = [], []
    for sn, (th, ov) in settings.items():
        for g in groups:
            for s in seeds:
                c = merge(setting(th, ov), {**GROUPS[g], "seed": s, "label": f"{sn}_{g}_seed{s}"})
                cfgs.append(c)
                keys.append((sn, g, s))
    res = N.run_catch_batch(cfgs, JOBS, None, name)
    out = {}
    for sn in settings:
        out[sn] = {}
        for g in groups:
            idx = [i for i, k in enumerate(keys) if k[0] == sn and k[1] == g]
            out[sn][g] = agg_group([res[i] for i in idx], [keys[i][2] for i in idx])
    return out, res, keys


def setting_total(gr, k=None, form="dur", ball="fixed"):
    pt = [t for a in gr.values() for t in a["per_trial"]]
    c = sum(a["caught"] for a in gr.values())
    b = sum(a["balls"] for a in gr.values())
    r = {"caught": c, "balls": b, "trials": len(pt), "aborts": int(sum(t["aborted"] for t in pt)),
         "disc_min_mm": min(t["disc_min_mm"] for t in pt), "guard_min_mm": min(t["guard_min_mm"] for t in pt),
         "motor_max": max(t["motor_max"] for t in pt), "sat_longest_s": max(t["sat_longest_s"] for t in pt),
         "tilt_max": max(t["tilt_max_deg"] for t in pt)}
    if k is not None:
        vs = [tier_viol(t, k, form, ball) for t in pt]
        r["viol"] = int(sum(bool(v) for v in vs))
        items = {}
        for v in vs:
            for x in v:
                items[x] = items.get(x, 0) + 1
        r["items"] = items
    return r


def brief(a):
    tv = " ".join(f"T{k}:{a['tiers'][f'{k}_dur']['viol_trials']}" for k in TIERS)
    pt = a["per_trial"]
    return (f"{a['caught']}/{a['balls']} ({100 * a['caught_share_balls']:.1f}%) all {a['n_all_caught']}/{a['trials']} "
            f"viol[{tv}] abort {a['aborted']} excl {len(a['excluded_not_launched'])} "
            f"disc {min(t['disc_min_mm'] for t in pt)} guard {min(t['guard_min_mm'] for t in pt)} "
            f"motor {max(t['motor_max'] for t in pt)} sat {max(t['sat_longest_s'] for t in pt)} "
            f"tilt {max(t['tilt_max_deg'] for t in pt)} miss {a['miss_reasons']}")


def save(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=N._json_np)


def p2p_grid(variants, dists=(0.5, 1.0, 1.5, 2.0, 3.0), headings=(0.0, 45.0), seed=101):
    """variants: {name: (thrust, full overrides)}. Returns {name: [p2p rows]}."""
    cfgs, keys = [], []
    for name, (th, ov) in variants.items():
        for h in headings:
            for d in dists:
                hr = np.radians(h)
                cfgs.append(merge(setting(th, ov), {"seed": seed, "p2p": {"d": [d * np.cos(hr), d * np.sin(hr), 0.0]}}))
                keys.append((th, name, h, d))
    res = N.run_catch_batch(cfgs, JOBS, None, "p2p")
    out = {}
    for (th, name, h, d), m in zip(keys, res):
        row = {"thrust": th, "variant": name, "heading_deg": h, "distance_m": d}
        if "error" in m:
            row["error"] = m["error"]
        else:
            row.update(m["p2p"])
        out.setdefault(name, []).append(row)
    return out


def reach(variants):
    """Single-ball reach map per setting: {name: (thrust, full overrides)} -> {name: cells with trial records}."""
    N.CATCH_SNAPSHOT_DIR[0] = "off"
    cfgs, keys = [], []
    for name, (th, ov) in variants.items():
        base = {**N.CATCH_DEFAULT, **setting(th, ov)}
        h_dn = N.hold_drop() if base["h_dn"] is None else float(base["h_dn"])
        p = N.catch_params(base)
        cc, cells = N.reach_cfgs(base, h_dn, N.ball_k(p), base["z_net0"])
        for c in cc:
            i = c.pop("_cell")
            cfgs.append(c)
            keys.append((name, i, cells[i]))
    res = N.run_catch_batch(cfgs, JOBS, None, "reach")
    N.CATCH_SNAPSHOT_DIR[0] = None
    out = {}
    for (name, i, cell), m in zip(keys, res):
        r = {k: cell[k] for k in ("dir", "d_m", "T_s") if k in cell}
        if "error" in m:
            r["error"] = m["error"]
        else:
            b = m["balls"][0]
            r.update(caught=b["outcome"] == "caught", reason=b.get("reason"), rec=trial_rec(cell.get("seed"), m))
        out.setdefault(name, []).append(r)
    return out


def load_setting(path):
    """Settings JSON (the --catch-limits file) -> driver overrides (on top of setting(thrust)); the file's
    max_tilt_deg / p_over go to the controller parameters exactly as main_catch does."""
    d = json.load(open(path))
    d.pop("note", None)
    po = d.pop("p_over", None) or {}
    if "max_tilt_deg" in d:
        po["max_tilt_deg"] = d.pop("max_tilt_deg")
    if po:
        d["p_over"] = po
    return d


def write_setting(path, thrust, ov, note):
    """Driver overrides (full, on top of setting(thrust)) -> settings JSON for --catch-limits."""
    full = setting(thrust, ov)
    po = dict(full.pop("p_over", None) or {})
    po.pop("f_max", None)                       # comes from --thrust-scale 5
    js = {"note": note}
    if "max_tilt_deg" in po:
        js["max_tilt_deg"] = po.pop("max_tilt_deg")
    if po:
        js["p_over"] = po
    lim = dict(N.CATCH_LIMITS)
    lim.update(full.pop("limits", None) or {})
    js["limits"] = lim
    js.update(full)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(js, f, indent=1)
    return js
