"""10-ball catch tests at thrust x5: python run10.py <tag> <seed_from> <seed_to> [settings ...] [--groups g1,g2]
settings: names of files in ../settings without .json (default x5_tier0_10balls x5_tier1_10balls).
Groups: med2 med1 hard2 hard1 (medium / hard tier, sequence, 2 s / 1 s interval)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "catch_speed_relaxed", "scripts"))
import drv2
from drv2 import N, merge, setting, load_setting, trial_rec, tier_viol, save

OUT = os.path.abspath(os.path.join(HERE, ".."))
GROUPS = {"med2": {"tier": "medium", "launch": "sequence", "interval_s": 2.0},
          "med1": {"tier": "medium", "launch": "sequence", "interval_s": 1.0},
          "hard2": {"tier": "hard", "launch": "sequence", "interval_s": 2.0},
          "hard1": {"tier": "hard", "launch": "sequence", "interval_s": 1.0}}
TIER_OF = {"x5_tier0_10balls": 0, "x5_tier1_10balls": 1}


def in_net_at(r, t):
    """Is ball record r (another ball) in the net at time t (from its first contact / pile support to the end, or to
    the time it left the net)?"""
    t0 = r.get("t_first_contact")
    if r.get("caught_on_pile"):
        t0 = min(x for x in (t0, r["caught_on_pile"]["t_first_support"]) if x is not None)
    if t0 is None or t0 > t:
        return False
    if r["outcome"] == "caught":
        return True
    return r.get("loss") is not None and r["loss"]["t_out_s"] > t


def ball_recs(m):
    bl = m["balls"]
    out = []
    for r in bl:
        q = {"ball": r["ball"] + 1, "outcome": r["outcome"], "reason": r.get("reason"),
             "loss": (r.get("loss") or {}).get("kind"), "on_pile": bool(r.get("caught_on_pile")),
             "t_launch": r.get("t_launch"), "gap_disc_mm": r.get("gap_disc_mm"), "shell_hits": r.get("shell_hits")}
        if r["outcome"] != "not_launched" and r.get("t_standby_cross_s") is not None:
            ta = r["t_launch"] + r["t_standby_cross_s"]          # arrival: crossing of the standby aim height
            q["t_arrive"] = round(ta, 3)
            q["in_net_at_arrival"] = int(sum(in_net_at(o, ta) for o in bl if o is not r and o["outcome"] != "not_launched"))
        if r.get("loss"):
            q["loss_detail"] = r["loss"]
        out.append(q)
    return out


def run(tag, seeds, names, groups):
    S = {n: load_setting(os.path.join(OUT, "settings", f"{n}.json")) for n in names}
    cfgs, keys = [], []
    for n, ov in S.items():
        for g in groups:
            for s in seeds:
                cfgs.append(merge(setting(5, ov), {**GROUPS[g], "seed": s, "label": f"{n}_{g}_seed{s}"}))
                keys.append((n, g, s))
    res = N.run_catch_batch(cfgs, drv2.JOBS, None, tag)
    out = {}
    for (n, g, s), m in zip(keys, res):
        d = out.setdefault(n, {}).setdefault(g, [])
        if "error" in m:
            d.append({"seed": s, "error": m["error"]})
            continue
        t = trial_rec(s, m)
        t["balls"] = ball_recs(m)
        t["net_lowest_m"] = m["safety"]["net_low"]["net_lowest_m"]
        t["tier_viol"] = tier_viol(t, TIER_OF.get(n, 0))
        d.append(t)
    save({"seeds": seeds, "settings": S, "groups": {g: GROUPS[g] for g in groups}, "results": out},
         os.path.join(OUT, "runs", f"{tag}.json"))
    for n, gr in out.items():
        for g, ts in gr.items():
            ok = [t for t in ts if "error" not in t]
            nc = sum(t["n_caught"] for t in ok)
            nl = sum(t["n_launched"] for t in ok)
            print(f"{n:18s} {g:6s} caught {nc}/{nl}  all10 {sum(t['n_caught'] == 10 for t in ok)}  >=8 "
                  f"{sum(t['n_caught'] >= 8 for t in ok)}  viol {sum(bool(t['tier_viol']) for t in ok)}  aborts "
                  f"{sum(t['aborted'] for t in ok)}  errors {len(ts) - len(ok)}  disc {min(t['disc_min_mm'] for t in ok)}  "
                  f"tilt {max(t['tilt_max_deg'] for t in ok)}  netlow {min(t['net_lowest_m'] for t in ok)}", flush=True)
    return out


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    groups = ["med2", "med1", "hard2", "hard1"]
    for x in sys.argv[1:]:
        if x.startswith("--groups="):
            groups = x.split("=", 1)[1].split(",")
    tag, s0, s1 = a[0], int(a[1]), int(a[2])
    names = a[3:] or ["x5_tier0_10balls", "x5_tier1_10balls"]
    run(tag, list(range(s0, s1 + 1)), names, groups)
