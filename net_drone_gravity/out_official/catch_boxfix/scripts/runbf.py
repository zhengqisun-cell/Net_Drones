"""Batches for the boxfix study: python runbf.py <set> [seed_from seed_to]
set: pilot (10 balls, x5, boxfix on, seeds 101-115), main10 (10 balls, x5, before / after, seeds 1-30),
     std (3-5 balls, x5 tier 0 / tier 1 and x1 tier 0, four standard groups, before / after, seeds 1-30)."""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
OO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(OO, "catch_speed_relaxed", "scripts"))
sys.path.insert(0, os.path.join(OO, "catch_10balls", "scripts"))
import drv2
from drv2 import N, merge, setting, load_setting, trial_rec, tier_viol, save
from run10 import ball_recs

OUT = os.path.abspath(os.path.join(HERE, ".."))
G10 = {"med2": {"tier": "medium", "launch": "sequence", "interval_s": 2.0},
       "med1": {"tier": "medium", "launch": "sequence", "interval_s": 1.0},
       "hard2": {"tier": "hard", "launch": "sequence", "interval_s": 2.0},
       "hard1": {"tier": "hard", "launch": "sequence", "interval_s": 1.0}}
GSTD = {"med_seq2": {"tier": "medium", "launch": "sequence", "interval_s": 2.0},
        "med_seq1": {"tier": "medium", "launch": "sequence", "interval_s": 1.0},
        "hard_seq2": {"tier": "hard", "launch": "sequence", "interval_s": 2.0},
        "med_salvo": {"tier": "medium", "launch": "salvo", "interval_s": 2.0}}
S10 = {"x5_tier0_10balls": (5, 0, os.path.join(OO, "catch_10balls", "settings", "x5_tier0_10balls.json"),
                            os.path.join(OUT, "settings", "x5_tier0_10balls_boxfix.json")),
       "x5_tier1_10balls": (5, 1, os.path.join(OO, "catch_10balls", "settings", "x5_tier1_10balls.json"),
                            os.path.join(OUT, "settings", "x5_tier1_10balls_boxfix.json"))}
SSTD = {"x5_tier0": (5, 0, os.path.join(OO, "catch_speed", "settings", "catch_speed_x5.json"),
                     os.path.join(OUT, "settings", "x5_tier0_boxfix.json")),
        "x5_tier1": (5, 1, os.path.join(OO, "catch_speed_relaxed", "settings", "tier1_x5.json"),
                     os.path.join(OUT, "settings", "x5_tier1_boxfix.json")),
        "x1_tier0": (1, 0, os.path.join(OO, "catch_speed", "settings", "catch_speed_x1.json"),
                     os.path.join(OUT, "settings", "x1_tier0_boxfix.json"))}


def rec(s, m, tier):
    t = trial_rec(s, m)
    t["balls"] = ball_recs(m)
    t["net_lowest_m"] = m["safety"]["net_low"]["net_lowest_m"]
    t["box_min_dist_m"] = m["safety"]["box"]["min_dist_m"]
    t["reference"] = m["reference"]
    t["tier_viol"] = tier_viol(t, tier)
    return t


def run(tag, S, groups, seeds, variants):
    cfgs, keys = [], []
    for n, (th, tier, p_before, p_after) in S.items():
        for var in variants:
            ov = load_setting(p_before if var == "before" else p_after)
            for g, gc in groups.items():
                for s in seeds:
                    cfgs.append(merge(setting(th, ov), {**gc, "seed": s, "label": f"{n}_{var}_{g}_seed{s}"}))
                    keys.append((n, var, g, s, tier))
    res = N.run_catch_batch(cfgs, drv2.JOBS, None, tag)
    out = {}
    for (n, var, g, s, tier), m in zip(keys, res):
        d = out.setdefault(n, {}).setdefault(var, {}).setdefault(g, [])
        d.append({"seed": s, "error": m["error"]} if "error" in m else rec(s, m, tier))
    save({"seeds": seeds, "groups": groups, "settings": {n: {"before": v[2], "after": v[3], "thrust": v[0], "tier": v[1]}
                                                       for n, v in S.items()}, "results": out},
         os.path.join(OUT, "runs", f"{tag}.json"))
    for n, vv in out.items():
        for var, gr in vv.items():
            for g, ts in gr.items():
                ok = [t for t in ts if "error" not in t]
                print(f"{n:18s} {var:6s} {g:9s} caught {sum(t['n_caught'] for t in ok)}/{sum(t['n_launched'] for t in ok)} "
                      f"viol {sum(bool(t['tier_viol']) for t in ok)} box {sum(not t['hard_ok']['box'] for t in ok)} "
                      f"net_low {sum(not t['hard_ok']['net_low'] for t in ok)} aborts {sum(t['aborted'] for t in ok)} "
                      f"ref_out {max(t['reference']['out_of_box_max_m'] for t in ok)} "
                      f"below {max(t['reference']['below_floor_max_m'] for t in ok)} "
                      f"brakes {sum(t['reference']['n_brake'] for t in ok)} errors {len(ts) - len(ok)}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "pilot":
        run("pilot_101-115", S10, G10, list(range(101, 116)), ["after"])
    elif which in ("main10", "std"):
        S, G, tag = (S10, G10, "main10_1-30") if which == "main10" else (SSTD, GSTD, "std_1-30")
        variants = sys.argv[2].split(",") if len(sys.argv) > 2 else ["before", "after"]
        prev = os.path.join(OUT, "runs", f"{tag}.json")
        old = json.load(open(prev)) if os.path.exists(prev) and variants != ["before", "after"] else None
        run(tag, S, G, list(range(1, 31)), variants)
        if old is not None:                 # merge: keep the variants not rerun (the fix-off runs are deterministic)
            new = json.load(open(prev))
            for n, vv in old["results"].items():
                for var, gr in vv.items():
                    if var not in variants:
                        new["results"][n][var] = gr
            save(new, prev)
