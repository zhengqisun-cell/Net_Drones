"""Robustness table (D): hard tier 2 s, seeds 1-30, baseline and the selected setting per family: nominal (from
final_1-30), net mass +20 % (robust_mnet), strings +-10 mm (robust_strings); the planner's model unchanged.
python robust_table.py -> robust_table.txt / .json"""
import os, json
from compile_nd import gstats, OUT
from make_settings import SELECTED

G = {"x1_T0": "hard_seq2", "x5_T0": "hard_seq2", "x5_T1": "hard_seq2", "x5_T0_10": "hard2", "x5_T1_10": "hard2"}
RUNS = [("nominal", "final_1-30"), ("net mass +20 %", "robust_mnet"), ("strings +-10 mm", "robust_strings")]


def main():
    res, L = {}, ["hard tier, 2 s interval, seeds 1-30; caught/launched, tier-violation trials (items), aborts, "
                  "smallest disc / guard clearance (mm), largest tilt (deg); the net model in the planner is not changed"]
    data = {tag: json.load(open(os.path.join(OUT, "runs", f"{tag}.json")))["results"] for _, tag in RUNS}
    for fam, g in G.items():
        L.append(f"== {fam} ({g})")
        for cand in ("base", SELECTED[fam]):
            for name, tag in RUNS:
                ts = data[tag].get(fam, {}).get(cand, {}).get(g)
                if ts is None:
                    continue
                s = gstats(ts)
                res.setdefault(fam, {}).setdefault(cand, {})[name] = s
                L.append(f"  {cand:7s} {name:16s} {s['caught']:4d}/{s['launched']:<4d} viol {s['viol']:2d}/30 "
                         f"{s['items']} abort {s['aborts']} disc {s['disc_min_mm']} guard {s['guard_min_mm']} "
                         f"tilt {s['tilt_max_deg']}")
    open(os.path.join(OUT, "robust_table.txt"), "w").write("\n".join(L) + "\n")
    json.dump(res, open(os.path.join(OUT, "robust_table.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
