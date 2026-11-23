"""Tables and charts of the 10-ball test: python compile10.py [run tag] (default main_1-30)."""
import sys, os, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
ROOT = os.path.abspath(os.path.join(OUT, ".."))
GN = {"med2": "medium 2 s", "med1": "medium 1 s", "hard2": "hard 2 s", "hard1": "hard 1 s"}
SN = {"x5_tier0_10balls": "tier 0 (existing lines)", "x5_tier1_10balls": "tier 1 (95 % / 25 mm)"}
# the 3-5-ball finals (seeds 1-30) of the same settings, for the first-balls comparison
REF = {"x5_tier0_10balls": (os.path.join(ROOT, "catch_speed", "final", "final2_30seeds.json"), "x5_scurve_j20_diag0.9"),
       "x5_tier1_10balls": (os.path.join(ROOT, "catch_speed_relaxed", "final", "final_seeds1-30.json"), "T1_x5")}
REFG = {"med2": "med_seq2", "med1": "med_seq1", "hard2": "hard_seq2"}


def miss_kind(b):
    if b["reason"] == "fell_out" and b.get("loss"):
        return b["loss"]
    return b["reason"]


def group_stats(ts):
    ok = [t for t in ts if "error" not in t]
    balls = [b for t in ok for b in t["balls"]]
    launched = [b for b in balls if b["outcome"] != "not_launched"]
    nl, nc = len(launched), sum(b["outcome"] == "caught" for b in launched)
    hist = collections.Counter(t["n_caught"] for t in ok)
    order = {}
    for b in launched:
        order.setdefault(b["ball"], []).append(b["outcome"] == "caught")
    innet = {}
    for b in launched:
        if b.get("in_net_at_arrival") is not None:
            innet.setdefault(b["in_net_at_arrival"], []).append(b["outcome"] == "caught")
    miss = collections.Counter(miss_kind(b) for b in launched if b["outcome"] != "caught")
    viol = collections.Counter(v for t in ok for v in t["tier_viol"])
    return {"trials": len(ok), "errors": [t["seed"] for t in ts if "error" in t],
            "caught": nc, "launched": nl, "share": round(nc / max(nl, 1), 4),
            "not_launched": [{"seed": t["seed"], "ball": b["ball"]} for t in ok for b in t["balls"]
                             if b["outcome"] == "not_launched"],
            "all10_trials": int(sum(t["n_caught"] == 10 for t in ok)),
            "ge8_trials": int(sum(t["n_caught"] >= 8 for t in ok)),
            "caught_per_trial_hist": {str(k): hist.get(k, 0) for k in range(11)},
            "caught_per_trial_median": float(np.median([t["n_caught"] for t in ok])),
            "by_launch_order": {str(k): [int(sum(v)), len(v)] for k, v in sorted(order.items())},
            "by_balls_in_net_at_arrival": {str(k): [int(sum(v)), len(v)] for k, v in sorted(innet.items())},
            "miss_kinds": dict(miss),
            "caught_on_pile": int(sum(b["on_pile"] for b in launched if b["outcome"] == "caught")),
            "lost_after_held": {k: int(sum(b.get("loss") == k for b in launched))
                                for k in ("knocked_out", "thrown_out", "bounced_out")},
            "tier_viol_trials": int(sum(bool(t["tier_viol"]) for t in ok)), "tier_viol_items": dict(viol),
            "disc_min_mm": min(t["disc_min_mm"] for t in ok), "guard_min_mm": min(t["guard_min_mm"] for t in ok),
            "motor_peak": max(t["motor_max"] for t in ok), "tilt_max_deg": max(t["tilt_max_deg"] for t in ok),
            "aborts": int(sum(t["aborted"] for t in ok)), "net_lowest_m": min(t["net_lowest_m"] for t in ok),
            "ball_hit_drone_trials": int(sum(t["shell_hits"] > 0 for t in ok))}


def first_balls_compare(name, g, ts):
    """Caught among each seed's first n balls (n = that seed's default 3-5 count) in the 10-ball run vs the 3-5-ball
    run of the same setting and seed (same elevation, azimuth, speed and launch times; barrel row differs)."""
    path, key = REF[name]
    if g not in REFG or not os.path.exists(path):
        return None
    ref = {t["seed"]: t for t in json.load(open(path))["results"][key][REFG[g]]["per_trial"]}
    a = b = n = 0
    for t in ts:
        if "error" in t or t["seed"] not in ref:
            continue
        r = ref[t["seed"]]
        k = r["n_balls"]
        a += sum(x["outcome"] == "caught" for x in t["balls"][:k])
        b += r["n_caught"]
        n += r["n_launched"]
    return {"first_n_balls": n, "caught_in_10ball_run": a, "caught_in_3to5ball_run": b}


def main(tag="main_1-30"):
    d = json.load(open(os.path.join(OUT, "runs", f"{tag}.json")))["results"]
    tab = {}
    L = [f"10-ball test, thrust x5, seeds {tag.split('_')[-1]} (10 balls per trial). Tier lines: each setting "
         f"its own tier (motor line exceeded > 0.10 s continuously; ball-to-disc gap 30 mm in both; guard 10 mm).", ""]
    for n, gr in d.items():
        tab[n] = {}
        for g in GN:
            if g not in gr:
                continue
            s = group_stats(gr[g])
            s["first_balls_vs_3to5"] = first_balls_compare(n, g, gr[g])
            tab[n][g] = s
            L.append(f"{SN.get(n, n)}, {GN[g]}: caught {s['caught']}/{s['launched']} ({100 * s['share']:.1f} %); all 10 "
                     f"{s['all10_trials']}/{s['trials']}; >= 8 {s['ge8_trials']}/{s['trials']}; median per trial {s['caught_per_trial_median']:g}")
            L.append("   caught per trial (0..10): " + " ".join(f"{s['caught_per_trial_hist'][str(k)]}" for k in range(11)))
            L.append("   by launch order: " + "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in s["by_launch_order"].items()))
            L.append("   by balls in the net at arrival: " + "  ".join(f"{k}:{v[0]}/{v[1]}" for k, v in s["by_balls_in_net_at_arrival"].items()))
            L.append(f"   misses: {s['miss_kinds']}; lost after being held: {s['lost_after_held']}; caught on the pile: "
                     f"{s['caught_on_pile']}; not launched: {s['not_launched'] or 'none'}")
            L.append(f"   safety: tier-violation trials {s['tier_viol_trials']}/{s['trials']} {s['tier_viol_items']}; disc min "
                     f"{s['disc_min_mm']} mm, guard min {s['guard_min_mm']} mm, motor peak {s['motor_peak']}, tilt max "
                     f"{s['tilt_max_deg']} deg, aborts {s['aborts']}, net lowest {s['net_lowest_m']} m, ball hit drone "
                     f"{s['ball_hit_drone_trials']} trials")
            fb = s["first_balls_vs_3to5"]
            if fb:
                L.append(f"   first 3-5 balls of each seed: caught {fb['caught_in_10ball_run']}/{fb['first_n_balls']} here vs "
                         f"{fb['caught_in_3to5ball_run']}/{fb['first_n_balls']} in the 3-5-ball run (same seeds, same setting)")
            L.append("")
    legend = ("Denominator = launched balls (none excluded unless listed under 'not launched'). Miss kinds: late (not there "
              "in time), offset (there but off target), crowded_out (dropped for another ball), hit_drone, knocked_out "
              "(held in the net, then left it within 0.6 s of another ball's arrival), thrown_out (held, left it with no "
              "arrival then: the formation's motion), bounced_out (touched the net but never held), trial_aborted. "
              "Arrival = crossing of the standby aim height; 'balls in the net at arrival' counts other balls in the net "
              "(caught: from first contact / pile support; lost: until they left).")
    open(os.path.join(OUT, f"summary_{tag}.txt"), "w").write("\n".join(L) + "\n" + legend + "\n")
    json.dump({"table": tab, "legend": legend}, open(os.path.join(OUT, f"summary_{tag}.json"), "w"), indent=1)
    print("\n".join(L))
    # charts: catch share by launch order (top) and by balls in the net at arrival (bottom); 2 s left, 1 s right
    fig, ax = plt.subplots(2, 2, figsize=(14, 9))
    style = {("x5_tier0_10balls", "med"): ("#2a6fdb", "-", "o"), ("x5_tier1_10balls", "med"): ("#2a6fdb", "--", "s"),
             ("x5_tier0_10balls", "hard"): ("#d9622b", "-", "o"), ("x5_tier1_10balls", "hard"): ("#d9622b", "--", "s")}
    for j, iv in enumerate(("2", "1")):
        for row, key, xl in ((0, "by_launch_order", "launch order (ball number)"),
                             (1, "by_balls_in_net_at_arrival", "balls already in the net when the ball arrives")):
            a = ax[row, j]
            for n in tab:
                for tier in ("med", "hard"):
                    g = f"{tier}{iv}"
                    if g not in tab[n]:
                        continue
                    dd = tab[n][g][key]
                    xs = [int(k) for k in dd if dd[k][1] >= (1 if row == 0 else 5)]
                    ys = [100 * dd[str(k)][0] / dd[str(k)][1] for k in xs]
                    c, ls, mk = style[(n, tier)]
                    a.plot(xs, ys, ls, marker=mk, color=c, lw=2 if ls == "-" else 1.6, ms=6,
                           label=f"{'medium' if tier == 'med' else 'hard'}, {SN[n]}")
            a.set_xlabel(xl)
            a.set_ylabel("caught [% of launched]")
            a.set_ylim(0, 105)
            a.grid(color="#dddddd", lw=0.8)
            a.spines[["top", "right"]].set_visible(False)
            a.set_title(f"{iv} s interval" + (" (points with >= 5 balls)" if row == 1 else ""), fontsize=10)
            if row == 0:
                a.set_xticks(range(1, 11))
    ax[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("10 balls in sequence, thrust x5, seeds 1-30 (30 trials per line): catch share by launch order and by "
                 "balls already in the net", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"catch_by_order_{tag}.png"), dpi=130)


if __name__ == "__main__":
    main(*sys.argv[1:])
