"""Before / after tables of the boxfix study: python compbf.py"""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
OO = os.path.abspath(os.path.join(OUT, ".."))
STORED = {("main10", "x5_tier0_10balls"): (os.path.join(OO, "catch_10balls", "runs", "main_1-30.json"), "x5_tier0_10balls", None),
          ("main10", "x5_tier1_10balls"): (os.path.join(OO, "catch_10balls", "runs", "main_1-30.json"), "x5_tier1_10balls", None),
          ("std", "x5_tier0"): (os.path.join(OO, "catch_speed", "final", "final2_30seeds.json"), "x5_scurve_j20_diag0.9", "results"),
          ("std", "x5_tier1"): (os.path.join(OO, "catch_speed_relaxed", "final", "final_seeds1-30.json"), "T1_x5", "results"),
          ("std", "x1_tier0"): (os.path.join(OO, "catch_speed", "final", "final3_30seeds.json"), "x1_scurve_j30_av1.6", "results")}


def sig(t):
    """What counts as a changed trial: catches, per-ball outcomes, safety items, clearances, motor and tilt peaks."""
    return (t["n_caught"], tuple(t["reasons"]), tuple(t["tier_viol"]), t["disc_min_mm"], t["guard_min_mm"],
            t["motor_max"], t["tilt_max_deg"], t["aborted"])


def stats(ts):
    ok = [t for t in ts if "error" not in t]
    r = {"trials": len(ok), "caught": sum(t["n_caught"] for t in ok), "launched": sum(t["n_launched"] for t in ok),
         "all_caught": sum(t["n_caught"] == t["n_launched"] for t in ok),
         "ge8": sum(t["n_caught"] >= 8 for t in ok),
         "viol": sum(bool(t["tier_viol"]) for t in ok), "box": sum(not t["hard_ok"]["box"] for t in ok),
         "net_low": sum(not t["hard_ok"]["net_low"] for t in ok), "aborts": sum(t["aborted"] for t in ok),
         "box_min_dist_m": min(t["box_min_dist_m"] for t in ok), "net_lowest_m": min(t["net_lowest_m"] for t in ok),
         "ref_out_of_box_max_m": max(t["reference"]["out_of_box_max_m"] for t in ok),
         "ref_below_floor_max_m": max(t["reference"]["below_floor_max_m"] for t in ok),
         "brake_calls": sum(t["reference"]["n_brake"] for t in ok),
         "brake_latched": sum(t["reference"]["n_brake_latched"] for t in ok),
         "brake_kept_plan": sum(t["reference"]["n_brake_kept_plan"] for t in ok),
         "floor_kept_plan": sum(t["reference"].get("n_floor_kept_plan", 0) for t in ok),
         "trials_with_brake": sum(t["reference"]["n_brake"] + t["reference"]["n_brake_kept_plan"]
                                 + t["reference"].get("n_floor_kept_plan", 0) > 0 for t in ok),
         "errors": [t["seed"] for t in ts if "error" in t]}
    return r


def main():
    out, L = {}, []
    for tag, title in (("main10_1-30", "10 balls, thrust x5, seeds 1-30"), ("std_1-30", "3-5 balls, seeds 1-30 (impact check)")):
        d = json.load(open(os.path.join(OUT, "runs", f"{tag}.json")))["results"]
        L += [f"== {title}", "setting           group      variant  caught    all  >=8 viol box net_low abort  box_min  "
              "net_low_m ref_out ref_below fallbacks(brake/latched/kept/floor-kept, trials)  changed trials"]
        for n, vv in d.items():
            for g in vv["before"]:
                B, A = vv["before"][g], vv["after"][g]
                sb, sa = stats(B), stats(A)
                Bm = {t["seed"]: t for t in B if "error" not in t}
                Am = {t["seed"]: t for t in A if "error" not in t}
                ch = [s for s in Bm if s in Am and sig(Bm[s]) != sig(Am[s])]
                chc = [(s, Bm[s]["n_caught"], Am[s]["n_caught"]) for s in ch if Bm[s]["n_caught"] != Am[s]["n_caught"]]
                nobrake = [s for s in Bm if s in Am and Bm[s]["reference"]["n_brake"] == 0
                           and Am[s]["reference"]["n_brake"] + Am[s]["reference"]["n_brake_kept_plan"]
                           + Am[s]["reference"].get("n_floor_kept_plan", 0) == 0]
                nb_changed = [s for s in nobrake if sig(Bm[s]) != sig(Am[s])]
                # the rerun "before" against the stored results of the earlier round (same code path)
                st = STORED.get((tag.split("_")[0], n))
                stored_diff = None
                if st and os.path.exists(st[0]):
                    sj = json.load(open(st[0]))["results"][st[1]]
                    gk = g
                    ts = sj[gk] if isinstance(sj[gk], list) else sj[gk]["per_trial"]
                    sm = {t["seed"]: t["n_caught"] for t in ts if "error" not in t}
                    stored_diff = [s for s in Bm if s in sm and sm[s] != Bm[s]["n_caught"]]
                out.setdefault(tag, {}).setdefault(n, {})[g] = {
                    "before": sb, "after": sa, "changed_trials": ch, "changed_caught": chc,
                    "trials_without_brake_in_both": len(nobrake), "of_which_changed": nb_changed,
                    "before_rerun_vs_stored_caught_diff_seeds": stored_diff}
                for var, s_ in (("before", sb), ("after", sa)):
                    L.append(f"{n:17s} {g:10s} {var:7s} {s_['caught']:>4d}/{s_['launched']:<4d} {s_['all_caught']:>3d} "
                             f"{s_['ge8']:>4d} {s_['viol']:>4d} {s_['box']:>3d} {s_['net_low']:>7d} {s_['aborts']:>5d} "
                             f"{s_['box_min_dist_m']:>8.3f} {s_['net_lowest_m']:>9.3f} {s_['ref_out_of_box_max_m']:>7.3f} "
                             f"{s_['ref_below_floor_max_m']:>9.4f} {s_['brake_calls']:>5d}/{s_['brake_latched']}/"
                             f"{s_['brake_kept_plan']}/{s_['floor_kept_plan']}, {s_['trials_with_brake']}"
                             + (f"   {len(ch)} changed, caught changed in {len(chc)}: {chc}; trials without any brake in "
                                f"both runs {len(nobrake)}, of them changed {len(nb_changed)}; rerun vs stored caught "
                                f"differs in {stored_diff}" if var == "after" else ""))
            L.append("")
    legend = ("caught/launched; all = trials with every launched ball caught; >=8 = trials with >= 8 caught (10 balls); "
              "viol = trials breaking the setting's own tier lines (tier 0: motor 90 % > 0.10 s, disc 30 mm; tier 1: 95 %, 25 mm; "
              "ball-to-disc 30 mm, guard 10 mm, abort, net low 0.30 m, separation, slack, box in both); box / net_low = trials "
              "with that safety item; box_min = smallest drone distance to the box wall (negative: outside); ref_out = "
              "largest distance of the planner's formation-centre reference outside the formation box; ref_below = largest "
              "depth of the reference below the height floor (net centre 0.7 m below the drones with the balls in the net, "
              "or the current / last planned height if lower); fallbacks = calls of the braking fallback / of them latched "
              "(boxfix) / braking paths that would have left the box, current plan kept instead (boxfix) / new paths that "
              "would have dipped more than 0.10 m below the floor, current plan kept until the next replan (boxfix), and "
              "trials with any of these. changed = trials whose catches, per-ball outcomes, safety items, clearances, motor or tilt peak "
              "differ. 'before' is a rerun with the fix off (identical to the stored earlier results where those exist).")
    open(os.path.join(OUT, "compare_boxfix.txt"), "w").write("\n".join(L) + "\n" + legend + "\n")
    json.dump({"table": out, "legend": legend}, open(os.path.join(OUT, "compare_boxfix.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
