"""Tables and chart for out_official/catch_speed_relaxed (main table, robustness, p2p, reach)."""
import json, os, sys
import numpy as np
from drv2 import OUT, TIERS, tier_viol, MOTOR_TIME_S
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GN = {"med_seq2": "medium 2 s", "med_seq1": "medium 1 s", "hard_seq2": "hard 2 s", "med_salvo": "medium salvo"}


def ld(*p):
    return json.load(open(os.path.join(OUT, *p)))


def line_key(k):
    m = TIERS[k]["motor"]
    return None if m is None else f"{m:.2f}"


def group_row(a, k):
    pt = a["per_trial"]
    lk = line_key(k)
    vd = [tier_viol(t, k, "dur") for t in pt]
    vp = [tier_viol(t, k, "peak") for t in pt] if lk else None
    vb = [tier_viol(t, k, "dur", "relaxed") for t in pt]
    items = {}
    for v in vd:
        for x in v:
            items[x] = items.get(x, 0) + 1
    r = {"caught": a["caught"], "launched": a["balls"], "trials": a["trials"],
         "excluded_not_launched": len(a["excluded_not_launched"]),
         "all_caught_trials": a["n_all_caught"],
         "all_caught_within_tier_trials": int(sum(t["all_caught"] and not v for t, v in zip(pt, vd))),
         "tier_viol_trials": int(sum(bool(v) for v in vd)), "tier_viol_items": items,
         "tier_viol_trials_peak_rule": None if vp is None else int(sum(bool(v) for v in vp)),
         "tier_viol_trials_ball_gap_relaxed": int(sum(bool(v) for v in vb)),
         "ball_disc_gap_min_mm": min((t["ball_disc_gap_min_mm"] for t in pt if t["ball_disc_gap_min_mm"] is not None),
                                     default=None),
         "disc_min_mm": min(t["disc_min_mm"] for t in pt), "guard_min_mm": min(t["guard_min_mm"] for t in pt),
         "motor_peak": max(t["motor_max"] for t in pt),
         "trials_motor_peak_above_line": None if lk is None else int(sum(t["motor_max"] > TIERS[k]["motor"] for t in pt)),
         "above_line_total_s_max": None if lk is None else max(t["motor_lines"][lk]["total_s"] for t in pt),
         "above_line_longest_s_max": None if lk is None else max(t["motor_lines"][lk]["longest_s"] for t in pt),
         "above_90_longest_s_max": max(t["motor_lines"]["0.90"]["longest_s"] for t in pt),
         "saturation_longest_s_max": max(t["sat_longest_s"] for t in pt),
         "saturation_total_s_max": max(t["sat_total_s"] for t in pt),
         "trials_with_saturation": int(sum(t["sat_total_s"] > 0 for t in pt)),
         "tilt_max_deg": max(t["tilt_max_deg"] for t in pt),
         "aborts": int(sum(t["aborted"] for t in pt)),
         "balls_fell_out": int(sum(t["fell_out"] for t in pt)),
         "ball_hit_drone_trials": int(sum(t["shell_hits"] > 0 for t in pt)),
         "miss_reasons": a["miss_reasons"]}
    return r


def main(tag="final_seeds1-30", out_tag=""):
    fin = ld("final", f"{tag}.json")["results"]
    tab = {}
    L = []
    hdr = (f"{'tier':4s} {'thr':3s} {'group':13s} {'caught':>9s} {'all':>5s} {'allOK':>5s} {'viol':>4s} {'vPk':>4s} {'vBr':>4s} "
           f"{'disc':>5s} {'guard':>5s} {'mPeak':>6s} {'>line':>5s} {'tot_s':>5s} {'long_s':>6s} {'sat_s':>5s} "
           f"{'tilt':>5s} {'abrt':>4s} {'fell':>4s} {'hit':>3s}  items")
    L.append(hdr)
    for k in TIERS:
        for th in (1, 5):
            sn = f"T{k}_x{th}"
            tab[sn] = {}
            tot = {"caught": 0, "launched": 0}
            for g in GN:
                r = group_row(fin[sn][g], k)
                tab[sn][g] = r
                tot["caught"] += r["caught"]
                tot["launched"] += r["launched"]
                f = lambda x: "-" if x is None else x
                L.append(f"{k:<4d} x{th:<2d} {GN[g]:13s} {r['caught']:>4d}/{r['launched']:<4d} {r['all_caught_trials']:>2d}/30 "
                         f"{r['all_caught_within_tier_trials']:>5d} {r['tier_viol_trials']:>4d} {f(r['tier_viol_trials_peak_rule']):>4} {r['tier_viol_trials_ball_gap_relaxed']:>4d} "
                         f"{r['disc_min_mm']:>5.1f} {r['guard_min_mm']:>5.1f} {r['motor_peak']:>6.3f} "
                         f"{f(r['trials_motor_peak_above_line']):>5} {f(r['above_line_total_s_max']):>5} "
                         f"{f(r['above_line_longest_s_max']):>6} {r['saturation_longest_s_max']:>5} {r['tilt_max_deg']:>5.1f} "
                         f"{r['aborts']:>4d} {r['balls_fell_out']:>4d} {r['ball_hit_drone_trials']:>3d}  {r['tier_viol_items']}")
            vt = sum(tab[sn][g]["tier_viol_trials"] for g in GN)
            vp = [tab[sn][g]["tier_viol_trials_peak_rule"] for g in GN]
            vb = sum(tab[sn][g]["tier_viol_trials_ball_gap_relaxed"] for g in GN)
            tab[sn]["total"] = {**tot, "tier_viol_trials": vt, "trials": 120,
                                "tier_viol_trials_peak_rule": None if vp[0] is None else sum(vp),
                                "tier_viol_trials_ball_gap_relaxed": vb}
            L.append(f"{k:<4d} x{th:<2d} {'TOTAL':13s} {tot['caught']:>4d}/{tot['launched']:<4d} "
                     f"({100 * tot['caught'] / tot['launched']:.1f} %)  tier-violation trials {vt}/120"
                     + ("" if vp[0] is None else f" (peak rule {sum(vp)}/120)") + f"; ball gap relaxed: {vb}/120")
            L.append("")
    ex = [f"{sn} {GN[g]} seed {e['seed']} ball {e['ball']}" for sn in fin for g in GN
          for e in fin[sn][g]["excluded_not_launched"]]
    EXCL = "; ".join(ex) if ex else "none"
    legend = ("columns: caught/launched balls; all = trials with every launched ball caught; allOK = of those, trials "
              "within the tier's lines; viol = trials breaking the tier's lines (motor: line exceeded > 0.10 s "
              "continuously); vPk = same with the motor line judged on the peak; vBr = same as viol "
              "but the ball-to-rotor-disc gap judged on the tier's disc line (viol keeps the existing 30 mm in every tier: "
              "the sim has no rotor downwash); disc/guard = min clearance [mm]; mPeak = "
              "motor peak (share of f_max); >line = trials with a peak above the tier's motor line; tot_s / long_s = max "
              "over trials of the total / longest continuous time above the tier's motor line; sat_s = longest motor "
              "command saturation (command >= f_max); tilt [deg]; abrt = aborted trials; fell = balls thrown out; "
              "hit = trials with a ball hitting a drone. Denominator = launched balls; balls not launched because the trial "
              "aborted before their launch are excluded: " + EXCL + ".")
    open(os.path.join(OUT, f"main_table{out_tag}.txt"), "w").write("\n".join(L) + "\n" + legend + "\n")
    json.dump({"tiers": TIERS, "motor_time_s": MOTOR_TIME_S, "guard_mm": 10.0, "seeds": list(range(1, 31)),
               "table": tab, "legend": legend}, open(os.path.join(OUT, f"main_table{out_tag}.json"), "w"), indent=1)
    print("\n".join(L))
    # chart: tier -> caught share; x1 / x5; hard 2 s, medium 1 s, all four groups; dashed: fresh seeds 201-230
    val = None
    if not out_tag and os.path.exists(os.path.join(OUT, "final", "final_seeds201-230.json")):
        fv = ld("final", "final_seeds201-230.json")["results"]
        val = {sn: {g: group_row(fv[sn][g], int(sn[1])) for g in GN} for sn in fv}

    def share(T, sn, g):
        if g == "all":
            c = sum(T[sn][x]["caught"] for x in GN)
            b = sum(T[sn][x]["launched"] for x in GN)
            v = sum(T[sn][x]["tier_viol_trials"] for x in GN)
            return 100 * c / b, v, 120
        return 100 * T[sn][g]["caught"] / T[sn][g]["launched"], T[sn][g]["tier_viol_trials"], 30

    fig, ax = plt.subplots(1, 3, figsize=(16, 5.0))
    cols = {1: "#2a6fdb", 5: "#d9622b"}
    panels = (("hard_seq2", "hard tier, 2 s interval"), ("med_seq1", "medium tier, 1 s interval"),
              ("all", "all four groups (516 balls)"))
    for j, (g, ttl) in enumerate(panels):
        a = ax[j]
        ys = {th: [share(tab, f"T{k}_x{th}", g) for k in TIERS] for th in (1, 5)}
        for th in (1, 5):
            y = [x[0] for x in ys[th]]
            a.plot(list(TIERS), y, "-o", color=cols[th], lw=2, ms=7, label=f"thrust x{th}, seeds 1-30")
            if val:
                yv = [share(val, f"T{k}_x{th}", g)[0] for k in TIERS]
                a.plot(list(TIERS), yv, "--", marker="o", mfc="white", color=cols[th], lw=1.2, ms=6,
                       label=f"thrust x{th}, fresh seeds 201-230")
            other = 5 if th == 1 else 1
            for k, (yy, vv, nn) in zip(TIERS, ys[th]):
                up = yy >= ys[other][k][0]
                a.annotate(f"{yy:.0f} %, {vv}/{nn}", (k, yy), textcoords="offset points",
                           xytext=(0, 8 if up else -16), ha="center", fontsize=8, color=cols[th])
        a.set_xticks(list(TIERS))
        a.set_xticklabels([f"tier {k}\nmotor {('%d %%' % round(100 * TIERS[k]['motor'])) if TIERS[k]['motor'] else 'any'}"
                           f"\ndisc {TIERS[k]['disc_mm']:g} mm" for k in TIERS], fontsize=8)
        a.set_xlim(-0.4, 3.4)
        a.set_ylabel("balls caught [% of launched]")
        a.set_title(ttl, fontsize=10)
        a.grid(axis="y", color="#dddddd", lw=0.8)
        a.spines[["top", "right"]].set_visible(False)
        lo = min(l.get_ydata().min() for l in a.get_lines())
        hi = max(l.get_ydata().max() for l in a.get_lines())
        a.set_ylim(lo - 6, hi + 6)
    ax[2].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Safety-line tier -> balls caught (labels: caught share, trials breaking that tier's lines / trials; "
                 "solid: seeds 1-30)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"tiers_caught{out_tag}.png"), dpi=130)
    if out_tag:
        return
    # robustness
    if os.path.exists(os.path.join(OUT, "robust", "robust_hard_seq2.json")):
        rb = ld("robust", "robust_hard_seq2.json")["results"]
        R = ["perturbation       setting   caught    tier-viol  disc   guard  motor  sat_s  aborts  items"]
        rj = {}
        for k in TIERS:
            for th in (1, 5):
                sn = f"T{k}_x{th}"
                for pn, gr in [("nominal", {"hard_seq2": fin[sn]["hard_seq2"]})] + \
                              [(key.split("|")[1], v) for key, v in rb.items() if key.split("|")[0] == sn]:
                    r = group_row(gr["hard_seq2"], k)
                    rj.setdefault(sn, {})[pn] = r
                    R.append(f"{pn:18s} {sn:8s} {r['caught']:>3d}/{r['launched']:<4d} {r['tier_viol_trials']:>4d}/30   "
                             f"{r['disc_min_mm']:>5.1f} {r['guard_min_mm']:>5.1f} {r['motor_peak']:>6.3f} "
                             f"{r['saturation_longest_s_max']:>5} {r['aborts']:>4d}   {r['tier_viol_items']}")
                R.append("")
        open(os.path.join(OUT, "robust", "robust_table.txt"), "w").write("\n".join(R) + "\n")
        json.dump(rj, open(os.path.join(OUT, "robust", "robust_table.json"), "w"), indent=1)
        print("\n".join(R))
    # p2p
    if os.path.exists(os.path.join(OUT, "p2p", "p2p_tiers.json")):
        pp = ld("p2p", "p2p_tiers.json")
        P = ["setting  heading  dist  T_plan  ready10  ready5  overshoot  disc   guard  motor  tilt"]
        for sn in sorted(pp, key=lambda s: (s[-1], s)):
            for r in pp[sn]:
                if "error" in r:
                    P.append(f"{sn} {r['heading_deg']} {r['distance_m']} ERROR")
                    continue
                P.append(f"{sn:8s} {r['heading_deg']:>5.0f}   {r['distance_m']:>4.1f}  {r['T_plan_s']:>6.2f}  "
                         f"{r['t_ready_100mm_s']!s:>7}  {r['t_ready_50mm_s']!s:>6}  {r['overshoot_mm']!s:>9}  "
                         f"{r['disc_min']['mm']:>5.1f}  {r['guard_min']['mm']:>5.1f}  {r['motor_max']:>5.3f}  {r['tilt_max_deg']}")
            P.append("")
        open(os.path.join(OUT, "p2p", "p2p_tiers.txt"), "w").write("\n".join(P) + "\n")
        print("\n".join(P))
    # reach
    if os.path.exists(os.path.join(OUT, "reach", "reach_tiers.json")):
        rr = ld("reach", "reach_tiers.json")
        Q = ["setting   cells  caught  caught within tier  tier-violation cells  disc_min  items"]
        rq = {}
        for sn in sorted(rr, key=lambda s: (s[-1], s)):
            k = int(sn[1])
            c = [x for x in rr[sn] if "caught" in x]
            vs = [tier_viol(x["rec"], k) for x in c]
            items = {}
            for v in vs:
                for x in v:
                    items[x] = items.get(x, 0) + 1
            rq[sn] = {"cells": len(c), "caught": int(sum(x["caught"] for x in c)),
                      "caught_within_tier": int(sum(x["caught"] and not v for x, v in zip(c, vs))),
                      "viol_cells": int(sum(bool(v) for v in vs)), "items": items,
                      "disc_min_mm": min(x["rec"]["disc_min_mm"] for x in c)}
            Q.append(f"{sn:8s} {len(c):>6d} {rq[sn]['caught']:>7d} {rq[sn]['caught_within_tier']:>18d} "
                     f"{rq[sn]['viol_cells']:>20d} {rq[sn]['disc_min_mm']:>9.1f}  {items}")
        open(os.path.join(OUT, "reach", "reach_tiers.txt"), "w").write("\n".join(Q) + "\n")
        json.dump(rq, open(os.path.join(OUT, "reach", "reach_tiers_summary.json"), "w"), indent=1)
        print("\n".join(Q))


if __name__ == "__main__":
    main(*sys.argv[1:])
