"""Finals / robustness / p2p / reach for the selected settings: python final_r.py <stage> [seed_from seed_to]
stage: settings | final | robust | p2p | reach.  sel.json: {"1": {"1": [round, variant]}, ...} (tier -> thrust)."""
from drv2 import *
OLD = os.path.join(REPO, "out_official", "catch_speed", "settings")
ROBUST = {"net_mass_+20%": {"m_net": 0.024},
          "strings_+-10mm": {"string_each": [0.21, 0.20, 0.19, 0.20]}}


def selected():
    S = {}
    for th in (1, 5):
        S[f"T0_x{th}"] = (th, load_setting(os.path.join(OLD, f"catch_speed_x{th}.json")))
    for k in (1, 2, 3):
        for th in (1, 5):
            S[f"T{k}_x{th}"] = (th, load_setting(os.path.join(OUT, "settings", f"tier{k}_x{th}.json")))
    return S


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "settings":
        sel = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sel.json")))
        for k, d in sel.items():
            for th, (rnd, name) in d.items():
                js = json.load(open(os.path.join(OUT, "screening", f"round_{rnd}.json")))
                ov = js["variants"][name][1]
                T = TIERS[int(k)]
                note = (f"tier {k} (motor line {T['motor'] or 'none (saturation reported)'}, rotor-disc line "
                        f"{T['disc_mm']:g} mm), thrust x{th}: screening variant {name} (round {rnd}, seeds 101-130)"
                        + (" -- use with --thrust-scale 5" if th == "5" else ""))
                write_setting(os.path.join(OUT, "settings", f"tier{k}_x{th}.json"), int(th), ov, note)
                print(k, th, name, json.dumps(ov))
    elif stage == "final":
        s0, s1 = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (1, 30)
        SEEDS = list(range(s0, s1 + 1))
        S = selected()
        out, res, keys = catch_groups(S, SEEDS, name="final")
        tag = f"final_seeds{s0}-{s1}"
        save({"seeds": SEEDS, "settings": {k: v for k, v in S.items()}, "results": out}, os.path.join(OUT, "final", f"{tag}.json"))
        lines = [f"{sn:10s} {g:10s} {brief(a)}" for sn, gr in out.items() for g, a in gr.items()]
        open(os.path.join(OUT, "final", f"{tag}.txt"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
    elif stage == "robust":
        SEEDS = list(range(1, 31))
        S0 = selected()
        S = {}
        for rn, po in ROBUST.items():
            for sn, (th, ov) in S0.items():
                S[f"{sn}|{rn}"] = (th, merge(ov, {"p_over": po}))
        out, res, keys = catch_groups(S, SEEDS, groups=("hard_seq2",), name="robust")
        save({"seeds": SEEDS, "perturbations": ROBUST, "results": out}, os.path.join(OUT, "robust", "robust_hard_seq2.json"))
        lines = [f"{sn:30s} {g:10s} {brief(a)}" for sn, gr in out.items() for g, a in gr.items()]
        open(os.path.join(OUT, "robust", "robust_hard_seq2.txt"), "w").write("\n".join(lines) + "\n")
        print("\n".join(lines))
    elif stage == "p2p":
        r = p2p_grid(selected())
        save(r, os.path.join(OUT, "p2p", "p2p_tiers.json"))
    elif stage == "reach":
        r = reach(selected())
        save(r, os.path.join(OUT, "reach", "reach_tiers.json"))
        for n, cells in r.items():
            c = [x for x in cells if "caught" in x]
            print(n, len(c), "caught", sum(x["caught"] for x in c))
