"""Screening on tuning seeds: python screen2.py <round> <variants.json> [seed_from seed_to]
variants.json: {name: [thrust, overrides-on-top-of-REC0[thrust]]}"""
from drv2 import *
if __name__ == "__main__":
    rnd = sys.argv[1]
    V = json.load(open(sys.argv[2]))
    s0, s1 = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (101, 115)
    SEEDS = list(range(s0, s1 + 1))
    settings = {n: (th, merge(REC0[th], ov)) for n, (th, ov) in V.items()}
    out, res, keys = catch_groups(settings, SEEDS, name=f"screen_{rnd}")
    save({"seeds": SEEDS, "variants": {n: [th, ov] for n, (th, ov) in settings.items()}, "results": out},
         os.path.join(OUT, "screening", f"round_{rnd}.json"))
    lines = []
    for sn, gr in out.items():
        for g, a in gr.items():
            lines.append(f"{sn:26s} {g:10s} {brief(a)}")
        tt = [setting_total(gr, k) for k in TIERS]
        lines.append(f"{sn:26s} TOTAL      {tt[0]['caught']}/{tt[0]['balls']} viol " +
                     " ".join(f"T{k}:{tt[k]['viol']}{tt[k]['items']}" for k in TIERS) +
                     f" disc {tt[0]['disc_min_mm']} motor {tt[0]['motor_max']} sat {tt[0]['sat_longest_s']}")
    open(os.path.join(OUT, "screening", f"round_{rnd}.txt"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
