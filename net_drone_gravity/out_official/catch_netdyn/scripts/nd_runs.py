"""Catch batches of the net-dynamics study.
python nd_runs.py <tag> <seed_from> <seed_to> <family[,family]> <candidate[,candidate]> [k=v ...]
families: x1_T0, x5_T0, x5_T1 (3-5 balls, four standard groups), x5_T0_10, x5_T1_10 (10 balls, four groups).
candidates: names in CANDIDATES (overrides on top of the family's settings file) or settings/<name>.json files.
Options: p_over={...} (robustness: Params overrides, e.g. {"m_net": 0.024} or {"string_each": [0.21, 0.2, 0.19, 0.2]}),
groups=g1,g2.
Resumable: each (family, candidate, group) chunk goes to runs/<tag>/ as soon as it is done (temporary file + rename);
a restart skips the chunks on disk. runs/<tag>.json collects them: per family, candidate and group the trial records
(drv2.trial_rec + per-ball records + tier violations at the family's tier)."""
import sys, os, json, copy
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
OO = os.path.abspath(os.path.join(OUT, ".."))
sys.path.insert(0, os.path.join(OO, "catch_speed_relaxed", "scripts"))
sys.path.insert(0, os.path.join(OO, "catch_10balls", "scripts"))
import drv2
from drv2 import N, merge, setting, load_setting, trial_rec, tier_viol, save
from run10 import ball_recs

GSTD = {"med_seq2": {"tier": "medium", "launch": "sequence", "interval_s": 2.0},
        "med_seq1": {"tier": "medium", "launch": "sequence", "interval_s": 1.0},
        "hard_seq2": {"tier": "hard", "launch": "sequence", "interval_s": 2.0},
        "med_salvo": {"tier": "medium", "launch": "salvo", "interval_s": 2.0}}
G10 = {"med2": {"tier": "medium", "launch": "sequence", "interval_s": 2.0},
       "med1": {"tier": "medium", "launch": "sequence", "interval_s": 1.0},
       "hard2": {"tier": "hard", "launch": "sequence", "interval_s": 2.0},
       "hard1": {"tier": "hard", "launch": "sequence", "interval_s": 1.0}}
FAMILIES = {"x1_T0": (1, 0, os.path.join(OO, "catch_speed", "settings", "catch_speed_x1.json"), GSTD),
            "x5_T0": (5, 0, os.path.join(OO, "catch_speed", "settings", "catch_speed_x5.json"), GSTD),
            "x5_T1": (5, 1, os.path.join(OO, "catch_speed_relaxed", "settings", "tier1_x5.json"), GSTD),
            "x5_T0_10": (5, 0, os.path.join(OO, "catch_10balls", "settings", "x5_tier0_10balls.json"), G10),
            "x5_T1_10": (5, 1, os.path.join(OO, "catch_10balls", "settings", "x5_tier1_10balls.json"), G10)}
MOTOR_LINE = {0: 0.90, 1: 0.95}
FF = {"plan": False, "drag_ff": True, "lead_s": 0.07}
ND = {"form": False, "drag_ff": True, "lead_s": 0.07}


def candidates(fam):
    th, tier, path, _ = FAMILIES[fam]
    base = load_setting(path)
    lim = base.get("limits", {})
    ah = float(lim.get("a_h", 3.0))
    up = {"limits": {"a_h": ah + 0.5, "a_tot": float(lim.get("a_tot", 3.0)) + 0.5}}
    ml = {"motor_line": MOTOR_LINE[tier]}
    return {"base": {},
            "ff": {"netdyn": {**FF, **ml}},
            "nd": {"netdyn": {**ND, **ml}},
            "nd_te0.1": {"netdyn": {**ND, **ml}, "t_early_s": 0.1},
            "nd_meas": {"netdyn": {**ND, **ml, "state": "measured"}},
            "nd_ah+": {"netdyn": {**ND, **ml}, **up},
            "ff_ah+": {"netdyn": {**FF, **ml}, **up},
            "nd_form": {"netdyn": {**ND, **ml, "form": True}}}     # with the formation modes (p2p only)


def cfg_for(fam, cand, extra=None):
    th, tier, path, _ = FAMILIES[fam]
    ov = load_setting(path)
    if cand in candidates(fam):
        c = copy.deepcopy(candidates(fam)[cand])
    else:
        c = load_setting(os.path.join(OUT, "settings", f"{cand}.json"))
    nd = c.pop("netdyn", None)
    base = merge(setting(th, ov), c)
    if nd is not None:
        base["netdyn"] = nd
    if extra:
        for k, v in extra.items():
            if k == "p_over":
                base["p_over"] = {**(base.get("p_over") or {}), **v}
            else:
                base[k] = v
    return base


def rec(s, m, tier):
    t = trial_rec(s, m)
    t["balls"] = ball_recs(m)
    t["net_lowest_m"] = m["safety"]["net_low"]["net_lowest_m"]
    t["box_min_dist_m"] = m["safety"]["box"]["min_dist_m"]
    t["reference"] = m.get("reference")
    t["tier_viol"] = tier_viol(t, tier)
    if "netdyn" in m:
        t["netdyn"] = m["netdyn"]["stats"]
    return t


def save_atomic(obj, path):
    """Write JSON to a temporary file next to path, then rename (no half-written file after a crash)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=N._json_np)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _job(arg):
    i, cfg = arg
    return i, N._catch_job(cfg)


def chunk_path(tag, fam, cand, g):
    return os.path.join(OUT, "runs", tag, f"{fam}__{cand}__{g}.json")


def assemble(tag):
    """Collect the finished chunks of a tag into runs/<tag>.json."""
    out, seeds = {}, set()
    d = os.path.join(OUT, "runs", tag)
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        fam, cand, g = fn[:-5].split("__")
        c = json.load(open(os.path.join(d, fn)))
        out.setdefault(fam, {}).setdefault(cand, {})[g] = c["trials"]
        seeds |= {t["seed"] for t in c["trials"]}
    save_atomic({"seeds": sorted(seeds), "results": out}, os.path.join(OUT, "runs", f"{tag}.json"))
    return out


def main():
    """Resumable: every (family, candidate, group) chunk is written to runs/<tag>/ as soon as its trials are done
    (temporary file + rename); a restart skips the chunks already on disk. runs/<tag>.json collects them at the end."""
    import tempfile, shutil
    tag, s0, s1, fams, cands = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(","), \
        sys.argv[5].split(",")
    extra, gsel = {}, None
    for a in sys.argv[6:]:
        k, v = a.split("=", 1)
        if k == "groups":
            gsel = v.split(",")
        else:
            extra[k] = json.loads(v)
    seeds = list(range(s0, s1 + 1))
    chunks = []                                  # (fam, cand, g, tier, [cfg ...])
    for fam in fams:
        th, tier, path, groups = FAMILIES[fam]
        for cand in cands:
            base = cfg_for(fam, cand, extra)
            for g, gc in groups.items():
                if gsel and g not in gsel:
                    continue
                if os.path.exists(chunk_path(tag, fam, cand, g)):
                    continue
                cl = []
                for s in seeds:
                    c = merge(base, {**gc, "seed": s, "label": f"{fam}_{cand}_{g}_seed{s}"})
                    if "netdyn" in base:
                        c["netdyn"] = base["netdyn"]
                    cl.append(c)
                chunks.append((fam, cand, g, tier, cl))
    print(f"{tag}: {len(chunks)} chunks to run", flush=True)
    if chunks:
        snap = tempfile.mkdtemp(prefix="net_sim_snap_")      # fresh snapshot cache, deleted at the end
        flat, owner = [], []
        for ci, (_, _, _, _, cl) in enumerate(chunks):
            for c in cl:
                c["snap_dir"] = snap
                flat.append(c)
                owner.append(ci)
        got = [dict() for _ in chunks]
        try:
            with N._pool(drv2.JOBS) as pool:
                for i, m in pool.imap_unordered(_job, list(enumerate(flat)), chunksize=1):
                    ci = owner[i]
                    got[ci][i] = m
                    fam, cand, g, tier, cl = chunks[ci]
                    if len(got[ci]) == len(cl):
                        trials = []
                        for j in sorted(got[ci]):
                            mm, s = got[ci][j], flat[j]["seed"]
                            trials.append({"seed": s, "error": mm["error"], "trace": mm.get("trace")} if "error" in mm
                                          else rec(s, mm, tier))
                        save_atomic({"family": fam, "candidate": cand, "group": g, "extra": extra,
                                     "trials": trials}, chunk_path(tag, fam, cand, g))
                        ok = [t for t in trials if "error" not in t]
                        print(f"  done {fam} {cand} {g}: caught {sum(t['n_caught'] for t in ok)}/"
                              f"{sum(t['n_launched'] for t in ok)} viol {sum(bool(t['tier_viol']) for t in ok)} "
                              f"errors {len(trials) - len(ok)}", flush=True)
        finally:
            shutil.rmtree(snap, ignore_errors=True)
    out = assemble(tag)
    for fam, cc in out.items():
        for cand, gr in cc.items():
            ok = [t for ts in gr.values() for t in ts if "error" not in t]
            if not ok:
                continue
            print(f"{fam:9s} {cand:12s} caught {sum(t['n_caught'] for t in ok)}/{sum(t['n_launched'] for t in ok)} "
                  f"viol {sum(bool(t['tier_viol']) for t in ok)}/{len(ok)} aborts {sum(t['aborted'] for t in ok)} "
                  f"errors {sum('error' in t for ts in gr.values() for t in ts)}", flush=True)


if __name__ == "__main__":
    main()
