"""Regression of the five references against the stored logs in out_official (netdyn off, i.e. the default):
1 m hold --ff, mission seed 1 (45.81 s), 5x mission seed 1 (46.12 s), catch medium seed 5 (4/5), 5x catch seed 5 (5/5).
Every array of the new log is compared bit for bit with the stored one; summaries without wall-clock times, files,
the cfg block and the "reference" key (added by boxfix); summary_equal also fails on keys the stored summaries do not
have yet (analysis keys added by later studies), summary_shared_values_differing lists real value differences. python regress.py <work_dir> -> prints, regress.json"""
import sys, os, json, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
REPO = os.path.abspath(os.path.join(OUT, "..", ".."))
OO = os.path.join(REPO, "out_official")
sys.path.insert(0, REPO)
import net_sim as N

SKIP = {"wall_s", "wall_time_s", "clearance_wall_time_s", "replan_wall_s", "files", "cfg", "reference", "snapshot",
        "wall_time", "elapsed_s"}


def strip(d):
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if k not in SKIP and "wall" not in k}
    if isinstance(d, list):
        return [strip(x) for x in d]
    return d


def shared_diff(a, b, path=""):
    """Values that differ in keys present in both summaries (the stored summaries predate a few analysis keys)."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) & set(b):
            out += shared_diff(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length"]
        for i, (x, y) in enumerate(zip(a, b)):
            out += shared_diff(x, y, f"{path}[{i}]")
    elif a != b:
        out.append(path)
    return out


def cmp_logs(a, b):
    A, B = np.load(a, allow_pickle=True), np.load(b, allow_pickle=True)
    ka, kb = set(A.files), set(B.files)
    diff = [k for k in sorted(ka & kb) if not (A[k].shape == B[k].shape and
                                                np.array_equal(A[k], B[k], equal_nan=A[k].dtype.kind == "f"))]
    return {"arrays": len(ka & kb), "only_new": sorted(ka - kb), "only_stored": sorted(kb - ka), "different": diff}


def main():
    W = sys.argv[1]
    os.makedirs(W, exist_ok=True)
    res = {}
    # 1. hold --ff
    subprocess.run([sys.executable, os.path.join(REPO, "net_sim.py"), "--ff", os.path.join(OO, "sweep_summary.json"),
                    "--out", os.path.join(W, "hold")],
                   check=True, stdout=subprocess.DEVNULL)
    res["hold_ff"] = cmp_logs(os.path.join(W, "hold", "hold_ff_log.npz"), os.path.join(OO, "hold_ff_log.npz"))
    a = strip(json.load(open(os.path.join(W, "hold", "hold_ff_summary.json"))))
    b = strip(json.load(open(os.path.join(OO, "hold_ff_summary.json"))))
    res["hold_ff"]["summary_diff_keys"] = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
    res["hold_ff"]["feedforward"] = (a.get("feedforward"), b.get("feedforward"))
    # 2, 3. missions (cfg from the stored summaries)
    for name, d, tag in (("mission_seed1", "missions_tol50", "mission_seed1_relaxed_f1.2"),
                         ("mission_seed1_x5", os.path.join("thrust5", "missions_tilt45"), "mission_seed1_relaxed_f1.3")):
        st = json.load(open(os.path.join(OO, d, f"{tag}_summary.json")))
        cfg = {k: v for k, v in st["cfg"].items() if k != "params_note"}
        o = os.path.join(W, name)
        os.makedirs(o, exist_ok=True)
        met = N.run_mission(cfg, out=o, tag=tag, verbose=False)
        r = cmp_logs(os.path.join(o, f"{tag}_log.npz"), os.path.join(OO, d, f"{tag}_log.npz"))
        new = json.load(open(os.path.join(o, f"{tag}_summary.json")))
        r["summary_equal"] = strip(new) == strip(st)
        r["summary_shared_values_differing"] = shared_diff(strip(new), strip(st))
        r["total_leg_time_s"] = new.get("total_leg_time_s")
        res[name] = r
    # 4, 5. catch medium seed 5, x1 and x5
    for name, d in (("catch_seed5", os.path.join("catch", "trials")),
                    ("catch_seed5_x5", os.path.join("thrust5", "catch", "trials"))):
        tag = "catch_medium_sequence_seed5"
        st = json.load(open(os.path.join(OO, d, f"{tag}_summary.json")))
        cfg = dict(st["cfg"])
        o = os.path.join(W, name)
        os.makedirs(o, exist_ok=True)
        met = N.run_catch(cfg, out=o, tag=tag, verbose=False)
        r = cmp_logs(os.path.join(o, f"{tag}_log.npz"), os.path.join(OO, d, f"{tag}_log.npz"))
        new = json.load(open(os.path.join(o, f"{tag}_summary.json")))
        r["summary_equal"] = strip(new) == strip(st)
        r["summary_shared_values_differing"] = shared_diff(strip(new), strip(st))
        r["caught"] = f"{met['n_caught']}/{met['n_balls']}"
        res[name] = r
    json.dump(res, open(os.path.join(W, "regress.json"), "w"), indent=1)
    for k, v in res.items():
        print(k, v)


if __name__ == "__main__":
    main()
