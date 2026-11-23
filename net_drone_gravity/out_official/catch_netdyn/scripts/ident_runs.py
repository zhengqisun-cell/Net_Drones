"""Identification data for the net-dynamics study: point-to-point moves (cfg p2p) of the current planner with 0..10 balls
dropped into the net first (p2p n_drop). Writes ident/raw_<set>.npz (compact arrays from 2 s before the move to the end).
python ident_runs.py [set ...]    sets: x1 (thrust x1, n = 0..5), x5 (thrust x5, n = 0, 2, 5, 8, 10), x5b (x5, n = 1, 3, 4, 6, 7, 9)"""
import sys, os, json, tempfile, shutil
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))
OO = os.path.abspath(os.path.join(OUT, ".."))
REPO = os.path.abspath(os.path.join(OO, ".."))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(OO, "catch_speed_relaxed", "scripts"))
import net_sim as N
import drv2

KEYS = ("t", "pos", "vel", "ref", "ref_vel", "net_c", "net_com", "tilt", "motor", "tension", "str_dir", "ref_c")
H_DN10 = json.load(open(os.path.join(OO, "catch_10balls", "settings", "x5_tier0_10balls.json")))["h_dn_table"]
SETS = {"x1": (1, os.path.join(OO, "catch_speed", "settings", "catch_speed_x1.json"), (0, 1, 2, 3, 4, 5)),
        "x5": (5, os.path.join(OO, "catch_speed", "settings", "catch_speed_x5.json"), (0, 2, 5, 8, 10)),
        "x5b": (5, os.path.join(OO, "catch_speed", "settings", "catch_speed_x5.json"), (1, 3, 4, 6, 7, 9))}
MOVES = ([("h", d, h) for h in (0.0, 45.0) for d in (0.5, 1.0, 2.0, 3.0)]
         + [("v", 0.4, None), ("v", -0.4, None), ("o", 1.0, 0.4)])


def move_d(kind, a, b):
    if kind == "h":
        return [a * np.cos(np.radians(b)), a * np.sin(np.radians(b)), 0.0]
    if kind == "v":
        return [0.0, 0.0, a]
    return [a, 0.0, b]


def job(arg):
    cfg, tmp = arg
    try:
        m = N.run_catch(cfg, out=tmp, tag=cfg["label"], verbose=False)
        path = os.path.join(tmp, f"{cfg['label']}_log.npz")
        L = np.load(path)
        tm = m["p2p"]["t_move_s"]
        w = L["t"] >= tm - 2.0
        d = {k: L[k][w].astype(np.float32 if k != "t" else np.float64) for k in KEYS}
        L.close()
        os.remove(path)
        os.remove(os.path.join(tmp, f"{cfg['label']}_summary.json"))
        return {"label": cfg["label"], "t_move": tm, "p2p": m["p2p"], "data": d}
    except Exception as exc:
        import traceback
        return {"label": cfg["label"], "error": traceback.format_exc()}


def cfgs_for(name, seed=101):
    th, sp, ns = SETS[name]
    base = drv2.setting(th, drv2.load_setting(sp))
    out = []
    for n in ns:
        for kind, a, b in MOVES:
            d = move_d(kind, a, b)
            lab = f"{name[:2]}_n{n}_{kind}{a:g}" + ("" if b is None else f"_{b:g}")
            c = drv2.merge(base, {"seed": seed, "label": lab, "p2p": {"d": d, "n_drop": n, "t_after_s": 6.0}})
            if th == 5:
                c["h_dn_table"] = H_DN10
            out.append(c)
    return out


if __name__ == "__main__":
    sets = sys.argv[1:] or list(SETS)
    os.makedirs(os.path.join(OUT, "ident"), exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="netdyn_ident_")
    try:
        for name in sets:
            cf = cfgs_for(name)
            with N._pool(drv2.JOBS) as pool:
                res = pool.map(job, [(c, tmp) for c in cf], chunksize=1)
            err = [r for r in res if "error" in r]
            for e in err[:3]:
                print(e["error"])
            arr = {}
            meta = []
            for r in res:
                if "error" in r:
                    continue
                for k, v in r["data"].items():
                    arr[f"{r['label']}|{k}"] = v
                meta.append({"label": r["label"], "t_move": r["t_move"], "p2p": r["p2p"]})
            np.savez_compressed(os.path.join(OUT, "ident", f"raw_{name}.npz"), **arr)
            json.dump(meta, open(os.path.join(OUT, "ident", f"raw_{name}_meta.json"), "w"), indent=1, default=N._json_np)
            print(f"{name}: {len(res)} runs, {len(err)} errors", flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
