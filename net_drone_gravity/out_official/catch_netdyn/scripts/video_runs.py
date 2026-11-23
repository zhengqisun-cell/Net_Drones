"""Logs for the comparison videos (old planner vs net-dynamics planner), then render and compose:
  p2p: thrust x5, 5 balls in the net, 0.5 m move along x (x5_T0 base vs x5_T0_nd);
  catch: thrust x5 tier 0, medium 2 s, seed 17 (old: ball 5 late, 4/5; net-dynamics planner 5/5).
python video_runs.py   -> videos/<name>/{old,new}/ logs and videos (p2p_video.py / render_scene.py --video);
side by side with ../catch_boxfix/scripts/compose.py"""
import os, sys, json, subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import nd_runs as R
import ident_runs as IR
from nd_runs import N, merge

V = os.path.join(R.OUT, "videos")
REPO = os.path.abspath(os.path.join(R.OO, ".."))
CASES = {"p2p_x5_n5_0.5m": [("old", "base"), ("new", "nd")],
         "catch_x5T0_medium2s_seed17": [("old", "base"), ("new", "nd")]}


def main():
    for name, pair in CASES.items():
        for side, cand in pair:
            c = R.cfg_for("x5_T0", cand)
            if name.startswith("p2p"):
                c = merge(c, {"seed": 101, "p2p": {"d": [0.5, 0.0, 0.0], "n_drop": 5, "t_after_s": 6.0}})
                c["h_dn_table"] = IR.H_DN10
                c["n_balls"] = (3, 5)
            else:
                c = merge(c, {**R.GSTD["med_seq2"], "seed": 17})
            if "netdyn" in R.cfg_for("x5_T0", cand):
                c["netdyn"] = R.cfg_for("x5_T0", cand)["netdyn"]
            out = os.path.join(V, name, side)
            os.makedirs(out, exist_ok=True)
            m = N.run_catch(c, out=out, tag=side, verbose=False)
            print(name, side, m.get("n_caught"), (m.get("p2p") or {}).get("t_ready_50mm_s"), flush=True)
            lg = os.path.join(out, f"{side}_log.npz")
            if m.get("p2p"):
                title = "old planner (S-curve)" if side == "old" else "net-dynamics planner"
                cmd = [sys.executable, os.path.join(HERE, "p2p_video.py"), lg, os.path.join(out, f"{side}_video.mp4"),
                       title]
            else:
                cmd = [sys.executable, os.path.join(REPO, "render_scene.py"), "--log", lg, "--video", "--out", out]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
