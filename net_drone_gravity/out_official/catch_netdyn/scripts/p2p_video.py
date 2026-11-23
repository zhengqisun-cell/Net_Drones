"""Video of a point-to-point log (catch p2p mode, no ball predictions): a side camera that follows the move, the balls
resting in the net, the target (red), the net-centre path (blue); overlays with the time from the move start, the
net centre's horizontal distance to the target and the formation centre's distance (offscreen renderer, window check
as in render_scene). python p2p_video.py <log.npz> <out.mp4> "<title>" [t_before t_after]"""
import os, sys, json
import numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
sys.path.insert(0, REPO)
import mujoco
import render_scene as RS
from PIL import Image, ImageDraw


def main():
    logp, mp4, title = sys.argv[1], sys.argv[2], sys.argv[3]
    tb, ta = (float(sys.argv[4]), float(sys.argv[5])) if len(sys.argv) > 5 else (1.0, 4.5)
    import av
    log = dict(np.load(logp))
    summary = json.load(open(logp.replace("_log.npz", "_summary.json")))
    p = RS.load_params(summary)
    pp = summary["p2p"]
    tm = pp["t_move_s"]
    W, H, fps = 960, 540, 30
    af = RS.load_airframe()
    off = np.array(summary["box"]["offset_m"], float)
    sc = RS.Scene(p, W, H, "mesh", af, floor_half=8.0, box=(summary["box"]["size_m"], off.tolist()))
    sun = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
    times = tm - tb + np.arange(int((tb + ta) * fps) + 1) / fps
    lt = log["t"]
    tgt_c = np.array(log["ref_c"][-1], float)                       # final formation reference (target)
    net_end = None
    bp = log.get("ball_pos")
    container = av.open(mp4, mode="w")
    st = container.add_stream("libx264", rate=fps)
    st.width, st.height, st.pix_fmt = W, H, "yuv420p"
    st.options = {"crf": "20", "preset": "medium", "movflags": "+faststart"}
    guard = RS.WindowGuard()
    renderer = mujoco.Renderer(sc.m, height=H, width=W, max_geom=20000)
    f_big, f_mid, f_small = RS._font(22), RS._font(18), RS._font(15)
    fc0 = None
    path = []
    try:
        guard(renderer, "after creating the renderer")
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        opt.sitegroup[:] = 0
        for fi, t in enumerate(times):
            fr, pos, quat = RS.interp_state(log, t)
            pos = sc.aligned_pos(fr, pos, quat)
            sc.set_state(fr, pos, quat)
            fc = pos.mean(axis=0)
            nc = fr[:p.n * p.n].reshape(p.n, p.n, 3)[p.n // 2, p.n // 2]
            if net_end is None:
                net_end = tgt_c - np.array([0, 0, float(np.median(log["pos"][-100:, :, 2].mean(1) - log["net_c"][-100:, 2]))])
                fc0 = fc.copy()
            path.append(nc.copy())
            cam = RS.mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            mid = 0.5 * (fc0 + tgt_c)
            cam.lookat[:] = [mid[0], mid[1], 0.5 * (fc[2] + nc[2])]
            cam.distance, cam.azimuth, cam.elevation = 2.6 + 0.6 * np.linalg.norm(tgt_c - fc0), -90.0, -12.0
            sc.m.light_pos[sun] = (fc[0], fc[1], fc[2] + 3.0)
            renderer.update_scene(sc.d, camera=cam, scene_option=opt)
            scn = renderer.scene
            RS._add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [0.025, 0, 0], net_end, [0.95, 0.1, 0.1, 1.0])
            RS._add_segment(scn, net_end, [net_end[0], net_end[1], 0.0], 0.006, [0.95, 0.1, 0.1, 0.5])
            for a_, b_ in zip(path[:-1:2], path[2::2]):
                if np.linalg.norm(b_ - a_) > 1e-4:
                    RS._add_segment(scn, a_, b_, 0.008, [0.10, 0.35, 0.90, 1.0])
            if bp is not None and bp.shape[1]:
                i = int(np.clip(np.searchsorted(lt, t), 0, len(lt) - 1))
                for x in bp[i]:
                    if np.isfinite(x).all():
                        RS._add_geom(scn, mujoco.mjtGeom.mjGEOM_SPHERE, [p.ball_d / 2, 0, 0], x, [1.0, 0.75, 0.1, 1.0])
            img = renderer.render().copy()
            if fi % 10 == 0 or fi == len(times) - 1:
                guard(renderer, f"after rendering frame {fi}")
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im)
            txt = dict(fill=(20, 20, 20), stroke_width=3, stroke_fill=(255, 255, 255))
            dn = 1e3 * np.hypot(*(nc[:2] - tgt_c[:2]))
            df = 1e3 * np.linalg.norm(fc - tgt_c)
            ready = (pp["t_ready_50mm_s"] is not None and t - tm >= pp["t_ready_50mm_s"])
            dr.text((12, 10), title, font=f_big, **txt)
            dr.text((12, 40), f"t = {t - tm:+5.2f} s from the move start    planned duration {pp['T_plan_s']:.2f} s",
                    font=f_mid, **txt)
            dr.text((12, 64), f"net centre to target (horizontal) {dn:6.0f} mm", font=f_mid, **txt)
            dr.text((12, 88), f"formation centre to target {df:6.0f} mm", font=f_mid, **txt)
            dr.text((12, 112), f"ready (5 cm, stays) at {pp['t_ready_50mm_s']} s" + ("   READY" if ready else ""),
                    font=f_mid, fill=(10, 130, 40) if ready else (20, 20, 20), stroke_width=3,
                    stroke_fill=(255, 255, 255))
            dr.text((12, H - 28), "red: target of the net centre   blue: net-centre path   yellow: balls in the net",
                    font=f_small, **txt)
            for pk in st.encode(av.VideoFrame.from_ndarray(np.asarray(im), format="rgb24")):
                container.mux(pk)
        for pk in st.encode():
            container.mux(pk)
    finally:
        container.close()
        renderer.close()
    print(mp4, len(times), "frames")


if __name__ == "__main__":
    main()
