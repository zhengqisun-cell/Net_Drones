"""MuJoCo-rendered screenshots of the net-and-drones scene, made from a saved hold-run log.

Offscreen only: mujoco.Renderer gets its OpenGL context from GLFW through a window created
with the "visible" hint off. Every run lists the top-level windows owned by this process
while the renderer exists and stops (closing the renderer) if any of them is visible.
No viewer, no matplotlib window, no simulation: the scene is rebuilt from the log.

    python render_scene.py                                   # default set from out_official/hold_ff_log.npz
    python render_scene.py --times 21 22.5 --view wide       # extra frames, one camera
    python render_scene.py --log out/hold_ff_log.npz --out some/folder
    python render_scene.py --style schematic                 # schematic drones, into screenshots/
    python render_scene.py --log out_official/missions_tol50/mission_seed1_relaxed_f1.2_log.npz --video
                                                             # a mission log: mp4 of the whole mission (two views)
    python render_scene.py --log out_official/catch/videos/<trial>_log.npz
                                                             # a catch log (net_sim --mode catch): mp4 of the trial,
                                                             # balls, launcher, predicted paths, plus key-frame PNGs
    python render_scene.py --log out_official/travel/<run>_log.npz --out out_official/travel/screenshots
                                                             # a travel-mode log: camera, floor and sun follow
                                                             # the formation; default shots at cruise

Drone style "mesh" (default when assets/crazyflow_cf21B/ exists, output screenshots_mesh/) draws
Crazyflow's Crazyflie 2.1 Brushless airframe meshes (learnsyslab/drone-models, see SOURCE.txt there)
on each drone body; its XML is read as data and only its appearance is used.

The model is net_sim.build_xml(p) with visual-only additions (rotor discs, arms, hook marker,
lights, floor material, tendon widths and colours, tendons drawn straight even when slack). They are added here, in the render path;
net_sim.py is not changed and its simulation never sees them.
"""
import argparse
import ctypes
import dataclasses
import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from ctypes import wintypes

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import net_sim as ns          # noqa: E402  (model builder and parameters only)
import mujoco                 # noqa: E402
from PIL import Image, ImageDraw, ImageFont   # noqa: E402

# colours (rgba)
C_CCW = "0.85 0.25 0.20 0.55"      # rotor disc, counter-clockwise seen from above
C_CW = "0.15 0.45 0.85 0.55"       # rotor disc, clockwise
C_FRAME = "0.20 0.20 0.22 1"
C_HOOK = "0.95 0.15 0.85 1"
C_STRING = "1.00 0.62 0.05 1"
C_NET = "0.10 0.28 0.38 1"
C_KNOT = "0.92 0.92 0.88 1"


# ---------------------------------------------------------------- window check
_user32 = ctypes.WinDLL("user32", use_last_error=True)
_WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_user32.EnumWindows.argtypes = [_WNDENUMPROC, wintypes.LPARAM]
_user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
_user32.IsWindowVisible.argtypes = [wintypes.HWND]
_user32.GetClassNameW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
_user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]


def own_windows():
    """Top-level windows of this process: list of (hwnd, class, title, visible)."""
    pid, found = os.getpid(), []

    def cb(hwnd, _):
        owner = wintypes.DWORD()
        _user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid:
            cls, title = ctypes.create_unicode_buffer(256), ctypes.create_unicode_buffer(256)
            _user32.GetClassNameW(hwnd, cls, 256)
            _user32.GetWindowTextW(hwnd, title, 256)
            found.append((hwnd, cls.value, title.value, bool(_user32.IsWindowVisible(hwnd))))
        return True

    _user32.EnumWindows(_WNDENUMPROC(cb), 0)
    return found


class WindowGuard:
    """Checks, at every call, that no window of this process is visible."""

    def __init__(self):
        self.checks, self.max_windows, self.seen = 0, 0, set()

    def __call__(self, renderer, where):
        w = own_windows()
        self.checks += 1
        self.max_windows = max(self.max_windows, len(w))
        self.seen.update((c, t) for _, c, t, _ in w)
        vis = [x for x in w if x[3]]
        if vis:
            renderer.close()
            raise SystemExit(f"STOP: visible window(s) of this process {where}: {vis} (renderer closed)")


# ---------------------------------------------------------------- model
def load_params(summary):
    d = summary["params"]
    cls = ns.OfficialParams if d.get("drone_model") == "official" else ns.Params
    names = {f.name for f in dataclasses.fields(cls)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in d.items() if k in names}
    return ns.apply_payload(cls(**kw), summary.get("payload"))      # payload sphere (red), if the run had one


def rotor_layout(p):
    """Rotor (x, y) in the body frame and spin (+1 counter-clockwise seen from above)."""
    if ns.is_official(p):
        xy = [(rx * p.rotor_xy, ry * p.rotor_xy) for rx, ry in ns.ROTORS_OFFICIAL]
        yaw = ns.MIX[2]                 # reaction yaw torque sign: -1 <=> rotor spins CCW
    else:
        a = p.arm / np.sqrt(2)
        xy = [(rx * a, ry * a) for rx, ry in ns.CORNERS]
        yaw = ns.SPIN
    return xy, [1 if s < 0 else -1 for s in yaw]


# ---------------------------------------------------------------- Crazyflow airframe meshes (style "mesh")
MESH_DIR = os.path.join(HERE, "assets", "crazyflow_cf21B")
_MESH_XML = "cf21B_500.xml"
_ALLOWED_TAGS = {"mujoco", "compiler", "asset", "material", "mesh", "default", "geom", "site",
                 "worldbody", "body", "inertial", "camera"}


def load_airframe(folder=MESH_DIR):
    """Read Crazyflow's cf21B_500.xml as data (downloaded, untrusted) and return what the render needs.

    Only appearance is taken: materials, mesh files and where the visual mesh geoms sit in the drone
    body frame. Mass, inertia, collision geoms and cameras are ignored. Refuses unexpected elements
    (plugins, includes, ...) and any mesh path that is absolute or leaves the folder.
    Returns dict(materials={name: rgba}, meshes={name: (abs path, scale)},
                 geoms=[(mesh, material, pos, euler)], motors={name: xyz}).
    """
    root = ET.parse(os.path.join(folder, _MESH_XML)).getroot()
    bad = sorted({e.tag for e in root.iter()} - _ALLOWED_TAGS)
    if bad:
        raise SystemExit(f"{_MESH_XML}: unexpected element(s) {bad}; not loading it")
    comp = root.find("compiler")
    meshdir = comp.get("meshdir", "") if comp is not None else ""
    base = os.path.realpath(folder)

    def inside(rel):
        if os.path.isabs(rel) or ":" in rel or ".." in rel.replace("\\", "/").split("/"):
            raise SystemExit(f"{_MESH_XML}: refusing path {rel!r}")
        path = os.path.realpath(os.path.join(folder, meshdir, rel))
        if os.path.commonpath([base, path]) != base or not path.lower().endswith(".stl") or not os.path.isfile(path):
            raise SystemExit(f"{_MESH_XML}: refusing path {rel!r}")
        return path

    materials = {m.get("name"): m.get("rgba") for m in root.iter("material")}
    meshes = {m.get("name"): (inside(m.get("file")), m.get("scale", "1 1 1")) for m in root.iter("mesh")}
    drone = root.find("worldbody/body[@name='drone']")
    geoms = []
    for g in drone.findall("geom"):
        if g.get("mesh") and float(materials[g.get("material")].split()[3]) > 0:   # skip the transparent LEDs
            geoms.append((g.get("mesh"), g.get("material"), g.get("pos", "0 0 0"), g.get("euler", "0 0 0")))
    for b in drone.findall("body"):               # propellers: one geom per child body, no own offset
        for g in b.findall("geom"):
            assert g.get("pos") is None and g.get("euler") is None and b.get("quat") is None
            geoms.append((g.get("mesh"), g.get("material"), b.get("pos", "0 0 0"), b.get("euler", "0 0 0")))
    motors = {s.get("name"): np.array(s.get("pos").split(), float) for s in drone.findall("site")
              if s.get("name", "").startswith("motor")}
    return dict(materials=materials, meshes=meshes, geoms=geoms, motors=motors)


def airframe_geometry(af):
    """Measured from the meshes, airframe alone with its body origin at the world origin."""
    w = ['<mujoco><compiler angle="degree"/><asset>']
    for name, (path, scale) in af["meshes"].items():
        w.append(f'<mesh name="{name}" file="{path}" scale="{scale}"/>')
    w.append('</asset><worldbody><body name="cf">')
    for i, (mesh, _, pos, euler) in enumerate(af["geoms"]):
        w.append(f'<geom name="g{i}" type="mesh" mesh="{mesh}" pos="{pos}" euler="{euler}" mass="0" '
                 f'contype="0" conaffinity="0"/>')
    w.append('<inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/></body></worldbody></mujoco>')
    m = mujoco.MjModel.from_xml_string("".join(w))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    parts = {}
    for i, (mesh, *_rest) in enumerate(af["geoms"]):
        mid = m.geom_dataid[i]
        v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
        xyz = d.geom_xpos[i] + v @ d.geom_xmat[i].reshape(3, 3).T
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        if mesh in parts:
            lo, hi = np.minimum(lo, parts[mesh][0]), np.maximum(hi, parts[mesh][1])
        parts[mesh] = (lo, hi)
    return parts


def visual_xml(p, width, height, style="schematic", af=None, floor_half=None, box=None):
    """net_sim.build_xml(p) plus visual-only detail. Never used for simulation.
    floor_half: draw the floor plane this large (m, half side; travel logs leave the 5 m default).
    box: (size, offset) of a mission box: its 12 edges are drawn as thin visual capsules."""
    root = ET.fromstring(ns.build_xml(p))
    vis = ET.Element("visual")
    ET.SubElement(vis, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(vis, "quality", shadowsize="8192", offsamples="8")
    ET.SubElement(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.35 0.35 0.35", specular="0.05 0.05 0.05")
    ET.SubElement(vis, "map", znear="0.005", zfar="50", shadowclip="1.2")
    root.insert(1, vis)
    root.insert(2, ET.Element("statistic", extent="2", center="0 0 0.6"))
    asset = ET.Element("asset")
    ET.SubElement(asset, "texture", name="sky", type="skybox", builtin="gradient",
                  rgb1="0.93 0.95 0.98", rgb2="0.60 0.68 0.80", width="256", height="256")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", width="512", height="512",
                  rgb1="0.86 0.87 0.88", rgb2="0.78 0.80 0.82", mark="edge", markrgb="0.55 0.57 0.60")
    ET.SubElement(asset, "material", name="floor", texture="grid", texrepeat="2 2", texuniform="true")
    if style == "mesh":
        for name, rgba in af["materials"].items():
            ET.SubElement(asset, "material", name=f"cf_{name}", rgba=rgba)
        for name, (path, scale) in af["meshes"].items():
            ET.SubElement(asset, "mesh", name=f"cf_{name}", file=path, scale=scale)
    root.insert(3, asset)

    wb = root.find("worldbody")
    floor = wb.find("geom[@name='floor']")
    floor.set("material", "floor")
    if floor_half:
        floor.set("size", f"{floor_half} {floor_half} 0.1")
    if box is not None:
        size, off = box
        lo = -np.asarray(off, float)
        cs = [lo + size * np.array([i, j, k]) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        for a_ in range(8):
            for b_ in range(a_ + 1, 8):
                if np.sum(np.abs(cs[a_] - cs[b_]) > 1e-9) == 1:
                    ET.SubElement(wb, "geom", type="capsule", size="0.02", contype="0", conaffinity="0", mass="0",
                                  rgba="0.85 0.10 0.10 1", fromto=" ".join(f"{x:.4f}" for x in (*cs[a_], *cs[b_])))
    ET.SubElement(wb, "light", name="sun", directional="true", castshadow="true", pos="0 0 4",
                  dir="0 0 -1", diffuse="0.5 0.5 0.5", specular="0.1 0.1 0.1")
    ET.SubElement(wb, "light", name="fill", directional="true", castshadow="false", pos="-2 2 3",
                  dir="0.5 -0.5 -0.6", diffuse="0.25 0.25 0.25", specular="0 0 0")

    xy, spin = rotor_layout(p)
    v = dict(contype="0", conaffinity="0", mass="0")
    for k in range(4):
        body = wb.find(f"body[@name='cf{k}']")
        box = body.find("geom")
        box.set("group", "3")              # the 70 x 70 x 20 mm collision box: hidden in the render
        if style == "mesh":
            # Crazyflow airframe, body origin on ours (its motor sites are at +-35.36 mm, ours at
            # +-35.355 mm: 5 um apart, so no scaling); faint discs keep the rotor planes and spin visible
            for mesh, mat, pos, euler in af["geoms"]:
                ET.SubElement(body, "geom", type="mesh", mesh=f"cf_{mesh}", material=f"cf_{mat}",
                              pos=pos, euler=euler, **v)
            for (rx, ry), s in zip(xy, spin):
                rgba = (C_CCW if s > 0 else C_CW).rsplit(" ", 1)[0] + " 0.22"
                ET.SubElement(body, "geom", type="cylinder", size=f"{p.prop_radius} 0.0003",
                              pos=f"{rx} {ry} {p.prop_z}", rgba=rgba, **v)
        else:
            ET.SubElement(body, "geom", type="box", size="0.016 0.011 0.006", rgba=C_FRAME, **v)
            for (rx, ry), s in zip(xy, spin):
                ET.SubElement(body, "geom", type="capsule", size="0.0025",
                              fromto=f"0 0 0 {rx} {ry} 0", rgba=C_FRAME, **v)
                ET.SubElement(body, "geom", type="cylinder", size="0.0045",
                              fromto=f"{rx} {ry} -0.004 {rx} {ry} {p.prop_z - 0.001}", rgba=C_FRAME, **v)
                ET.SubElement(body, "geom", type="cylinder", size=f"{p.prop_radius} 0.0008",
                              pos=f"{rx} {ry} {p.prop_z}", rgba=C_CCW if s > 0 else C_CW, **v)
            ET.SubElement(body, "geom", type="capsule", size="0.0012",
                          fromto=f"0 0 -0.006 0 0 {p.hook_z}", rgba=C_FRAME, **v)
        ET.SubElement(body, "geom", type="sphere", size="0.0035", pos=f"0 0 {p.hook_z}", rgba=C_HOOK, **v)
    for body in wb.findall("body"):
        if body.get("name", "").startswith("n"):
            body.find("geom").set("rgba", C_KNOT)
    for t in root.find("tendon"):
        # MuJoCo draws a slack length-limited tendon as a catenary hanging below its end points; for
        # links lying on the floor that curve goes through the floor. Drawn straight here (render
        # model only: the limit is what the simulation uses, this model is never stepped).
        t.set("limited", "false")
        if t.get("name").startswith("str"):
            t.set("width", "0.0022")
            t.set("rgba", C_STRING)
        else:
            t.set("width", "0.0014")
            t.set("rgba", C_NET)
    root.remove(root.find("sensor"))       # tendon limit-force sensors need limited tendons; not needed here
    return ET.tostring(root, encoding="unicode")


class Scene:
    def __init__(self, p, width, height, style="schematic", af=None, floor_half=None, box=None):
        self.p = p
        self.m = mujoco.MjModel.from_xml_string(visual_xml(p, width, height, style, af, floor_half, box))
        self.d = mujoco.MjData(self.m)
        m, n = self.m, p.n
        nid = lambda t, name: mujoco.mj_name2id(m, t, name)
        self.dq = [m.jnt_qposadr[m.body_jntadr[nid(mujoco.mjtObj.mjOBJ_BODY, f"cf{k}")]] for k in range(4)]
        kb = [nid(mujoco.mjtObj.mjOBJ_BODY, f"n{i}_{j}") for i in range(n) for j in range(n)]
        self.kq = np.array([m.jnt_qposadr[m.body_jntadr[b]] for b in kb])   # three slide joints x, y, z
        self.kpos0 = m.body_pos[kb].copy()
        self.sites = ([nid(mujoco.mjtObj.mjOBJ_SITE, f"s{i}_{j}") for i in range(n) for j in range(n)]
                      + [nid(mujoco.mjtObj.mjOBJ_SITE, f"hook{k}") for k in range(4)])

    def aligned_pos(self, frame, pos, quat):
        """Drone CoM positions moved onto the logged hook sites.

        net_sim logs site positions (knots, hooks) computed at the start of mj_step but drone qpos
        after it, so the logged drone pose is one time step (1 ms) ahead of the logged net: the hook
        rebuilt from it is off by |v| dt (up to 0.6 mm while climbing). Translating each drone by
        that step puts drones and net at the same instant; the attitude change in 1 ms is negligible.
        """
        R = ns.quat_to_mat(quat)
        return frame[self.p.n ** 2:] - R[:, :, 2] * self.p.hook_z

    def set_state(self, frame, pos, quat):
        q = self.d.qpos
        for k in range(4):
            q[self.dq[k]:self.dq[k] + 3] = pos[k]
            q[self.dq[k] + 3:self.dq[k] + 7] = quat[k]
        rel = frame[:self.p.n ** 2] - self.kpos0
        for a in range(3):
            q[self.kq + a] = rel[:, a]
        mujoco.mj_forward(self.m, self.d)
        err = np.linalg.norm(self.d.site_xpos[self.sites] - frame, axis=1)
        nk = self.p.n ** 2
        return err[:nk].max(), err[nk:].max()


# ---------------------------------------------------------------- shots
def state_numbers(p, frame, pos, quat):
    n, c = p.n, p.n // 2
    knots = frame[:n * n].reshape(n, n, 3)
    corners = np.array([knots[n - 1 if sx > 0 else 0, n - 1 if sy > 0 else 0] for sx, sy in ns.CORNERS])
    tilt, lean = [], []
    for k in range(4):
        R = ns.quat_to_mat(quat[k])
        tilt.append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
        out = np.array(ns.CORNERS[k], float) / np.sqrt(2)
        lean.append(float(R[:2, 2] @ out))          # > 0: thrust axis tips away from the net centre
    return dict(sag=round(float(corners[:, 2].mean() - knots[c, c, 2]), 4) + 0.0, tilt=float(np.mean(tilt)),
                outward=all(x > 0 for x in lean) if max(tilt) > 2 else None,
                z_drone=float(pos[:, 2].mean()), z_min=float(knots[..., 2].min()))


def camera(view, frame, pos, p, follow=False, box=None):
    """follow: centre the wide / top / side views on the formation (travel logs) instead of the origin;
    follow="3d" (mission logs) also follows the height. view "box": the whole mission box from outside."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    allp = np.vstack([frame, pos])
    zmid = 0.5 * (allp[:, 2].max() + allp[:, 2].min())
    fovy = 45.0
    cx, cy = (pos[:, 0].mean(), pos[:, 1].mean()) if follow else (0.0, 0.0)
    if view == "box":
        size, off = box
        cam.lookat[:] = (size / 2 - off[0], size / 2 - off[1], size / 2 - off[2])
        cam.distance, cam.azimuth, cam.elevation = 2.3 * size, -60.0, -22.0
    elif view == "wide" and follow == "3d":
        cam.lookat[:] = (cx, cy, zmid - 0.1)
        cam.distance, cam.azimuth, cam.elevation = 3.0, -60.0, -22.0
    elif view == "wide":
        cam.lookat[:] = (cx, cy, max(0.6 * zmid, 0.25))  # between the scene and its shadow
        cam.distance, cam.azimuth, cam.elevation = 3.0, -60.0, -22.0
    elif view == "closeup":                  # drone 0 and its string, seen across the diagonal
        corner = frame[p.n * p.n - 1]
        cam.lookat[:] = 0.6 * pos[0] + 0.4 * corner
        cam.distance, cam.azimuth, cam.elevation = 0.42, -45.0, -12.0
    elif view == "drone":                    # tight view of drone 0 alone: airframe, hook, start of the string
        cam.lookat[:] = pos[0] + np.array([0.0, 0.0, -0.005])
        cam.distance, cam.azimuth, cam.elevation = 0.24, -45.0, -18.0
    elif view == "top":
        cam.lookat[:] = (cx, cy, zmid)
        cam.orthographic, fovy = 1, 1.75     # orthographic: fovy = visible height [m]
        cam.distance, cam.azimuth, cam.elevation = 4.0, 90.0, -90.0
    elif view == "side":                     # looking along +y: the x-z plane
        cam.lookat[:] = (cx, cy, zmid)
        cam.orthographic, fovy = 1, (1.6 if follow == "3d" else 1.15)
        cam.distance, cam.azimuth, cam.elevation = 4.0, 90.0, 0.0
    else:
        raise ValueError(view)
    return cam, fovy


def caption(img, lines):
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    y = 10
    for ln in lines:
        dr.text((14, y), ln, fill=(20, 20, 20), font=font, stroke_width=3, stroke_fill=(255, 255, 255))
        y += 28
    return im


def default_shots(log):
    t = log["frames_t"]
    fr = log["frames"]
    z = fr[:, :81, 2]
    up = (z > 0.012).mean(axis=1)                          # share of knots off the floor
    i_half = int(np.argmin(np.abs(up - 0.5) + 10 * (t < 0.5) + 10 * (t > 6)))
    s = (log["t"] >= 22.0) & (log["t"] < 26.0)
    e0 = np.linalg.norm(log["pos"][s, 0] - log["ref"][s, 0], axis=1)
    t_gust = float(log["t"][s][np.argmax(e0)])
    return [("01_ground", 0.25, "wide", "on the ground before takeoff"),
            ("02_takeoff", float(t[i_half]), "wide", f"takeoff, {up[i_half] * 100:.0f} % of knots off the floor"),
            ("03_hold", 20.0, "wide", "steady hold, D = 1.25 m"),
            ("04_gust_peak", t_gust, "wide", "gust on drone 0 (left): peak displacement"),
            ("05_descent", 34.5, "wide", "mid-descent"),
            ("06_landed", 39.5, "wide", "landed, motors off"),
            ("07_hold_closeup", 20.0, "closeup", "hold: drone 0 (+x,+y), hook and string"),
            ("07b_hold_drone", 20.0, "drone", "hold: drone 0 alone, magenta = string hook"),
            ("08_hold_top", 20.0, "top", "hold: top view (orthographic), +x right, +y up"),
            ("09_hold_side", 20.0, "side", "hold: side view along +y (orthographic), +x right")]


def travel_shots(summary):
    """Default shots of a travel log: steady cruise (middle of its last 3 s), wide, side and top."""
    t0, t1 = summary["steady"]["window_s"] if summary.get("steady") else summary["phases"]["cruise"]["t_s"]
    t = 0.5 * (t0 + t1)
    c = summary["cfg"]
    what = f"cruise {c['speed']:g} m/s, heading {summary['heading_deg']:g} deg ({c['heading']}), a_max {c['accel']:g} m/s^2"
    side = ", side view along +y, motion along +x" if summary["heading_deg"] == 0 else ", view along +y"
    return [("travel_01_cruise_wide", t, "wide", what),
            ("travel_02_cruise_side", t, "side", what + side),
            ("travel_03_cruise_top", t, "top", what + ", top view, +x right")]


# ---------------------------------------------------------------- mission video
def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def interp_state(log, t):
    """Knots, hooks, drone positions and attitudes at time t, linear between the 20 Hz logged frames
    (attitudes: normalised linear blend of the quaternions; adjacent samples are 50 ms apart)."""
    ft = log["frames_t"]
    i = int(np.clip(np.searchsorted(ft, t) - 1, 0, len(ft) - 2))
    w = float(np.clip((t - ft[i]) / (ft[i + 1] - ft[i]), 0.0, 1.0))
    fr = (1 - w) * log["frames"][i] + w * log["frames"][i + 1]
    pos = (1 - w) * log["frames_pos"][i] + w * log["frames_pos"][i + 1]
    q0, q1 = log["frames_quat"][i], log["frames_quat"][i + 1].copy()
    q1[np.sum(q0 * q1, axis=1) < 0] *= -1
    q = (1 - w) * q0 + w * q1
    return fr, pos, q / np.linalg.norm(q, axis=1, keepdims=True)


def mission_status(summary, t):
    """(leg index or None, status text, colour) of a mission at time t."""
    legs = summary["legs"]
    if t < legs[0]["t_dep"]:
        return None, "HOVER after takeoff (net spread, waiting to depart)", (90, 90, 90)
    k = max(i for i, r in enumerate(legs) if r["t_dep"] <= t + 1e-9)
    r = legs[k]
    if t < r["t_ref_stop"]:
        return k, "MOVING", (20, 90, 200)
    if r["t_settled"] is None or t < r["t_settled"]:
        return k, "SETTLING: waiting to arrive (50 / 50 / 20 mm)", (200, 110, 0)
    end = r["t_settled"] + 2.0
    if t < end:
        return k, "ARRIVED: confirming for 2 s", (0, 140, 60)
    fs = summary.get("final_strict") or {}
    ts = fs.get("t_strict_settled")
    if ts is not None and t >= ts:
        return k, "LAST TARGET: strict 10 / 10 / 5 mm rule met, confirming", (0, 120, 120)
    return k, "LAST TARGET reached: hovering until the strict 10 / 10 / 5 mm rule is met", (0, 120, 120)


def video_shots(summary, t_end):
    """Moments saved as PNG from the video: takeoff hover, fastest level leg, an oblique climb, an oblique
    descent, the longest settling wait, the last frame."""
    legs = summary["legs"]
    lvl = [r for r in legs if abs(r["elevation_deg"]) < 20] or legs
    fast = max(lvl, key=lambda r: r["peak_speed_m_s"])
    up = [r for r in legs if 20 <= r["elevation_deg"] <= 70] or legs
    dn = [r for r in legs if -70 <= r["elevation_deg"] <= -20] or legs
    cl, ds = max(up, key=lambda r: r["L"]), max(dn, key=lambda r: r["L"])
    st = max(legs, key=lambda r: r["settle_time_s"] or 0.0)
    mid = lambda r: 0.5 * (r["t_dep"] + r["t_ref_stop"])
    return [("frame_01_takeoff_hover", legs[0]["t_dep"] - 0.5),
            (f"frame_02_fast_level_leg{fast['leg'] + 1}", mid(fast)),
            (f"frame_03_oblique_climb_leg{cl['leg'] + 1}", mid(cl)),
            (f"frame_04_oblique_descent_leg{ds['leg'] + 1}", mid(ds)),
            (f"frame_05_settling_leg{st['leg'] + 1}", 0.5 * (st["t_ref_stop"] + st["t_settled"])),
            ("frame_06_last_target", t_end)]


def _project(glcam, pts, aspect):
    """Pixel coordinates (x right, y down, fraction of the image) of world points for a mjvGLCamera."""
    pos, fwd, up = np.array(glcam.pos), np.array(glcam.forward), np.array(glcam.up)
    right = np.cross(fwd, up)
    v = np.asarray(pts, float) - pos
    z = v @ fwd
    x, y = (v @ right) * glcam.frustum_near / z, (v @ up) * glcam.frustum_near / z
    hw = glcam.frustum_width or 0.5 * (glcam.frustum_top - glcam.frustum_bottom) * aspect
    u = (x - (glcam.frustum_center - hw)) / (2 * hw)
    w = (glcam.frustum_top - y) / (glcam.frustum_top - glcam.frustum_bottom)
    return np.stack([u, w], axis=1), z > 0


def _add_geom(scn, gtype, size, pos, rgba, mat=None):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                        np.eye(3).flatten() if mat is None else mat, np.asarray(rgba, np.float32))
    scn.ngeom += 1


def _add_segment(scn, a, b, radius, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3).flatten(),
                        np.asarray(rgba, np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(a, float), np.asarray(b, float))
    scn.ngeom += 1


def render_mission_video(log, summary, p, af, style, out, fps=30.0, size=(1280, 720), t0=None, t1=None,
                         crf=20, every_check=1):
    """Two side-by-side views of a mission log, H.264 mp4 through PyAV (libx264): left, the whole box with
    the targets (current one highlighted, reached ones green), the net-centre path flown so far; right, a
    camera following the formation. Real time (1x) at `fps`; scene states between the 20 Hz log frames are
    interpolated. The window check runs at every frame (every_check)."""
    import av
    W, H = size
    Wp = W // 2
    box = (summary["box"]["size_m"], np.array(summary["box"]["offset_m"], float))
    off = box[1]
    legs = summary["legs"]
    T_net = np.array(summary["targets_box"]) - off                       # net-centre targets, sim coordinates
    t0 = legs[0]["t_dep"] - 1.5 if t0 is None else t0
    t1 = float(log["frames_t"][-1]) if t1 is None else min(t1, float(log["frames_t"][-1]))
    times = t0 + np.arange(int(np.floor((t1 - t0) * fps + 1e-9)) + 1) / fps
    shots = {name: int(np.argmin(np.abs(times - t))) for name, t in video_shots(summary, times[-1])}
    shot_at = {}
    for name, i in shots.items():
        shot_at.setdefault(i, []).append(name)
    # net-centre path and formation speed from the 100 Hz log
    lt = log["t"]
    vel = np.linalg.norm(log["vel"].mean(axis=1), axis=1)
    path_t = log["frames_t"]
    path = np.array([interp_state(log, t)[0][:p.n * p.n].reshape(p.n, p.n, 3)[p.n // 2, p.n // 2] for t in path_t])
    sc = Scene(p, Wp, H, style, af, floor_half=box[0] / 2 + 6.0, box=(box[0], off.tolist()))
    sun = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
    os.makedirs(out, exist_ok=True)
    stem = os.path.basename(summary.get("_log_path", "mission")).replace("_log.npz", "")
    mp4 = os.path.join(out, f"{stem}_video.mp4")
    container = av.open(mp4, mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "medium", "movflags": "+faststart"}
    f_big, f_mid, f_small = _font(22), _font(18), _font(15)
    guard = WindowGuard()
    print(f"video: {len(times)} frames, t = {times[0]:.2f} .. {times[-1]:.2f} s, {fps:g} fps, {W} x {H}; "
          f"windows of this process before the renderer: {len(own_windows())}", flush=True)
    wall0 = time.time()
    renderer = mujoco.Renderer(sc.m, height=H, width=Wp, max_geom=20000)
    saved = []
    try:
        guard(renderer, "after creating the renderer")
        w = own_windows()
        print(f"after creating the renderer: {len(w)} window(s) of this process, visible: "
              f"{sum(x[3] for x in w)}  {[(c, t) for _, c, t, _ in w]}", flush=True)
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        opt.sitegroup[:] = 0
        for fi, t in enumerate(times):
            fr, pos, quat = interp_state(log, t)
            pos = sc.aligned_pos(fr, pos, quat)
            sc.set_state(fr, pos, quat)
            fc = pos.mean(axis=0)
            nc = fr[:p.n * p.n].reshape(p.n, p.n, 3)[p.n // 2, p.n // 2]
            k, status, col = mission_status(summary, t)
            reached = [i for i, r in enumerate(legs) if r["t_settled"] is not None and t >= r["t_settled"]]
            cur = k if k is not None else 0
            # left: the box
            cam, _ = camera("box", fr, pos, p, box=(box[0], off))
            cam.distance = 2.55 * box[0]
            cam.lookat[2] = 0.42 * box[0] - off[2]
            sc.m.vis.global_.fovy, sc.m.vis.global_.orthographic = 45.0, 0
            sc.m.light_pos[sun] = (fc[0], fc[1], max(4.0, fc[2] + 3.0))
            renderer.update_scene(sc.d, camera=cam, scene_option=opt)
            scn = renderer.scene
            sphere = mujoco.mjtGeom.mjGEOM_SPHERE
            for i, x in enumerate(T_net):
                if i == cur and i not in reached:
                    _add_geom(scn, sphere, [0.22, 0, 0], x, [0.95, 0.10, 0.10, 1.0])
                elif i in reached:
                    _add_geom(scn, sphere, [0.13, 0, 0], x, [0.10, 0.70, 0.25, 1.0])
                else:
                    _add_geom(scn, sphere, [0.13, 0, 0], x, [0.55, 0.55, 0.60, 0.9])
                _add_segment(scn, x, [x[0], x[1], 0.0], 0.012, [0.5, 0.5, 0.55, 0.35])
            j = int(np.searchsorted(path_t, t))
            pts = np.vstack([path[:j:2], nc[None]])
            for a_, b_ in zip(pts[:-1], pts[1:]):
                if np.linalg.norm(b_ - a_) > 1e-4:
                    _add_segment(scn, a_, b_, 0.035, [0.10, 0.35, 0.90, 1.0])
            _add_segment(scn, nc, [nc[0], nc[1], 0.0], 0.01, [0.10, 0.35, 0.90, 0.5])   # drop line to the floor
            left = renderer.render().copy()
            glc = scn.camera[0]
            uv, front = _project(glc, T_net, Wp / H)
            # right: follow camera
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = 0.5 * (fc + nc) + np.array([0.0, 0.0, 0.05])
            cam.distance, cam.azimuth, cam.elevation = 2.7, -60.0, -20.0
            renderer.update_scene(sc.d, camera=cam, scene_option=opt)
            right = renderer.render().copy()
            if fi % every_check == 0 or fi == len(times) - 1:
                guard(renderer, f"after rendering frame {fi} (t = {t:.2f} s)")
            # overlays
            im = Image.fromarray(np.hstack([left, right]))
            dr = ImageDraw.Draw(im)
            txt = dict(fill=(20, 20, 20), stroke_width=3, stroke_fill=(255, 255, 255))
            for i, ((u, v_), ok) in enumerate(zip(uv, front)):
                if ok:
                    dr.text((u * Wp + 9, v_ * H - 22), str(i + 1), font=f_small, **txt)
            speed = float(np.interp(t, lt, vel))
            leg_txt = "before leg 1" if k is None else f"leg {k + 1} / {len(legs)}"
            dr.text((12, 10), f"t = {t:6.2f} s    {leg_txt}", font=f_big, **txt)
            dr.text((12, 40), f"speed {speed:4.2f} m/s", font=f_big, **txt)
            dr.text((12, 70), status, font=f_mid, fill=col, stroke_width=3, stroke_fill=(255, 255, 255))
            dr.text((12, H - 50), "10 x 10 x 10 m box (red edges)    blue: net-centre path flown", font=f_small, **txt)
            dr.text((12, H - 28), "targets (net centre): red = current, green = reached, grey = later", font=f_small, **txt)
            s = state_numbers(p, fr, pos, quat)
            tgt = T_net[cur] + off
            dr.text((Wp + 12, 10), "follow camera (1x real time)", font=f_big, **txt)
            dr.text((Wp + 12, 40), f"net sag {s['sag']:.3f} m   mean tilt {s['tilt']:4.1f} deg", font=f_mid, **txt)
            dr.text((Wp + 12, 64), f"net centre ({nc[0] + off[0]:.2f}, {nc[1] + off[1]:.2f}, {nc[2] + off[2]:.2f}) m",
                    font=f_mid, **txt)
            dr.text((Wp + 12, 88), f"target {cur + 1}   ({tgt[0]:.2f}, {tgt[1]:.2f}, {tgt[2]:.2f}) m", font=f_mid, **txt)
            dr.text((Wp + 12, H - 28), "Crazyflie 2.1 Brushless meshes; orange = corner strings", font=f_small, **txt)
            dr.line([(Wp, 0), (Wp, H)], fill=(40, 40, 40), width=2)
            img = np.asarray(im)
            for name in shot_at.get(fi, []):
                path_png = os.path.join(out, f"{name}.png")
                im.save(path_png)
                saved.append((path_png, float(t)))
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
            if fi % 150 == 0 or fi == len(times) - 1:
                el = time.time() - wall0
                print(f"  frame {fi + 1} / {len(times)}  t = {t:6.2f} s  {el:6.1f} s elapsed, "
                      f"~{el / (fi + 1) * (len(times) - fi - 1):5.0f} s left", flush=True)
        guard(renderer, "before closing the renderer")
    finally:
        renderer.close()
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    wall = time.time() - wall0
    print(f"window check: {guard.checks} checks while the renderer existed (every frame), at most "
          f"{guard.max_windows} window(s) of this process, none visible; classes/titles seen: {sorted(guard.seen)}")
    print(f"after close: {len(own_windows())} window(s) of this process")
    mb = os.path.getsize(mp4) / 1e6
    print(f"video: {mp4}\n  {len(times)} frames, {len(times) / fps:.2f} s at {fps:g} fps, {W} x {H}, H.264 "
          f"(libx264 via PyAV, crf {crf}, yuv420p), {mb:.1f} MB; render + encode {wall:.0f} s")
    for path_png, t in saved:
        print(f"  frame {path_png}  t = {t:.2f} s")
    return mp4


# ---------------------------------------------------------------- catch video (net_sim --mode catch logs)
BALL_RGB = [(0.95, 0.45, 0.05), (0.10, 0.55, 0.95), (0.85, 0.15, 0.55), (0.15, 0.70, 0.25), (0.55, 0.35, 0.85),
            (0.95, 0.80, 0.10), (0.40, 0.40, 0.40), (0.55, 0.30, 0.12), (0.05, 0.75, 0.75), (0.10, 0.10, 0.45)]
REASON_TXT = {"late": "missed: not there in time", "offset": "missed: there but off target",
              "fell_out": "in the net, then fell out", "hit_drone": "hit a drone", "crowded_out":
              "missed: dropped for another ball", "trial_aborted": "trial aborted"}


def ball_status(summary, b, t):
    """(text, rgb) of ball b at time t, from the trial's per-ball outcome."""
    r = summary["balls"][b]
    if t < r["t_launch"]:
        return "waiting", (120, 120, 120)
    fc = r.get("t_first_contact")
    tc = r.get("t_cross_rotor")
    if r["outcome"] == "caught":
        if fc is not None and t >= fc:
            return "IN THE NET", (0, 140, 60)
        pile = r.get("caught_on_pile")
        if pile is not None and t >= pile["t_first_support"]:
            return "IN THE NET (on the other balls)", (0, 140, 60)
        return "in flight", (20, 90, 200)
    reason = r.get("reason")
    loss = r.get("loss")
    if reason == "fell_out" and loss is not None and fc is not None and t >= fc:
        txt = {"knocked_out": "KNOCKED OUT", "thrown_out": "THROWN OUT", "bounced_out": "BOUNCED OUT"}[loss["kind"]]
        return "in the net ... " + (txt if t >= loss["t_out_s"] else ""), (200, 110, 0)
    if reason == "fell_out" and fc is not None and t >= fc:
        return "in the net ... " + ("FELL OUT" if t >= fc + r["net_contact_s"] else ""), (200, 110, 0)
    hit = r.get("shell_hit_first")
    if reason == "hit_drone" and hit and t >= hit["t_s"]:
        return "HIT DRONE " + str(hit["drone"]), (200, 20, 20)
    t_miss = tc if tc is not None else r["free_flight"]["t_floor_s"] + r["t_launch"]
    if t >= t_miss:
        return REASON_TXT.get(reason, "missed").upper(), (200, 20, 20)
    return "in flight", (20, 90, 200)


def catch_shots(summary, t_end):
    """Key moments: before the first launch, the first ball mid-flight, the first catch, the first miss (or a second
    catch), the end."""
    balls = [r for r in summary["balls"] if r["outcome"] != "not_launched"]
    t0 = min(r["t_launch"] for r in balls)
    shots = [("frame_01_before_launch", t0 - 0.3)]
    r0 = balls[0]
    tc0 = r0.get("t_cross_rotor") or (r0["t_launch"] + 1.0)
    shots.append(("frame_02_ball1_in_flight", 0.5 * (r0["t_launch"] + tc0)))
    caught = [r for r in balls if r["outcome"] == "caught" and r.get("t_first_contact")]
    if caught:
        r = caught[0]
        shots.append((f"frame_03_catch_ball{r['ball'] + 1}", r["t_first_contact"] + 0.05))
    missed = [r for r in balls if r["outcome"] != "caught"]
    if missed:
        r = missed[0]
        tm = r.get("t_cross_rotor") or r["t_launch"] + r["free_flight"]["t_floor_s"] - 0.3
        shots.append((f"frame_04_miss_ball{r['ball'] + 1}_{r.get('reason')}", tm + 0.1))
    elif len(caught) > 1:
        r = caught[-1]
        shots.append((f"frame_04_catch_ball{r['ball'] + 1}", r["t_first_contact"] + 0.05))
    shots.append(("frame_05_end", t_end))
    return shots


def render_catch_video(log, summary, p, af, style, out, fps=30.0, size=(1280, 720), t0=None, t1=None, crf=20):
    """Two views of a catch trial, H.264 mp4: left, the launcher and the catch area seen from the side (balls drawn
    3x enlarged, with 0.4 s trails, each ball's predicted path and its predicted crossing of the current net-centre
    height from the logged EKF state); right, a camera following the formation (balls at true size). 1x real time,
    window check at every frame; PNG frames at the key moments (catch_shots)."""
    import av
    W, H = size
    Wp = W // 2
    box = (summary["box"]["size_m"], np.array(summary["box"]["offset_m"], float))
    off = box[1]
    balls = summary["balls"]
    nb = len(balls)
    t_l0 = min(r["t_launch"] for r in balls if r["outcome"] != "not_launched")
    t0 = t_l0 - 1.5 if t0 is None else t0
    t1 = float(log["frames_t"][-1]) if t1 is None else min(t1, float(log["frames_t"][-1]))
    times = t0 + np.arange(int(np.floor((t1 - t0) * fps + 1e-9)) + 1) / fps
    shot_at = {}
    for name, t in catch_shots(summary, times[-1]):
        shot_at.setdefault(int(np.argmin(np.abs(times - t))), []).append(name)
    lt = log["t"]
    bp, best = log["ball_pos"], log["ball_est"]
    k_drag = summary["ball"]["k_1_m"]
    r_b = summary["ball"]["d_m"] / 2
    L0 = np.array(summary["launcher_box_m"], float) - off
    sb = np.array(summary["standby_box"], float) - off
    sc = Scene(p, Wp, H, style, af, floor_half=box[0] / 2 + 6.0, box=(box[0], off.tolist()))
    sun = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
    os.makedirs(out, exist_ok=True)
    stem = os.path.basename(summary.get("_log_path", "catch")).replace("_log.npz", "")
    mp4 = os.path.join(out, f"{stem}_video.mp4")
    container = av.open(mp4, mode="w")
    stream = container.add_stream("libx264", rate=int(round(fps)))
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.options = {"crf": str(crf), "preset": "medium", "movflags": "+faststart"}
    f_big, f_mid, f_small = _font(22), _font(18), _font(15)
    guard = WindowGuard()
    print(f"video: {len(times)} frames, t = {times[0]:.2f} .. {times[-1]:.2f} s, {fps:g} fps, {W} x {H}; "
          f"windows of this process before the renderer: {len(own_windows())}", flush=True)
    wall0 = time.time()
    renderer = mujoco.Renderer(sc.m, height=H, width=Wp, max_geom=20000)
    saved = []
    sphere = mujoco.mjtGeom.mjGEOM_SPHERE
    boxg = mujoco.mjtGeom.mjGEOM_BOX
    try:
        guard(renderer, "after creating the renderer")
        w = own_windows()
        print(f"after creating the renderer: {len(w)} window(s) of this process, visible: "
              f"{sum(x[3] for x in w)}  {[(c, t) for _, c, t, _ in w]}", flush=True)
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0
        opt.sitegroup[:] = 0
        for fi, t in enumerate(times):
            fr, pos, quat = interp_state(log, t)
            pos = sc.aligned_pos(fr, pos, quat)
            sc.set_state(fr, pos, quat)
            fc = pos.mean(axis=0)
            nc = fr[:p.n * p.n].reshape(p.n, p.n, 3)[p.n // 2, p.n // 2]
            i = int(np.clip(np.searchsorted(lt, t), 1, len(lt) - 1))
            w_ = float(np.clip((t - lt[i - 1]) / (lt[i] - lt[i - 1]), 0, 1))
            bnow = (1 - w_) * bp[i - 1] + w_ * bp[i]
            preds = {}
            for b in range(nb):
                st, _ = ball_status(summary, b, t)
                e = best[i - 1, b]
                if st == "in flight" and np.isfinite(e).all():
                    tr = ns.flight_rk4(e, k_drag, dt=0.02, n_max=250, z_stop=0.02)
                    c = ns.crossing(tr, float(nc[2]))
                    preds[b] = (tr, c)

            def decorate(scn, big):
                rad = r_b * (3.0 if big else 1.0)
                _add_geom(scn, boxg, [0.10, 0.20, 0.15], L0 - np.array([0, 0, 0.15]), [0.25, 0.25, 0.28, 1.0])
                for b in range(nb):
                    col = list(BALL_RGB[b % len(BALL_RGB)])
                    x = bnow[b]
                    if np.isfinite(x).all() and t >= balls[b]["t_launch"]:
                        _add_geom(scn, sphere, [rad, 0, 0], x, col + [1.0])
                        j0 = max(0, i - 40)
                        tail = bp[j0:i, b]
                        tail = tail[np.isfinite(tail).all(axis=1)][::4]
                        for a_, b_ in zip(tail[:-1], tail[1:]):
                            if np.linalg.norm(b_ - a_) > 1e-3:
                                _add_segment(scn, a_, b_, rad * 0.35, col + [0.6])
                    if b in preds:
                        tr, c = preds[b]
                        for q in tr[::3, 1:4]:
                            _add_geom(scn, sphere, [rad * 0.3, 0, 0], q, col + [0.35])
                        if c is not None:
                            mk = np.array([0.10 if big else 0.05, 0, 0])
                            _add_geom(scn, sphere, mk, c[1:4], col + [0.55])
                            _add_segment(scn, c[1:4], [c[1], c[2], 0.0], 0.006, col + [0.35])
            # left: side view of the launcher and the catch area
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            mid = 0.5 * (L0 + sb)
            cam.lookat[:] = (mid[0] + 0.6, mid[1], 2.6)
            cam.distance, cam.azimuth, cam.elevation = 9.5, 90.0, -10.0      # from -y: +x to the right
            sc.m.vis.global_.fovy, sc.m.vis.global_.orthographic = 45.0, 0
            sc.m.light_pos[sun] = (fc[0], fc[1], max(4.0, fc[2] + 3.0))
            renderer.update_scene(sc.d, camera=cam, scene_option=opt)
            decorate(renderer.scene, True)
            left = renderer.render().copy()
            # right: follow camera
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            cam.lookat[:] = 0.5 * (fc + nc) + np.array([0.0, 0.0, 0.15])
            cam.distance, cam.azimuth, cam.elevation = 3.4, -70.0, -18.0
            renderer.update_scene(sc.d, camera=cam, scene_option=opt)
            decorate(renderer.scene, False)
            right = renderer.render().copy()
            guard(renderer, f"after rendering frame {fi} (t = {t:.2f} s)")
            im = Image.fromarray(np.hstack([left, right]))
            dr = ImageDraw.Draw(im)
            txt = dict(fill=(20, 20, 20), stroke_width=3, stroke_fill=(255, 255, 255))
            n_in = sum(ball_status(summary, b, t)[0].startswith("IN THE NET") for b in range(nb))
            n_l = sum(t >= r["t_launch"] for r in balls)
            dr.text((12, 10), f"t = {t:6.2f} s    launched {n_l} / {nb}    in the net {n_in}", font=f_big, **txt)
            cfg = summary["cfg"]
            dr.text((12, 40), f"tier {cfg.get('tier')}, {cfg.get('launch')}"
                    + (f", interval {cfg.get('interval_s'):g} s" if cfg.get("launch") == "sequence" else "")
                    + f", noise {1e3 * cfg.get('noise_m', 0):g} mm", font=f_mid, **txt)
            for b in range(nb):
                st, col = ball_status(summary, b, t)
                rgb = tuple(int(255 * c) for c in BALL_RGB[b % len(BALL_RGB)])
                dr.ellipse([14, 72 + 24 * b, 30, 88 + 24 * b], fill=rgb)
                dr.text((38, 68 + 24 * b), f"ball {b + 1}: {st}", font=f_mid, fill=col, stroke_width=3,
                        stroke_fill=(255, 255, 255))
            dr.text((12, H - 50), "balls drawn 3x enlarged; dots: predicted path (EKF), big dot: predicted", font=f_small,
                    **txt)
            dr.text((12, H - 28), "crossing of the current net-centre height; grey block: launcher", font=f_small, **txt)
            s = state_numbers(p, fr, pos, quat)
            dr.text((Wp + 12, 10), "follow camera (1x real time), balls true size", font=f_big, **txt)
            dr.text((Wp + 12, 40), f"net sag {s['sag']:.3f} m   mean tilt {s['tilt']:4.1f} deg   drones z "
                    f"{s['z_drone']:.2f} m", font=f_mid, **txt)
            speed = float(np.interp(t, lt, np.linalg.norm(log["vel"].mean(axis=1), axis=1)))
            dr.text((Wp + 12, 64), f"formation speed {speed:4.2f} m/s", font=f_mid, **txt)
            dr.text((Wp + 12, H - 28), "Crazyflie 2.1 Brushless meshes; orange = corner strings", font=f_small, **txt)
            dr.line([(Wp, 0), (Wp, H)], fill=(40, 40, 40), width=2)
            img = np.asarray(im)
            for name in shot_at.get(fi, []):
                path_png = os.path.join(out, f"{stem}_{name}.png")
                im.save(path_png)
                saved.append((path_png, float(t)))
            for packet in stream.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                container.mux(packet)
            if fi % 150 == 0 or fi == len(times) - 1:
                el = time.time() - wall0
                print(f"  frame {fi + 1} / {len(times)}  t = {t:6.2f} s  {el:6.1f} s elapsed", flush=True)
        guard(renderer, "before closing the renderer")
    finally:
        renderer.close()
        for packet in stream.encode():
            container.mux(packet)
        container.close()
    print(f"window check: {guard.checks} checks while the renderer existed (every frame), at most "
          f"{guard.max_windows} window(s) of this process, none visible; classes/titles seen: {sorted(guard.seen)}")
    print(f"after close: {len(own_windows())} window(s) of this process")
    print(f"video: {mp4}  {len(times)} frames, {len(times) / fps:.2f} s, {os.path.getsize(mp4) / 1e6:.1f} MB, "
          f"render + encode {time.time() - wall0:.0f} s")
    for path_png, t in saved:
        print(f"  frame {path_png}  t = {t:.2f} s")
    return mp4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=os.path.join(HERE, "out_official", "hold_ff_log.npz"))
    ap.add_argument("--out", default=None, help="output folder (default: screenshots/ next to the log)")
    ap.add_argument("--times", type=float, nargs="*", default=None, help="render these times instead of the default set")
    ap.add_argument("--view", default="wide", choices=["wide", "closeup", "drone", "top", "side"], help="camera for --times")
    ap.add_argument("--size", default="1280x960")
    have_mesh = os.path.isfile(os.path.join(MESH_DIR, _MESH_XML))
    ap.add_argument("--style", default="mesh" if have_mesh else "schematic", choices=["mesh", "schematic"],
                    help="drone drawing: Crazyflow airframe meshes (default when assets/crazyflow_cf21B exists) "
                         "or the schematic hub, arms and rotor discs")
    ap.add_argument("--video", action="store_true",
                    help="mission logs: an mp4 (H.264) of the whole mission instead of screenshots, two views side "
                         "by side (the box with targets and path; a camera following the formation), 1x real time; "
                         "output video/ next to the log, plus 6 PNG frames from the video")
    ap.add_argument("--fps", type=float, default=30.0, help="video frame rate (default 30)")
    ap.add_argument("--video-size", default="1280x720", help="video size, both views together (default 1280x720)")
    ap.add_argument("--t0", type=float, default=None, help="video start in s (default 1.5 s before the first leg)")
    ap.add_argument("--t1", type=float, default=None, help="video end in s (default: end of the log)")
    ap.add_argument("--crf", type=int, default=20, help="H.264 quality, lower = better / larger (default 20)")
    a = ap.parse_args()
    if a.video:
        a.size = a.video_size
    W, H = (int(x) for x in a.size.lower().split("x"))
    log = dict(np.load(a.log))
    with open(a.log.replace("_log.npz", "_summary.json")) as f:
        summary = json.load(f)
    p = load_params(summary)
    if summary.get("mode") == "catch":            # catch logs: always the video (two views) with its key-frame PNGs
        W, H = (int(x) for x in a.video_size.lower().split("x"))
        af = load_airframe() if a.style == "mesh" else None
        summary["_log_path"] = a.log
        render_catch_video(log, summary, p, af, a.style,
                           a.out or os.path.join(os.path.dirname(os.path.abspath(a.log)), "video"),
                           fps=a.fps, size=(W, H), t0=a.t0, t1=a.t1, crf=a.crf)
        return
    travel = summary.get("mode") in ("travel", "transfer", "mission")
    mission = summary.get("mode") == "mission"
    box = (summary["box"]["size_m"], summary["box"]["offset_m"]) if mission and summary.get("box") else None
    reach = None
    if travel:
        fp = log["frames_pos"]
        reach = float(np.abs(fp[:, :, :2]).max()) + 6.0
        print(f"travel log: the formation reaches {reach - 6.0:.1f} m from the origin; camera, floor "
              f"(half side {reach:.0f} m) and sun follow it")
    sub = "screenshots_mesh" if a.style == "mesh" else "screenshots"
    out = a.out or os.path.join(os.path.dirname(os.path.abspath(a.log)), sub)
    os.makedirs(out, exist_ok=True)

    af = None
    if a.style == "mesh":
        af = load_airframe()
        parts = airframe_geometry(af)
        lo = np.min([v[0] for v in parts.values()], axis=0)
        hi = np.max([v[1] for v in parts.values()], axis=0)
        print(f"airframe meshes ({len(af['geoms'])} geoms) from {MESH_DIR}")
        print("  motor sites [mm]: " + ", ".join(f"{k} ({v[0] * 1e3:+.2f}, {v[1] * 1e3:+.2f}, {v[2] * 1e3:+.2f})"
                                               for k, v in af["motors"].items()))
        print(f"  bounding box [mm]: x {lo[0] * 1e3:+.1f}..{hi[0] * 1e3:+.1f}  y {lo[1] * 1e3:+.1f}..{hi[1] * 1e3:+.1f}"
              f"  z {lo[2] * 1e3:+.1f}..{hi[2] * 1e3:+.1f}  (size {(hi - lo)[0] * 1e3:.1f} x {(hi - lo)[1] * 1e3:.1f}"
              f" x {(hi - lo)[2] * 1e3:.1f})")
        for name, (l_, h_) in sorted(parts.items(), key=lambda kv: kv[1][0][2]):
            print(f"    {name:15s} x {l_[0] * 1e3:+6.1f}..{h_[0] * 1e3:+6.1f}  y {l_[1] * 1e3:+6.1f}..{h_[1] * 1e3:+6.1f}"
                  f"  z {l_[2] * 1e3:+6.1f}..{h_[2] * 1e3:+6.1f} mm")
        print(f"  hook at z = {p.hook_z * 1e3:+.1f} mm; lowest airframe part reaches z = {lo[2] * 1e3:+.1f} mm")

    sc = Scene(p, W, H, a.style, af, floor_half=reach, box=box)
    sun = mujoco.mj_name2id(sc.m, mujoco.mjtObj.mjOBJ_LIGHT, "sun")
    # reconstruction check over every logged frame
    ek = eh = eh_raw = shift = 0.0
    for i in range(len(log["frames_t"])):
        fr, pos, quat = log["frames"][i], log["frames_pos"][i], log["frames_quat"][i]
        eh_raw = max(eh_raw, sc.set_state(fr, pos, quat)[1])
        pa = sc.aligned_pos(fr, pos, quat)
        shift = max(shift, np.linalg.norm(pa - pos, axis=1).max())
        a_, b_ = sc.set_state(fr, pa, quat)
        ek, eh = max(ek, a_), max(eh, b_)
    print(f"reconstruction over {len(log['frames_t'])} frames: worst knot mismatch {ek * 1e3:.2e} mm, "
          f"worst hook mismatch {eh * 1e3:.2e} mm (drones moved by at most {shift * 1e3:.3f} mm onto the "
          f"logged hooks; with the logged drone qpos as is: {eh_raw * 1e3:.3f} mm, the 1 ms log offset)")

    if a.video:
        if not mission:
            raise SystemExit("--video needs a mission log (summary mode 'mission')")
        summary["_log_path"] = a.log
        render_mission_video(log, summary, p, af, a.style,
                             a.out or os.path.join(os.path.dirname(os.path.abspath(a.log)), "video"),
                             fps=a.fps, size=(W, H), t0=a.t0, t1=a.t1, crf=a.crf)
        return
    if a.times is not None:
        shots = [(f"t{t:05.2f}_{a.view}", t, a.view, a.view) for t in a.times]
    elif summary.get("render_shots"):                     # transfer logs name their own moments
        shots = [tuple(x) for x in summary["render_shots"]]
    else:
        shots = travel_shots(summary) if travel else default_shots(log)
    guard = WindowGuard()
    print(f"windows of this process before the renderer: {len(own_windows())}")
    t0 = time.time()
    renderer = mujoco.Renderer(sc.m, height=H, width=W)
    try:
        guard(renderer, "after creating the renderer")
        w = own_windows()
        print(f"after creating the renderer: {len(w)} window(s) of this process, visible: "
              f"{sum(x[3] for x in w)}  {[(c, t) for _, c, t, _ in w]}")
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0                 # hide the collision boxes
        opt.sitegroup[:] = 0                 # hide all sites (hook marker is a visual geom)
        written = []
        for name, t, view, desc in shots:
            i = int(np.argmin(np.abs(log["frames_t"] - t)))
            fr, quat = log["frames"][i], log["frames_quat"][i]
            pos = sc.aligned_pos(fr, log["frames_pos"][i], quat)
            mk, mh = sc.set_state(fr, pos, quat)
            cam, fovy = camera(view, fr, pos, p, follow=("3d" if mission else travel), box=box)
            if travel:                                # directional-light shadows are drawn around the light
                sc.m.light_pos[sun] = (pos[:, 0].mean(), pos[:, 1].mean(), max(4.0, pos[:, 2].mean() + 3.0))
            sc.m.vis.global_.fovy = fovy              # orthographic: visible height in metres
            sc.m.vis.global_.orthographic = cam.orthographic
            renderer.update_scene(sc.d, camera=cam, scene_option=opt)
            img = renderer.render()
            guard(renderer, f"after rendering {name}")
            s = state_numbers(p, fr, pos, quat)
            lines = [f"t = {log['frames_t'][i]:.2f} s   {desc}",
                     f"net sag {s['sag']:.3f} m   mean tilt {s['tilt']:.1f} deg   drone z {s['z_drone']:.2f} m"]
            if view != "side":
                lines.append(("faint rotor discs" if a.style == "mesh" else "rotors")
                             + ": red = CCW, blue = CW (seen from above); orange = corner strings")
            path = os.path.join(out, f"{name}.png")
            caption(img, lines).save(path)
            written.append((path, img.shape[1], img.shape[0]))
            print(f"  {name}: t={log['frames_t'][i]:.2f} s view={view} sag={s['sag']:.3f} m tilt={s['tilt']:.1f} deg "
                  f"lean outward={s['outward']} black={img.max() == 0} mismatch knot/hook "
                  f"{mk * 1e3:.1e}/{mh * 1e3:.1e} mm")
        guard(renderer, "before closing the renderer")
    finally:
        renderer.close()
    dt = time.time() - t0
    print(f"window check: {guard.checks} checks while the renderer existed, at most {guard.max_windows} window(s) "
          f"of this process, none visible; classes/titles seen: {sorted(guard.seen)}")
    print(f"render time {dt:.2f} s for {len(written)} images; after close: {len(own_windows())} window(s)")
    for path, w_, h_ in written:
        print(f"  {path}  {w_} x {h_}")


if __name__ == "__main__":
    main()
