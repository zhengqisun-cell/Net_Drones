import numpy as np
import mujoco
import glfw

XML_PATH = "cloth_drones.xml"

# 需要与你 build 脚本一致
BASE_Z = 2.0

model = None
data = None
cam = None
opt = None
scene = None
context = None

button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0


def mouse_button_callback(window, button, action, mods):
    global button_left, button_middle, button_right
    if button == glfw.MOUSE_BUTTON_LEFT:
        button_left = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        button_middle = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        button_right = (action == glfw.PRESS)


def cursor_pos_callback(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    global cam, scene, model

    if not (button_left or button_middle or button_right):
        lastx = xpos
        lasty = ypos
        return

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    width, height = glfw.get_window_size(window)
    if height == 0:
        height = 1

    if button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_V if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) != glfw.PRESS else mujoco.mjtMouse.mjMOUSE_ROTATE_H
    elif button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_V if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) != glfw.PRESS else mujoco.mjtMouse.mjMOUSE_MOVE_H
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll_callback(window, xoffset, yoffset):
    global cam, scene, model
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)


def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)


def run_sim():
    global model, data, cam, opt, scene, context

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    window = glfw.create_window(1600, 900, "Net Wrap (v6: corner Δz-aware + contact-driven radius)", None, None)
    glfw.make_context_current(window)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=70000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)
    cam.distance = 13.0
    cam.elevation = -25
    cam.azimuth = 90
    cam.lookat[:] = np.array([0, 0, 1.6])

    payload_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    payload_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_geom")
    R = float(model.geom_size[payload_gid][0])  # 球半径

    # --- 主节点 joint: j_i_j ---
    node = {}
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not jname or not jname.startswith("j_"):
            continue
        parts = jname.split("_")
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            i = int(parts[1]); j = int(parts[2])
            qadr = model.jnt_qposadr[jid]
            vadr = model.jnt_dofadr[jid]
            quat0 = np.array(data.qpos[qadr+3:qadr+7], copy=True)
            node[(i, j)] = dict(jid=jid, qadr=qadr, vadr=vadr, quat0=quat0)

    # --- drones: drone_i_j（四角+中心） ---
    drones = []
    for body_id in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if not bname or not bname.startswith("drone_"):
            continue
        parts = bname.split("_")
        if len(parts) != 3:
            continue
        i, j = int(parts[1]), int(parts[2])

        ax = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"thrX_{i}_{j}")
        ay = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"thrY_{i}_{j}")
        az = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"thrZ_{i}_{j}")

        info = node[(i, j)]
        drones.append(dict(i=i, j=j, body=body_id, ax=ax, ay=ay, az=az, **info))

    # --- edge 节点：每步投影回插值点，防滑动 ---
    edges = []
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if not jname or not jname.startswith("j_edge_"):
            continue
        parts = jname.split("_")
        if len(parts) != 6:
            continue
        typ = parts[2]  # h/v
        i = int(parts[3]); j = int(parts[4]); k = int(parts[5])
        qadr = model.jnt_qposadr[jid]
        vadr = model.jnt_dofadr[jid]
        quat0 = np.array(data.qpos[qadr+3:qadr+7], copy=True)
        edges.append(dict(typ=typ, i=i, j=j, k=k, qadr=qadr, vadr=vadr, quat0=quat0))
    subdiv = max(e["k"] for e in edges)

    # --- PD gains ---
    kp_xy = 140.0
    kd_xy = 22.0
    kp_z  = 320.0
    kd_z  = 32.0

    # --- 接触统计：与 payload 接触的“网相关 body 数” ---
    def count_contact_bodies_with_payload():
        touched_bodies = set()
        for ci in range(data.ncon):
            c = data.contact[ci]
            g1, g2 = c.geom1, c.geom2
            if g1 == payload_gid:
                other = g2
            elif g2 == payload_gid:
                other = g1
            else:
                continue

            gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other) or ""
            if gname.startswith("nodegeom_") or gname.startswith("edgegeom_"):
                bid = model.geom_bodyid[other]
                touched_bodies.add(bid)
        return len(touched_bodies)

    # ==========================
    # v6 控制思路（你关心的点在这）
    # 1) 四角无人机的 Z 目标不再是“一个全局 z_corner”
    #    而是对每个角单独计算：
    #       z_target_corner_i = payload_z - beta*R - extra_down
    #    这样每个角都在追踪自己与球的相对高度差（Δz），能主动抬/压到球“赤道附近”。
    #
    # 2) 半径 r 仍然用“接触驱动”去收紧（接触不够就更收紧）。
    #
    # 3) 中心无人机只做“上方轻压/稳定”，高度采用 payload_z + offset，但加上上限，避免抬球飞天。
    # ==========================

    # 半径状态（接触驱动）
    r = 3.6
    C_target = 18
    kr = 0.020
    r_min, r_max = 1.05, 4.6

    # 角无人机 Z 目标参数
    beta = 0.35        # 球心往下偏移比例（0.35*R 让网先碰到下半球/赤道）
    extra_gain = 0.010 # 接触不足时的额外下压增益（乘以 err）
    z_min = BASE_Z + 0.10
    z_max = BASE_Z + 2.80

    # 中心无人机：球上方“盖子”
    center_offset = 0.45 * R + 0.25  # 球面上方一点
    center_z_max  = BASE_Z + 2.20    # 限制中心抬得太高（避免抬球飞天）

    # “无接触”时的接近策略：先让角无人机对齐到球的赤道高度，不要让网塌地
    no_contact_time = 0.0
    APPROACH_SEC = 2.0

    def desired_positions(payload_pos, r, C):
        px, py, pz = payload_pos
        targets = {}

        # 接触不足的误差（>0 表示还不够贴合）
        err = max(0.0, float(C_target - C))

        # 每个角无人机自己的 z 目标（Δz-aware）
        # z_eq：目标“接触带”高度（接近球的赤道/下半球）
        #z_eq = pz - beta * R
        extra_down = extra_gain * err  # 接触不够就更下压一点（促进更多节点贴球）

        for d in drones:
            if (d["i"], d["j"]) == (2, 2):
                zc = min(pz + center_offset, center_z_max)
                targets[(d["i"], d["j"])] = np.array([px, py, zc])
            else:
                sx = -1.0 if d["i"] == 0 else (1.0 if d["i"] == 4 else 0.0)
                sy = -1.0 if d["j"] == 0 else (1.0 if d["j"] == 4 else 0.0)

                # 角无人机 z 目标：围绕球的赤道高度，叠加“接触不足下压”
                zt = np.clip(100, z_min, z_max)
                targets[(d["i"], d["j"])] = np.array([px + sx * r, py + sy * r, zt])

        return targets

    sim_t = 0.0
    while not glfw.window_should_close(window):
        # 1) edge 小球投影防滑
        for e in edges:
            if e["typ"] == "h":
                p0 = data.qpos[node[(e["i"], e["j"])]["qadr"] : node[(e["i"], e["j"])]["qadr"] + 3]
                p1 = data.qpos[node[(e["i"] + 1, e["j"])]["qadr"] : node[(e["i"] + 1, e["j"])]["qadr"] + 3]
            else:
                p0 = data.qpos[node[(e["i"], e["j"])]["qadr"] : node[(e["i"], e["j"])]["qadr"] + 3]
                p1 = data.qpos[node[(e["i"], e["j"] + 1)]["qadr"] : node[(e["i"], e["j"] + 1)]["qadr"] + 3]

            a = e["k"] / (subdiv + 1.0)
            p = (1.0 - a) * p0 + a * p1

            qadr = e["qadr"]
            vadr = e["vadr"]
            data.qpos[qadr:qadr+3] = p
            data.qpos[qadr+3:qadr+7] = e["quat0"]
            data.qvel[vadr:vadr+6] = 0.0

        # 2) 读取 payload + 接触数
        ppos = data.xpos[payload_bid].copy()
        C = count_contact_bodies_with_payload()

        # 3) 半径 r：接触驱动（不够就收紧；无接触阶段不要暴力缩到最小）
        if C == 0:
            no_contact_time += model.opt.timestep

            # 目标半径：先围在球旁边（避免网塌地），并慢慢靠近
            r_goal = max(1.6, 1.2 * R + 1.0)
            r = np.clip(0.995 * r + 0.005 * r_goal, r_min, r_max)

            # 若长时间无接触：稍微扩大“扫一下”，帮助边缘碰到球
            if no_contact_time > APPROACH_SEC:
                r = min(r + 0.01, r_max)
        else:
            no_contact_time = 0.0
            err = float(C_target - C)
            r = np.clip(r - kr * err, r_min, r_max)
            if C >= C_target:
                r = max(r_min, min(r, 1.15))  # 达标后固定小半径

        # 4) 计算目标 + PD 控制
        targets = desired_positions(ppos, r, C)

        for d in drones:
            qadr = d["qadr"]
            vadr = d["vadr"]

            pos = data.qpos[qadr:qadr+3]
            vel = data.qvel[vadr:vadr+3]
            tgt = targets[(d["i"], d["j"])]

            e = tgt - pos
            u = np.array([
                kp_xy * e[0] - kd_xy * vel[0],
                kp_xy * e[1] - kd_xy * vel[1],
                kp_z  * e[2] - kd_z  * vel[2],
            ])

            data.ctrl[d["ax"]] = np.clip(u[0], -250.0, 250.0)
            data.ctrl[d["ay"]] = np.clip(u[1], -250.0, 250.0)
            data.ctrl[d["az"]] = np.clip(u[2], -600.0, 600.0)

            data.qpos[qadr+3:qadr+7] = d["quat0"]

        mujoco.mj_step(model, data)
        sim_t += model.opt.timestep

        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    run_sim()
