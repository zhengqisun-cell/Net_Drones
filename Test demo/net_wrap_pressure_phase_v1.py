import numpy as np
import mujoco
import glfw

XML_PATH = "cloth_drones.xml"
BASE_Z = 2.0  # 与 build 脚本保持一致

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

    window = glfw.create_window(1600, 900, "Net Wrap (pressure-phase control)", None, None)
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
    cam.lookat[:] = np.array([0, 0, 1.8])

    payload_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    payload_gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "payload_geom")
    R = float(model.geom_size[payload_gid][0])

    # 主节点 joint: j_i_j
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

    # drones: 只四个角（因为 XML 已去掉中心）
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

    # edge 节点：投影回插值点，防滑
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
    subdiv = max(e["k"] for e in edges) if edges else 0

    # 所有“网格节点” body（压力感受器）
    # 规则：凡是 geom 名称以 nodegeom_ 开头的 body，都算一个节点
    pressure_bodies = set()
    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        if gname.startswith("nodegeom_"):
            bid = model.geom_bodyid[gid]
            pressure_bodies.add(bid)
    pressure_bodies = sorted(list(pressure_bodies))
    N_nodes = len(pressure_bodies)
    print(f"Pressure nodes count: {N_nodes}")

    # PD gains
    kp_xy = 140.0
    kd_xy = 22.0
    kp_z  = 320.0
    kd_z  = 32.0

    # 阶段阈值
    f_flat = 0.05
    f_mid_low = 0.05
    f_mid_high = 0.1
    f_close = 0.5

    # 展开状态下的参考位置：保持接近初始“平网”
    rest_corner_pos = {}
    for d in drones:
        qadr = d["qadr"]
        pos0 = np.array(data.qpos[qadr:qadr+3], copy=True)
        rest_corner_pos[(d["i"], d["j"])] = pos0

    # 收缩时半径范围（围绕 payload 的四角）
    r_min = 1.05
    r_max = 3.2
    r = 2.6  # 当前半径状态

    def count_pressure_nodes():
        # 统计“与 payload 接触的网格节点数量”（去重 body）
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

            bid = model.geom_bodyid[other]
            if bid in pressure_bodies:
                touched_bodies.add(bid)
        return len(touched_bodies)

    def phase_from_fraction(f):
        if f < f_flat:
            return 0  # 展开阶段
        if f_mid_low <= f <= f_mid_high:
            return 1  # 预捕捉阶段
        if f > f_close:
            return 2  # 收口阶段
        # 中间模糊区域
        return 1 if f > (f_flat + f_mid_low) * 0.5 else 0

    def desired_positions(payload_pos, phase, f):
        nonlocal r
        px, py, pz = payload_pos
        targets = {}

        if phase == 0:
            # 展开：网尽量平、展开、接近初始状态
            for d in drones:
                key = (d["i"], d["j"])
                rest = rest_corner_pos[key]
                tgt = rest.copy()
                tgt[2] = BASE_Z + 0.35
                targets[key] = tgt

            r_goal = r_max
            r = 0.99 * r + 0.01 * r_goal

        elif phase == 1:
            # 预捕捉：约一半节点有压力
            # 目标：围着 payload 移动，让物体进入网内部
            r_goal = max(1.6, 1.2 * R + 1.0)
            r = np.clip(0.995 * r + 0.005 * r_goal, r_min, r_max)

            z_ring = pz - 0.35 * R
            for d in drones:
                i, j = d["i"], d["j"]
                sx = -1.0 if i == 0 else 1.0
                sy = -1.0 if j == 0 else 1.0
                key = (i, j)
                targets[key] = np.array([
                    px + sx * r,
                    py + sy * r,
                    z_ring
                ])

        else:
            # 收口：大部分节点有压力
            r_goal = r_min
            r = np.clip(0.99 * r + 0.01 * r_goal, r_min, r_max)

            z_ring = pz - 0.20 * R
            for d in drones:
                i, j = d["i"], d["j"]
                sx = -1.0 if i == 0 else 1.0
                sy = -1.0 if j == 0 else 1.0
                key = (i, j)
                targets[key] = np.array([
                    px + sx * r,
                    py + sy * r,
                    z_ring
                ])

        return targets

    sim_t = 0.0
    print("Pressure-phase control started.")
    while not glfw.window_should_close(window):
        # 1) edge 小球投影防滑
        for e in edges:
            if subdiv <= 0:
                break
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

        # 2) 压力节点计数 + 阶段判断
        ppos = data.xpos[payload_bid].copy()
        C = count_pressure_nodes()
        f = C / N_nodes if N_nodes > 0 else 0.0
        phase = phase_from_fraction(f)

        # 3) 生成目标并 PD 控制
        targets = desired_positions(ppos, phase, f)

        for d in drones:
            key = (d["i"], d["j"])
            tgt = targets[key]

            qadr = d["qadr"]
            vadr = d["vadr"]

            pos = data.qpos[qadr:qadr+3]
            vel = data.qvel[vadr:vadr+3]

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

        if int(sim_t * 10) % 50 == 0:
            print(f"t={sim_t:.2f}, C={C}, f={f:.2f}, phase={phase}, r={r:.2f}")

        width, height = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    run_sim()
