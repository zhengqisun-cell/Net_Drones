# -*- coding: utf-8 -*-
import numpy as np
import mujoco
import glfw

import net_5x5_drones as base


# ---------------------------------------------------------------------
# Rigid formation controller parameters / 刚体编队控制参数
# ---------------------------------------------------------------------
# SIM_SUBSTEPS_PER_FRAME:
#   EN: Number of controller updates inside one rendered frame.
#   CN: 每一帧渲染内执行多少次控制子步。越大越平滑、越稳定，但更耗算力。
SIM_SUBSTEPS_PER_FRAME = 10

# FORMATION_HARD_PIN:
#   EN: True = directly pin each controlled drone to target every substep.
#       False = speed-limited tracking.
#   CN: True 表示每个子步都把 5 架控制无人机直接放到目标点（刚性最强）。
#       False 表示限速跟踪（更“真实”，但会出现跟踪滞后）。
FORMATION_HARD_PIN = True

# FORMATION_SPEED [m/s], used only when FORMATION_HARD_PIN=False:
#   EN: Max per-axis chase speed in soft tracking mode.
#   CN: 仅在限速模式下生效；每个坐标轴追踪目标的最大速度（米/秒）。
#       值太小会跟不上旋转，值太大则效果接近硬钉。
FORMATION_SPEED = 1.2

# ---------------------------------------------------------------------
# Optional center trajectory / 编队中心轨迹参数（可选）
# ---------------------------------------------------------------------
# CENTER_PATH_ENABLED:
#   EN: If True, the rigid formation rotates while translating in world frame.
#   CN: True 时编队会“边旋转边平移”；False 时只绕初始中心旋转。
CENTER_PATH_ENABLED = False

# CENTER_PATH_AMP [m]:
#   EN: XYZ sinusoidal amplitude of center motion.
#   CN: 中心在 X/Y/Z 方向正弦位移幅值（米）。值越大，活动范围越大。
CENTER_PATH_AMP = np.array([1.2, 1.0, 0.8], dtype=float)

# CENTER_PATH_FREQ [rad/s]:
#   EN: XYZ angular frequencies for center sinusoidal motion.
#   CN: 中心轨迹三轴角频率（弧度/秒）。越大，中心轨迹变化越快。
CENTER_PATH_FREQ = np.array([0.18, 0.15, 0.21], dtype=float)

# CENTER_PATH_PHASE [rad]:
#   EN: XYZ phase offsets for center sinusoidal motion.
#   CN: 中心轨迹三轴相位偏置（弧度），用于调整轨迹起始姿态。
CENTER_PATH_PHASE = np.array([0.0, 1.1, 2.0], dtype=float)

# ---------------------------------------------------------------------
# World-frame attitude law / 世界坐标系姿态规律
# ---------------------------------------------------------------------
# Rotation composition: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
# 旋转顺序：先 roll，再 pitch，再 yaw（按右侧矩阵作用顺序理解）。

# ROLL_AMP_DEG, PITCH_AMP_DEG [deg]:
#   EN: Oscillation amplitudes for roll/pitch.
#   CN: 横滚/俯仰振荡幅值（度）。
ROLL_AMP_DEG = 90.0
PITCH_AMP_DEG = 90.0

# ROLL_FREQ, PITCH_FREQ [rad/s]:
#   EN: Oscillation frequencies for roll/pitch.
#   CN: 横滚/俯仰振荡频率（弧度/秒）。越大表示摆动越快。
ROLL_FREQ = 0.25
PITCH_FREQ = 0.35

# YAW_RATE_DEG [deg/s]:
#   EN: Constant yaw spin rate in world frame.
#   CN: 世界系偏航自转角速度（度/秒）。
YAW_RATE_DEG = 15.0


model = None
data = None
cam = None
opt = None
scene = None
context = None

button_left = False
button_middle = False
button_right = False
lastx = 0.0
lasty = 0.0


def mouse_button_callback(window, button, action, mods):
    # 鼠标按键状态缓存，用于 MuJoCo 默认相机交互。
    global button_left, button_middle, button_right
    if button == glfw.MOUSE_BUTTON_LEFT:
        button_left = action == glfw.PRESS
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        button_middle = action == glfw.PRESS
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        button_right = action == glfw.PRESS


def cursor_pos_callback(window, xpos, ypos):
    # 鼠标拖动控制相机：左键旋转、右键平移、中键缩放。
    global lastx, lasty
    if not (button_left or button_middle or button_right):
        lastx, lasty = xpos, ypos
        return

    dx = xpos - lastx
    dy = ypos - lasty
    lastx, lasty = xpos, ypos

    width, height = glfw.get_window_size(window)
    if height == 0:
        height = 1

    if button_left:
        shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    elif button_right:
        shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll_callback(window, xoffset, yoffset):
    # 滚轮缩放视角。
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)


def key_callback(window, key, scancode, action, mods):
    # 按 ESC 关闭窗口。
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)


def rot_x(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]], dtype=float)


def rot_y(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]], dtype=float)


def rot_z(a):
    ca, sa = np.cos(a), np.sin(a)
    return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float)


def rotation_world(t):
    # 旋转角定义：
    # roll/pitch 为正弦振荡，yaw 为线性增长（匀速转圈）。
    roll = np.deg2rad(ROLL_AMP_DEG) * np.sin(ROLL_FREQ * t)
    pitch = np.deg2rad(PITCH_AMP_DEG) * np.sin(PITCH_FREQ * t + 0.7)
    yaw = np.deg2rad(YAW_RATE_DEG) * t
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def run_sim():
    global model, data, cam, opt, scene, context

    # 复用原脚本的 XML 生成逻辑与场景参数。
    base.ensure_xml(force=base.XML_FORCE_REBUILD_ON_RUN)
    model = mujoco.MjModel.from_xml_path(base.XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    window = glfw.create_window(base.VIEWER_WIDTH, base.VIEWER_HEIGHT, "Net rigid free-rotation", None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(base.VIEWER_SWAP_INTERVAL)

    glfw.set_mouse_button_callback(window, mouse_button_callback)
    glfw.set_cursor_pos_callback(window, cursor_pos_callback)
    glfw.set_scroll_callback(window, scroll_callback)
    glfw.set_key_callback(window, key_callback)

    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scene = mujoco.MjvScene(model, maxgeom=20000)
    context = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)
    mujoco.mjv_defaultCamera(cam)
    mujoco.mjv_defaultOption(opt)

    cam.distance = base.CAM_DISTANCE
    cam.elevation = base.CAM_ELEVATION
    cam.azimuth = base.CAM_AZIMUTH
    cam.lookat[:] = np.array(base.CAM_LOOKAT, dtype=float)

    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(base.VIS_CONTACT_POINT)
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(base.VIS_CONTACT_FORCE)

    dt = float(model.opt.timestep)
    node_body_ids = []
    for body_id in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if bname.startswith("drone_") or bname.startswith("n_"):
            node_body_ids.append(body_id)
    node_body_ids = np.array(node_body_ids, dtype=int)

    marker_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "net_center_marker")
    marker_mocap_id = int(model.body_mocapid[marker_bid]) if marker_bid >= 0 else -1

    def update_net_center_marker():
        if marker_mocap_id < 0 or node_body_ids.size == 0:
            return
        center = np.mean(data.xpos[node_body_ids, :3], axis=0)
        data.mocap_pos[marker_mocap_id] = center
        data.mocap_quat[marker_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    # 从 body 名称自动推断 4 角 + 中心 这 5 架控制无人机。
    drone_entries = {}
    for body_id in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not bname.startswith("drone_"):
            continue
        parts = bname.split("_")
        if len(parts) != 3 or (not parts[1].isdigit()) or (not parts[2].isdigit()):
            continue
        i, j = int(parts[1]), int(parts[2])
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"j_{i}_{j}")
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        quat0 = np.array(data.qpos[qadr + 3:qadr + 7], dtype=float)
        drone_entries[(i, j)] = dict(body=body_id, qadr=qadr, dadr=dadr, quat0=quat0)

    if not drone_entries:
        raise RuntimeError("No drones found in model.")

    i_vals = sorted({k[0] for k in drone_entries.keys()})
    j_vals = sorted({k[1] for k in drone_entries.keys()})
    i_min, i_max = i_vals[0], i_vals[-1]
    j_min, j_max = j_vals[0], j_vals[-1]
    i_mid, j_mid = (i_min + i_max) // 2, (j_min + j_max) // 2
    control_keys = [
        (i_min, j_min),
        (i_max, j_min),
        (i_min, j_max),
        (i_max, j_max),
        (i_mid, j_mid),
    ]
    missing = [k for k in control_keys if k not in drone_entries]
    if missing:
        raise RuntimeError(f"Missing expected control drones: {missing}")

    drone_qadr = np.array([drone_entries[k]["qadr"] for k in control_keys], dtype=int)
    drone_dadr = np.array([drone_entries[k]["dadr"] for k in control_keys], dtype=int)
    drone_quat0 = np.array([drone_entries[k]["quat0"] for k in control_keys], dtype=float)

    # 冻结局部编队几何（相对中心向量）：
    # rel_local[k] = p_k(0) - p_center(0)
    # 后续每帧使用统一旋转矩阵 R 作用到 rel_local，确保相对关系不变。
    init_xyz = np.array([data.qpos[int(q):int(q) + 3] for q in drone_qadr], dtype=float)
    center0 = init_xyz[4].copy()
    rel_local = init_xyz - center0[None, :]

    last_print_bucket = -1
    update_net_center_marker()
    mujoco.mj_forward(model, data)

    while not glfw.window_should_close(window):
        t = float(data.time)
        R = rotation_world(t)

        if CENTER_PATH_ENABLED:
            # 中心平移轨迹（逐轴正弦）。
            center_target = center0 + CENTER_PATH_AMP * np.sin(CENTER_PATH_FREQ * t + CENTER_PATH_PHASE)
        else:
            center_target = center0

        # 核心刚体方程：
        # target_i = center_target + R @ rel_local_i
        rel_world = (R @ rel_local.T).T
        targets = center_target[None, :] + rel_world

        for _ in range(SIM_SUBSTEPS_PER_FRAME):
            for k in range(5):
                q = int(drone_qadr[k])
                d = int(drone_dadr[k])
                tx, ty, tz = targets[k]

                if FORMATION_HARD_PIN:
                    # 硬钉：每个子步直接写入目标点，保持编队刚性。
                    px, py, pz = tx, ty, tz
                else:
                    # 软跟踪：按速度上限逼近目标点。
                    x, y, z = data.qpos[q + 0], data.qpos[q + 1], data.qpos[q + 2]
                    vmax = max(FORMATION_SPEED, 1e-6)
                    px = x + np.clip(tx - x, -vmax * dt, vmax * dt)
                    py = y + np.clip(ty - y, -vmax * dt, vmax * dt)
                    pz = z + np.clip(tz - z, -vmax * dt, vmax * dt)

                data.qpos[q + 0] = px
                data.qpos[q + 1] = py
                data.qpos[q + 2] = pz
                data.qpos[q + 3:q + 7] = drone_quat0[k]
                data.qvel[d:d + 6] = 0.0

            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

        update_net_center_marker()
        mujoco.mj_forward(model, data)

        bucket = int(data.time * 2.0)
        if bucket != last_print_bucket:
            last_print_bucket = bucket
            cur_xyz = np.array([data.qpos[int(q):int(q) + 3] for q in drone_qadr], dtype=float)
            cur_rel = cur_xyz - cur_xyz[4][None, :]
            target_rel = rel_world
            rel_err = np.linalg.norm(cur_rel - target_rel, axis=1)
            # rel_err_max 表示当前“相对几何保持误差”最大值。
            print(
                f"t={data.time:6.2f}s center=({center_target[0]:5.2f},{center_target[1]:5.2f},{center_target[2]:5.2f}) "
                f"rel_err_max={np.max(rel_err):.4e}"
            )

        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    run_sim()

