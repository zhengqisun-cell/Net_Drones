import math
import os
import time

import mujoco


HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "net_drone_fishing.xml")

# -----------------------------
# 物理参数
# -----------------------------
# 重力大小（沿世界坐标 -Z 方向）。例如 0.1 表示轻微下坠，0.0 表示失重。
G_MAG = 0.0

# -----------------------------
# 无人机执行模式
# -----------------------------
# 每帧渲染内的控制子步数，越大控制越平滑，但计算开销更高。
SIM_SUBSTEPS_PER_FRAME = 8
# True=硬钉目标点（相对形状保持最好）；False=限速跟踪（更像真实执行器）。
FORMATION_HARD_PIN = True
# 仅在 FORMATION_HARD_PIN=False 时生效，表示单轴最大跟踪速度（m/s）。
FORMATION_SPEED = 1.8

# -----------------------------
# 阶段1：AIM（瞄准）
# -----------------------------
# 对准判定阈值（角度，deg）。越小表示瞄准越严格。
AIM_ALIGN_DEG = 6.0
# 姿态旋转速度上限（deg/s）。越大对准越快但动作更激进。
AIM_ROT_RATE_DEG = 70.0
# 连续满足对准条件的保持时间（秒）后，才进入第二阶段。
AIM_HOLD_SEC = 0.8

# -----------------------------
# 阶段2：APPROACH（接近）
# -----------------------------
# 网整体朝目标前进速度（m/s）。
APPROACH_SPEED = 0.70
# 旧的接近阈值（保留参数，当前逻辑主要由“进入/退出”滞回阈值驱动）。
APPROACH_CLOSE_DIST = 1.8
# 物块“进入网中心范围”的阈值（m，进入阈值，较小）。
PAYLOAD_CENTER_ENTER_RADIUS = 1.5
# 物块“退出网中心范围”的阈值（m，退出阈值，较大，用于滞回防抖）。
PAYLOAD_CENTER_EXIT_RADIUS = 4.0

# -----------------------------
# 阶段3：CLOSE（收口）
# -----------------------------
# 收口时间常数（秒）。越小收口越快。
CLOSE_TAU = 0.9
# 底面中心无人机远离角点聚集中心的位移量（m），用于拉开空间。
CLOSE_BOTTOM_PUSH = 0.80

# -----------------------------
# 日志参数
# -----------------------------
# 控制台打印周期（秒）。
PRINT_EVERY = 0.5


def v_add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def v_sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def v_scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def v_dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def v_cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def v_norm(a):
    return math.sqrt(max(0.0, v_dot(a, a)))


def v_unit(a, fallback=(1.0, 0.0, 0.0)):
    n = v_norm(a)
    if n < 1e-12:
        return fallback
    return (a[0] / n, a[1] / n, a[2] / n)


def v_lerp(a, b, alpha):
    return (
        (1.0 - alpha) * a[0] + alpha * b[0],
        (1.0 - alpha) * a[1] + alpha * b[1],
        (1.0 - alpha) * a[2] + alpha * b[2],
    )


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def mat_identity():
    return (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )


def mat_mul(a, b):
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = (
                a[i][0] * b[0][j] +
                a[i][1] * b[1][j] +
                a[i][2] * b[2][j]
            )
    return (
        (out[0][0], out[0][1], out[0][2]),
        (out[1][0], out[1][1], out[1][2]),
        (out[2][0], out[2][1], out[2][2]),
    )


def mat_vec(m, v):
    return (
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    )


def axis_angle_rot(axis_u, angle):
    x, y, z = axis_u
    c = math.cos(angle)
    s = math.sin(angle)
    one_c = 1.0 - c
    return (
        (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
        (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
        (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
    )


def clip_step(cur, tgt, vmax, dt):
    d = tgt - cur
    lim = vmax * dt
    if d > lim:
        d = lim
    elif d < -lim:
        d = -lim
    return cur + d


def mean_pos_from_bodies(data, body_ids):
    sx = sy = sz = 0.0
    n = float(len(body_ids))
    for bid in body_ids:
        sx += float(data.xpos[bid, 0])
        sy += float(data.xpos[bid, 1])
        sz += float(data.xpos[bid, 2])
    return (sx / n, sy / n, sz / n)


def main():
    if not os.path.exists(XML_PATH):
        raise FileNotFoundError(f"XML not found: {XML_PATH}")

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Set gravity
    model.opt.gravity[0] = 0.0
    model.opt.gravity[1] = 0.0
    model.opt.gravity[2] = -abs(float(G_MAG))
    mujoco.mj_forward(model, data)

    # 5 control drones
    drone_names = ["drone_0_0", "drone_10_0", "drone_0_10", "drone_10_10", "drone_5_5"]
    corner_ids = [0, 1, 2, 3]
    bottom_id = 4

    qadr = []
    dadr = []
    quat0 = []
    drone_body_ids = []
    for bname in drone_names:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bname)
        if bid < 0:
            raise RuntimeError(f"Drone body not found: {bname}")
        drone_body_ids.append(int(bid))
        parts = bname.split("_")
        jname = f"j_{parts[1]}_{parts[2]}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"Joint not found: {jname}")
        qa = int(model.jnt_qposadr[jid])
        da = int(model.jnt_dofadr[jid])
        qadr.append(qa)
        dadr.append(da)
        quat0.append([
            float(data.qpos[qa + 3]),
            float(data.qpos[qa + 4]),
            float(data.qpos[qa + 5]),
            float(data.qpos[qa + 6]),
        ])

    # Payloads: discover all free-joints named j_payload*
    payload_infos = []
    for jid in range(model.njnt):
        jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or ""
        if not jname.startswith("j_payload"):
            continue
        payload_infos.append(
            {
                "joint_name": jname,
                "qadr": int(model.jnt_qposadr[jid]),
            }
        )
    if not payload_infos:
        raise RuntimeError("No payload joints found (expected names starting with j_payload).")
    payload_infos.sort(key=lambda x: x["joint_name"])
    payload_in_center = {info["joint_name"]: False for info in payload_infos}

    # Net nodes for geometric center and marker
    node_body_ids = []
    for body_id in range(model.nbody):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if nm.startswith("drone_") or nm.startswith("n_"):
            node_body_ids.append(int(body_id))
    if not node_body_ids:
        raise RuntimeError("No net node bodies found.")

    marker_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "net_center_marker")
    marker_mocap_id = int(model.body_mocapid[marker_bid]) if marker_bid >= 0 else -1

    def update_marker():
        if marker_mocap_id < 0:
            return
        c = mean_pos_from_bodies(data, node_body_ids)
        data.mocap_pos[marker_mocap_id] = c
        data.mocap_quat[marker_mocap_id] = (1.0, 0.0, 0.0, 0.0)

    # Initial rigid template around pivot (5-drone geometric center)
    init_pos = []
    for qa in qadr:
        init_pos.append((
            float(data.qpos[qa + 0]),
            float(data.qpos[qa + 1]),
            float(data.qpos[qa + 2]),
        ))
    pivot_cmd = (
        sum(p[0] for p in init_pos) / 5.0,
        sum(p[1] for p in init_pos) / 5.0,
        sum(p[2] for p in init_pos) / 5.0,
    )
    rel_local = [v_sub(p, pivot_cmd) for p in init_pos]
    R_cmd = mat_identity()

    # State machine
    phase = "AIM"
    aim_hold = 0.0
    close_alpha = 0.0
    close_corner_start = None
    close_corner_center = None
    close_bottom_start = None
    close_away_dir = None

    def init_close_phase():
        nonlocal close_alpha, close_corner_start, close_corner_center, close_bottom_start, close_away_dir
        close_alpha = 0.0
        close_corner_start = []
        for k in corner_ids:
            close_corner_start.append((
                float(data.qpos[qadr[k] + 0]),
                float(data.qpos[qadr[k] + 1]),
                float(data.qpos[qadr[k] + 2]),
            ))
        close_corner_center = (
            sum(p[0] for p in close_corner_start) / 4.0,
            sum(p[1] for p in close_corner_start) / 4.0,
            sum(p[2] for p in close_corner_start) / 4.0,
        )
        close_bottom_start = (
            float(data.qpos[qadr[bottom_id] + 0]),
            float(data.qpos[qadr[bottom_id] + 1]),
            float(data.qpos[qadr[bottom_id] + 2]),
        )
        close_away_dir = v_unit(
            v_sub(close_bottom_start, close_corner_center),
            fallback=(0.0, 0.0, -1.0),
        )

    try:
        import mujoco.viewer as viewer
    except Exception as exc:
        raise RuntimeError("mujoco.viewer is unavailable in this environment.") from exc

    dt = float(model.opt.timestep)
    last_print = time.time()

    with viewer.launch_passive(model, data) as v:
        v.cam.distance = 10.0
        v.cam.elevation = -18.0
        v.cam.azimuth = 120.0
        print("Capture controller started.")
        print("Phases: AIM -> APPROACH -> CLOSE")

        while v.is_running():
            frame_start = time.time()
            dt_ctrl = dt * float(SIM_SUBSTEPS_PER_FRAME)

            net_center = mean_pos_from_bodies(data, node_body_ids)
            bottom_pos = (
                float(data.qpos[qadr[bottom_id] + 0]),
                float(data.qpos[qadr[bottom_id] + 1]),
                float(data.qpos[qadr[bottom_id] + 2]),
            )

            # Evaluate all payloads against net-center radius.
            vec_cb_ref = v_sub(bottom_pos, net_center)  # center -> bottom
            u_cb_ref = v_unit(vec_cb_ref, fallback=(0.0, 0.0, -1.0))
            payload_states = []
            for info in payload_infos:
                q = info["qadr"]
                pos = (
                    float(data.qpos[q + 0]),
                    float(data.qpos[q + 1]),
                    float(data.qpos[q + 2]),
                )
                vec_pc_i = v_sub(net_center, pos)  # payload -> center
                d = v_norm(vec_pc_i)
                u_pc_i = v_unit(vec_pc_i, fallback=(0.0, 0.0, -1.0))
                cos_i = clamp(v_dot(u_cb_ref, u_pc_i), -1.0, 1.0)
                aim_i = math.degrees(math.acos(cos_i))
                line_i = v_norm(v_cross(vec_pc_i, u_cb_ref))
                # Hysteresis update:
                # - outside -> inside when d <= ENTER
                # - inside -> outside when d >= EXIT
                in_center_prev = payload_in_center[info["joint_name"]]
                if in_center_prev:
                    in_center_now = d < PAYLOAD_CENTER_EXIT_RADIUS
                else:
                    in_center_now = d <= PAYLOAD_CENTER_ENTER_RADIUS
                payload_in_center[info["joint_name"]] = in_center_now
                payload_states.append(
                    {
                        "name": info["joint_name"],
                        "pos": pos,
                        "dist": d,
                        "aim_deg": aim_i,
                        "line_err": line_i,
                        "in_center": in_center_now,
                    }
                )

            uncaptured = [s for s in payload_states if not s["in_center"]]
            # Target priority in phase 1/2: better alignment first.
            uncaptured.sort(key=lambda s: (s["aim_deg"], s["line_err"], s["dist"]))
            all_in_center = len(uncaptured) == 0

            active = None
            if uncaptured:
                active = uncaptured[0]
                payload_pos = active["pos"]
                center_payload_dist = active["dist"]
                aim_angle_deg = active["aim_deg"]
                line_err = active["line_err"]
            else:
                payload_pos = None
                center_payload_dist = 0.0
                aim_angle_deg = 0.0
                line_err = 0.0

            # ----- Phase logic -----
            if not all_in_center:
                # CLOSE is gated off until every payload is in-center.
                if phase == "CLOSE":
                    phase = "AIM"
                    aim_hold = 0.0

                if phase in ("AIM", "APPROACH") and payload_pos is not None:
                    # Aim by rotating bottom-axis command toward payload direction.
                    d_cur = v_unit(mat_vec(R_cmd, rel_local[bottom_id]), fallback=(0.0, 0.0, -1.0))
                    # Enforce order: payload -> net_center -> bottom_drone
                    d_tgt = v_unit(v_sub(pivot_cmd, payload_pos), fallback=d_cur)
                    axis = v_cross(d_cur, d_tgt)
                    axis_n = v_norm(axis)
                    dot_cd = clamp(v_dot(d_cur, d_tgt), -1.0, 1.0)
                    ang = math.atan2(axis_n, dot_cd)
                    max_step = math.radians(AIM_ROT_RATE_DEG) * dt_ctrl
                    step = min(ang, max_step)
                    if axis_n > 1e-10 and step > 0.0:
                        R_step = axis_angle_rot(v_scale(axis, 1.0 / axis_n), step)
                        R_cmd = mat_mul(R_step, R_cmd)

                if phase == "AIM":
                    if aim_angle_deg <= AIM_ALIGN_DEG:
                        aim_hold += dt_ctrl
                    else:
                        aim_hold = 0.0
                    if aim_hold >= AIM_HOLD_SEC:
                        phase = "APPROACH"

                elif phase == "APPROACH":
                    to_payload = v_sub(payload_pos, pivot_cmd)
                    d = v_norm(to_payload)
                    if d > 1e-9:
                        step = min(APPROACH_SPEED * dt_ctrl, d)
                        pivot_cmd = v_add(pivot_cmd, v_scale(v_unit(to_payload), step))

                    # Once current target enters center range (with hysteresis state), restart AIM.
                    if active is not None and active["in_center"]:
                        phase = "AIM"
                        aim_hold = 0.0
            else:
                # All payloads are in-center: now CLOSE is allowed.
                if phase != "CLOSE":
                    phase = "CLOSE"
                    init_close_phase()
                close_alpha = clamp(
                    close_alpha + dt_ctrl / max(CLOSE_TAU, 1e-6),
                    0.0,
                    1.0,
                )

            # ----- Target generation -----
            if phase in ("AIM", "APPROACH"):
                targets = []
                for rel in rel_local:
                    targets.append(v_add(pivot_cmd, mat_vec(R_cmd, rel)))
            else:
                targets = [None] * 5
                for i, k in enumerate(corner_ids):
                    targets[k] = v_lerp(close_corner_start[i], close_corner_center, close_alpha)
                targets[bottom_id] = v_add(close_bottom_start, v_scale(close_away_dir, CLOSE_BOTTOM_PUSH * close_alpha))

            # ----- Apply control -----
            for _ in range(SIM_SUBSTEPS_PER_FRAME):
                for k in range(5):
                    qa = qadr[k]
                    da = dadr[k]
                    tx, ty, tz = targets[k]

                    if FORMATION_HARD_PIN:
                        px, py, pz = tx, ty, tz
                    else:
                        px = clip_step(float(data.qpos[qa + 0]), tx, float(FORMATION_SPEED), dt)
                        py = clip_step(float(data.qpos[qa + 1]), ty, float(FORMATION_SPEED), dt)
                        pz = clip_step(float(data.qpos[qa + 2]), tz, float(FORMATION_SPEED), dt)

                    data.qpos[qa + 0] = px
                    data.qpos[qa + 1] = py
                    data.qpos[qa + 2] = pz
                    data.qpos[qa + 3] = quat0[k][0]
                    data.qpos[qa + 4] = quat0[k][1]
                    data.qpos[qa + 5] = quat0[k][2]
                    data.qpos[qa + 6] = quat0[k][3]
                    for d in range(6):
                        data.qvel[da + d] = 0.0

                mujoco.mj_forward(model, data)
                mujoco.mj_step(model, data)

            update_marker()
            mujoco.mj_forward(model, data)
            v.sync()

            # ----- Logging -----
            now = time.time()
            if now - last_print >= PRINT_EVERY:
                last_print = now
                active_name = active["name"] if active is not None else "none"
                dist_items = []
                for s in sorted(payload_states, key=lambda x: x["name"]):
                    short_name = s["name"].replace("j_", "")
                    dist_items.append(f"{short_name}:{s['dist']:.2f}")
                dist_text = ", ".join(dist_items)
                print(
                    f"phase={phase:8s} "
                    f"sim={float(data.time):6.2f}s "
                    f"target={active_name:16s} "
                    f"in_center={len(payload_states)-len(uncaptured)}/{len(payload_states)} "
                    f"aim={aim_angle_deg:5.1f}deg "
                    f"line_err={line_err:5.2f} "
                    f"dist={center_payload_dist:5.2f} "
                    f"close_alpha={close_alpha:4.2f} "
                    f"payload_dist=[{dist_text}]"
                )

            # Soft realtime pacing
            elapsed = time.time() - frame_start
            sleep_t = dt - elapsed
            if sleep_t > 0.0:
                time.sleep(sleep_t)


if __name__ == "__main__":
    main()
