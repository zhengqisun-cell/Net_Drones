# net_5x5_drones.py
# Simplified 11x11 square net (adjacent-only springs) in 0-g.
# Drones: 4 corners + center (total 5) to maintain shape and move the net.
#
# Run:
#   pip install mujoco glfw numpy
#   python net_5x5_drones.py
#
# Files:
#   - auto-generates net_drone.xml next to this script if missing.

import os
import itertools
import numpy as np
import mujoco
import glfw

HERE = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(HERE, "net_drone.xml")

# ----------------------------
# Tunable Parameters / 可调参数
# ----------------------------
# XML scene generation / XML场景生成参数
XML_NX = 11  # 网格X方向节点数 / Node count along X
XML_NY = 11  # 网格Y方向节点数 / Node count along Y
XML_SIZE = 4.0  # 网边长(米) / Net side length (m)
XML_BASE_Z = None  # 网初始高度(None=自动) / Initial net height (None=auto)
XML_BASE_Z_OFFSET = 4.2  # 自动高度: 物块上方偏移 / Auto height offset above debris
XML_TIMESTEP = 0.001  # 仿真步长 / Simulation timestep
XML_GRAVITY = (0, 0, 0)  # 重力 / Gravity vector
XML_NODE_RADIUS = 0.10  # 普通网节点半径 / Non-drone node radius
XML_NODE_DENSITY = 40.0  # 普通网节点密度 / Non-drone node density
XML_DRONE_SIZE = (0.16, 0.16, 0.10)  # 无人机盒尺寸半长 / Drone box half-size
XML_DRONE_DENSITY = 120.0  # 无人机密度 / Drone density
XML_STIFFNESS = 1600.0  # 网弹簧刚度 / Tendon stiffness
XML_DAMPING = 8.0  # 网弹簧阻尼 / Tendon damping
XML_DEBRIS_SIZE = (0.7, 0.7, 0.7)  # 物块盒尺寸半长 / Debris box half-size
XML_DEBRIS_DENSITY = 300.0  # 物块密度 / Debris density
XML_DEBRIS_POS = (1.0, 0.0, 0.0)  # 物块初始位置 / Debris initial position
DEBRIS_LOCKED = False  # 是否锁定物块位置 / Whether debris pose is locked each frame
XML_NET_CENTER_XY = None  # 网中心XY(None=跟随物块) / Net center XY (None=align debris)
XML_NET_CENTER_XY_OFFSET = (2.0, -3.8)  # 初始水平误差(仅在XML_NET_CENTER_XY=None时生效) / Initial XY offset (used only when XML_NET_CENTER_XY is None)
XML_CENTER_MARKER_SIZE = 0.12  # 白色中心球半径 / White center marker radius
XML_CENTER_MARKER_ALPHA = 0.25  # 白色中心球透明度 / White center marker alpha
XML_FORCE_REBUILD_ON_RUN = True  # 每次运行是否重建XML / Rebuild XML on each run
XML_GRID3D_ENABLED = False  # 是否显示3D空间网格线 / Show 3D reference grid
XML_GRID3D_HALF_EXTENTS = (4.0, 4.0, 3.0)  # 网格半尺寸(x,y,z) / Grid half extents (x,y,z)
XML_GRID3D_SPACING = 0.5  # 网格间距 / Grid spacing
XML_GRID3D_LINE_RADIUS = 0.003  # 网格线半径 / Grid line radius
XML_GRID3D_RGBA = (1.0, 1.0, 1.0, 0.12)  # 网格线颜色与透明度 / Grid line rgba
XML_GOAL_MARKER_ENABLED = True  # 是否显示目标点物块 / Show goal marker block
XML_GOAL_MARKER_POS = (5.0, 5.0, 5.0)  # 目标点坐标 / Goal marker position
XML_GOAL_MARKER_HALF_SIZE = (0.22, 0.22, 0.22)  # 目标点物块半尺寸 / Goal marker half-size
XML_GOAL_MARKER_RGBA = (0.9, 0.95, 1.0, 0.20)  # 目标点物块颜色与透明度 / Goal marker rgba
XML_CHECKER_FLOOR_ENABLED = True  # 是否启用棋盘格地板 / Enable checkerboard floor
XML_CHECKER_TILE_SIZE = 1.0  # 棋盘格边长(米) / Checker tile size (m)
XML_CHECKER_HALF_TILES_X = 10  # 地板X方向半格数(总格数=2*该值) / Floor half tiles along X
XML_CHECKER_HALF_TILES_Y = 10  # 地板Y方向半格数(总格数=2*该值) / Floor half tiles along Y
XML_CHECKER_FLOOR_Z = -2.0  # 地板高度 / Checker floor z height
XML_CHECKER_TILE_HALF_THICKNESS = 0.015  # 地板方格半厚度 / Checker tile half thickness
XML_CHECKER_COLOR_A = (0.92, 0.92, 0.92, 1.0)  # 棋盘浅色 / Checker light color
XML_CHECKER_COLOR_B = (0.16, 0.16, 0.16, 1.0)  # 棋盘深色 / Checker dark color

# Viewer and simulation runtime / 可视化与运行参数
VIEWER_WIDTH = 1600  # 窗口宽度 / Window width
VIEWER_HEIGHT = 900  # 窗口高度 / Window height
VIEWER_TITLE = "Net (4 corners + center drone)"  # 窗口标题 / Window title
VIEWER_SWAP_INTERVAL = 0  # 0=关闭VSync / 0 = disable vsync
CAM_DISTANCE = 10.0  # 相机距离 / Camera distance
CAM_ELEVATION = -25  # 相机俯仰角 / Camera elevation
CAM_AZIMUTH = 90  # 相机方位角 / Camera azimuth
CAM_LOOKAT = (3.0, 0.0, 0.8)  # 相机观察点 / Camera lookat
VIS_CONTACT_POINT = True  # 显示接触点 / Show contact points
VIS_CONTACT_FORCE = True  # 显示接触力 / Show contact forces
SIM_SUBSTEPS_PER_FRAME = 10  # 每帧仿真步数 / Sim steps per render frame

# State machine thresholds / 状态机阈值
CAPTURE_ENTER_DIST = 1.0  # APPROACH->CAPTURE 距离阈值 / enter capture distance
CAPTURE_EXIT_DIST = 2.2  # CAPTURE->APPROACH 距离阈值 / exit capture distance
FINAL_ENTER_CENTER_DIST = 0.25  # 进入FINAL_1的中心距阈值 / enter FINAL_1 center distance
FINAL_EXIT_CENTER_DIST = 0.35  # 退出FINAL阶段阈值 / leave final stages distance
CORNER_LOCK_TRIGGER_DIST = 2.0 * max(XML_DRONE_SIZE[0], XML_DRONE_SIZE[1]) + 0.04  # 锁链触发阈值(角点无人机贴合即触发) / lock trigger distance for corner drones being close/touching
MOVE_ENTER_AFTER_FINAL3_SEC = 2.0  # FINAL_3保持多久后进入MOVE(秒) / Delay before FINAL_3 -> MOVE (s)
MOVE_TARGET_POS = XML_GOAL_MARKER_POS  # MOVE阶段网中心目标点 / Net-center target point in MOVE
MOVE_CENTER_SPEED_XY = 0.45  # MOVE阶段网中心XY平移速度 / Net-center XY translation speed in MOVE
MOVE_CENTER_SPEED_Z = 0.30  # MOVE阶段网中心Z平移速度 / Net-center Z translation speed in MOVE
MOVE_FORMATION_HOLD_GAIN = 0.85  # MOVE队形保持强度(0-1, 越大越硬) / Formation hold strength in MOVE (0-1, larger = stiffer)
MOVE_STOP_CENTER_DIST = 0.9  # MOVE->STOP中心距离阈值(3D) / MOVE->STOP center distance threshold (3D)
MOVE_STOP_MAX_DRONE_SPEED = 0.10  # MOVE->STOP无人机最大速度阈值 / MOVE->STOP max drone speed threshold
CAPTURE_ASPECT_MIN = 0.45  # 进入/维持捕获的最小宽长比 / Min aspect for capture logic
CAPTURE_ALPHA_DIST_SCALE = CAPTURE_ENTER_DIST  # alpha与距离映射尺度 / Distance scale for alpha
CAPTURE_ALPHA_ASPECT_PENALTY = 0.15  # 网太窄时alpha惩罚系数 / Alpha penalty when aspect too narrow

# Controller gains and geometry shaping / 控制与形状参数
FLY_SPEED = 0.5  # 网中心平移速度 / Net center translation speed
ASPECT_REF = 0.55  # 参考宽长比 / Reference aspect ratio
K_WIDEN = 0.8  # APPROACH展宽增益 / Approach widen gain
K_RELAX_L = 0.35  # APPROACH长度放松增益 / Approach length relax gain
W_CAP = 1.25  # APPROACH最大宽度倍率 / Max width scale in approach
L_CAP = 1.15  # APPROACH最大长度倍率 / Max length scale in approach
ALPHA_TAU = 0.30  # CAPTURE alpha平滑时间常数 / alpha smoothing tau in capture
ALPHA_TAU_FINAL = 0.12  # FINAL alpha收敛时间常数 / alpha tau in final stages
CLOSE_RATIO = 0.55  # CAPTURE收口强度 / Closure ratio in capture
CLOSE_RATIO_FINAL = 0.92  # FINAL收口强度 / Closure ratio in final stages
CORNER_PAIR_LOCK_BLEND = 0.75  # 锁链组内拉齐基准强度 / Base lock-chain blend
CORNER_PAIR_LOCK_GAIN = 2.0  # 锁链组内拉齐增益 / Lock-chain blend gain
CORNER_GUARD_VS_CLOSE_GAIN = 1.25  # 统一远离约束相对收口倍数 / Keep-away gain vs closure
CORNER_GUARD_MARGIN_FINAL = 0.0  # 统一远离约束固定增量 / Keep-away fixed margin
CORNER_HOVER_OFFSET = 0.45  # 角点悬停高度偏移 / Corner hover z offset
DEBRIS_EDGE_LENGTH = 2.0 * float(XML_DEBRIS_SIZE[2])  # 物块边长(取Z向盒边长) / Debris edge length (z-axis box edge)
CORNER_CAPTURE_OFFSET = -2.0 * DEBRIS_EDGE_LENGTH  # CAPTURE角点下压偏移(= -2x 物块边长) / Corner capture z offset (= -2x debris edge length)
CENTER_HOVER_OFFSET = 0.75  # 中心悬停高度偏移 / Center hover z offset
CENTER_CAPTURE_OFFSET = 0.55  # CAPTURE中心高度偏移 / Center capture z offset
CENTER_APPROACH_ASPECT_Z_GAIN = 0.25  # APPROACH中中心随形变抬升增益 / Center z gain from aspect in approach
FINAL_CORNER_UNDER_CLEARANCE = 0.35  # FINAL角点下穿物块底面的净空 / Final corner under-clearance
FINAL_CENTER_TOP_CLEARANCE = 0.02  # FINAL中心贴顶净空 / Final center top clearance
VXY_NOM = 0.9  # XY最大运动速度 / Max XY motion speed
VZ_NOM = 0.8  # Z最大运动速度 / Max Z motion speed
SLOW_MIN_SCALE = 0.35  # 形变严重时最小速度比例 / Min speed scale when net is deformed
ALPHA_DECAY_NON_CAPTURE = 0.97  # 非捕获阶段alpha衰减 / Alpha decay outside capture/final
CAPTURE_HALFW_MIN_SCALE = 0.80  # CAPTURE时宽度最小倍率 / Min width scale in capture
CAPTURE_HALFW_ASPECT_GAIN = 0.60  # CAPTURE时宽度对形变补偿增益 / Width compensation gain in capture
CAPTURE_HALF_L_NARROW_FLOOR = 0.92  # 网过窄时长度最小倍率 / Min length scale when aspect is narrow
FINAL_MIN_SCALE = 0.08  # FINAL收口最小倍率 / Final minimum closure scale

# Safety / 数值安全参数
BOX_HALF_EXTENTS_FALLBACK = 0.10  # 无box几何时的半尺寸回退 / Fallback half-extent
DRONE_DEBRIS_CLEARANCE = 1e-3  # 无人机与物块最小间隙 / Min drone-debris clearance
EPS_NORM = 1e-9  # 范数零判断阈值 / Norm near-zero epsilon
EPS_ORTHO = 1e-12  # 方向正交化阈值 / Orthogonalization epsilon
EPS_GUARD = 1e-6  # 约束半径最小阈值 / Minimum guard radius epsilon

# ----------------------------
# XML generator
# ----------------------------
def build_net_xml(
    nx=XML_NX, ny=XML_NY,
    size=XML_SIZE,                 # square net side length (m)
    base_z=XML_BASE_Z,             # None -> auto place above debris
    timestep=XML_TIMESTEP,
    gravity=XML_GRAVITY,

    node_radius=XML_NODE_RADIUS,
    node_density=XML_NODE_DENSITY,

    drone_size=XML_DRONE_SIZE,
    drone_density=XML_DRONE_DENSITY,

    stiffness=XML_STIFFNESS,
    damping=XML_DAMPING,

    debris_size=XML_DEBRIS_SIZE,
    debris_density=XML_DEBRIS_DENSITY,
    debris_pos=XML_DEBRIS_POS,
    net_center_xy=XML_NET_CENTER_XY,       # None -> align with debris x,y
):
    width = size
    height = size
    dx = width / (nx - 1)
    dy = height / (ny - 1)

    if net_center_xy is None:
        net_center_xy = (
            float(debris_pos[0]) + float(XML_NET_CENTER_XY_OFFSET[0]),
            float(debris_pos[1]) + float(XML_NET_CENTER_XY_OFFSET[1]),
        )
    if base_z is None:
        base_z = float(debris_pos[2]) + float(debris_size[2]) + XML_BASE_Z_OFFSET
    cx0, cy0 = float(net_center_xy[0]), float(net_center_xy[1])

    # 4 corners + center drone
    drones = {(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1), (nx // 2, ny // 2)}

    def pos(i, j):
        x = cx0 - width / 2 + i * dx
        y = cy0 - height / 2 + j * dy
        z = base_z
        return x, y, z

    xml = []
    xml.append(f'<mujoco model="net_{nx}x{ny}_corner_center">')
    xml.append('  <compiler angle="degree" coordinate="local"/>')
    xml.append(f'  <option timestep="{timestep}" gravity="{gravity[0]} {gravity[1]} {gravity[2]}" integrator="RK4"/>')
    xml.append('  <size nconmax="20000" njmax="20000"/>')
    xml.append('  <visual>')
    xml.append('    <headlight active="1" ambient="0.42 0.42 0.42" diffuse="0.70 0.70 0.70" specular="0.20 0.20 0.20"/>')
    xml.append('  </visual>')
    xml.append('  <worldbody>')
    xml.append('    <light name="key_top" directional="true" castshadow="false" pos="0 0 14" dir="0 0 -1" diffuse="0.95 0.95 0.95" specular="0.25 0.25 0.25"/>')
    xml.append('    <light name="fill_posx" directional="true" castshadow="false" pos="9 6 9" dir="-0.7 -0.4 -0.6" diffuse="0.55 0.55 0.55" specular="0.10 0.10 0.10"/>')
    xml.append('    <light name="fill_negx" directional="true" castshadow="false" pos="-9 -6 9" dir="0.7 0.4 -0.6" diffuse="0.50 0.50 0.50" specular="0.08 0.08 0.08"/>')
    xml.append('    <light name="back_rim" directional="true" castshadow="false" pos="0 -10 7" dir="0 0.85 -0.5" diffuse="0.40 0.40 0.42" specular="0.06 0.06 0.07"/>')

    if XML_CHECKER_FLOOR_ENABLED:
        tile_size = max(float(XML_CHECKER_TILE_SIZE), 1e-6)
        half_x = max(int(XML_CHECKER_HALF_TILES_X), 1)
        half_y = max(int(XML_CHECKER_HALF_TILES_Y), 1)
        floor_z = float(XML_CHECKER_FLOOR_Z)
        half_t = max(float(XML_CHECKER_TILE_HALF_THICKNESS), 1e-4)
        c1 = XML_CHECKER_COLOR_A
        c2 = XML_CHECKER_COLOR_B
        half_s = 0.5 * tile_size
        for ix in range(-half_x, half_x):
            for iy in range(-half_y, half_y):
                cx_tile = (ix + 0.5) * tile_size
                cy_tile = (iy + 0.5) * tile_size
                rgba = c1 if ((ix + iy) & 1) == 0 else c2
                xml.append(
                    f'    <geom name="checker_{ix + half_x}_{iy + half_y}" type="box" '
                    f'pos="{cx_tile:.4f} {cy_tile:.4f} {floor_z:.4f}" size="{half_s:.4f} {half_s:.4f} {half_t:.4f}" '
                    f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" contype="0" conaffinity="0"/>'
                )
    if XML_GRID3D_ENABLED:
        gx, gy, gz = XML_GRID3D_HALF_EXTENTS
        gsp = max(float(XML_GRID3D_SPACING), 1e-6)
        gr = float(XML_GRID3D_LINE_RADIUS)
        grgba = XML_GRID3D_RGBA

        # 3D lattice lines (visual only): x-lines
        y_vals = np.arange(-gy, gy + 0.5 * gsp, gsp)
        z_vals = np.arange(-gz, gz + 0.5 * gsp, gsp)
        for y in y_vals:
            for z in z_vals:
                xml.append(
                    f'    <geom type="capsule" fromto="{-gx:.3f} {y:.3f} {z:.3f} {gx:.3f} {y:.3f} {z:.3f}" '
                    f'size="{gr:.4f}" rgba="{grgba[0]} {grgba[1]} {grgba[2]} {grgba[3]}" contype="0" conaffinity="0"/>'
                )

        # y-lines
        x_vals = np.arange(-gx, gx + 0.5 * gsp, gsp)
        for x in x_vals:
            for z in z_vals:
                xml.append(
                    f'    <geom type="capsule" fromto="{x:.3f} {-gy:.3f} {z:.3f} {x:.3f} {gy:.3f} {z:.3f}" '
                    f'size="{gr:.4f}" rgba="{grgba[0]} {grgba[1]} {grgba[2]} {grgba[3]}" contype="0" conaffinity="0"/>'
                )

        # z-lines
        for x in x_vals:
            for y in y_vals:
                xml.append(
                    f'    <geom type="capsule" fromto="{x:.3f} {y:.3f} {-gz:.3f} {x:.3f} {y:.3f} {gz:.3f}" '
                    f'size="{gr:.4f}" rgba="{grgba[0]} {grgba[1]} {grgba[2]} {grgba[3]}" contype="0" conaffinity="0"/>'
                )

    xml.append(f'    <body name="net_center_marker" mocap="true" pos="{cx0:.5f} {cy0:.5f} {base_z:.5f}">')
    xml.append(
        f'      <geom name="net_center_geom" type="sphere" size="{XML_CENTER_MARKER_SIZE}" '
        f'rgba="1 1 1 {XML_CENTER_MARKER_ALPHA}" contype="0" conaffinity="0"/>'
    )
    xml.append('    </body>')

    # net nodes
    for j in range(ny):
        for i in range(nx):
            x, y, z = pos(i, j)
            is_drone = (i, j) in drones
            if is_drone:
                bname = f"drone_{i}_{j}"
                joint = f'<joint name="j_{i}_{j}" type="free"/>'
                sx, sy, sz = drone_size
                geom = (
                    f'<geom name="dronegeom_{i}_{j}" type="box" size="{sx} {sy} {sz}" density="{drone_density}" '
                    f'rgba="0.2 0.2 0.85 1" contype="2" conaffinity="1" friction="1.1 0.2 0.2"/>'
                )
            else:
                bname = f"n_{i}_{j}"
                joint = f'<joint name="j_{i}_{j}" type="free" damping="0.002"/>'
                geom = (
                    f'<geom name="nodegeom_{i}_{j}" type="sphere" size="{node_radius}" density="{node_density}" '
                    f'rgba="0.85 0.2 0.2 1" contype="1" conaffinity="1" friction="1.1 0.2 0.2"/>'
                )

            xml.append(f'    <body name="{bname}" pos="{x:.5f} {y:.5f} {z:.5f}">')
            xml.append(f'      {joint}')
            xml.append(f'      {geom}')
            xml.append(f'      <site name="s_{i}_{j}" size="0.01"/>')
            xml.append('    </body>')

    # debris
    dxs, dys, dzs = debris_size
    px, py, pz = debris_pos
    xml.append(f'    <body name="debris" pos="{px} {py} {pz}">')
    xml.append('      <joint name="j_debris" type="free"/>')
    xml.append(
        f'      <geom name="debris_geom" type="box" size="{dxs} {dys} {dzs}" density="{debris_density}" '
        f'rgba="0.2 0.7 0.2 1" contype="1" conaffinity="1" friction="1.1 0.2 0.2"/>'
    )
    xml.append('    </body>')

    # translucent goal marker block (visual only)
    if XML_GOAL_MARKER_ENABLED:
        gx, gy, gz = XML_GOAL_MARKER_POS
        gsx, gsy, gsz = XML_GOAL_MARKER_HALF_SIZE
        gr, gg, gb, ga = XML_GOAL_MARKER_RGBA
        xml.append(
            f'    <geom name="goal_marker_block" type="box" '
            f'pos="{gx} {gy} {gz}" size="{gsx} {gsy} {gsz}" '
            f'rgba="{gr} {gg} {gb} {ga}" contype="0" conaffinity="0"/>'
        )

    xml.append('  </worldbody>')

    # adjacent-only tendons
    xml.append('  <tendon>')

    def add_tendon(i1, j1, i2, j2, L):
        xml.append(
            f'    <spatial name="t_{i1}_{j1}_{i2}_{j2}" stiffness="{stiffness}" damping="{damping}" springlength="{L:.6f}" width="0.01">'
        )
        xml.append(f'      <site site="s_{i1}_{j1}"/>')
        xml.append(f'      <site site="s_{i2}_{j2}"/>')
        xml.append('    </spatial>')

    for j in range(ny):
        for i in range(nx - 1):
            add_tendon(i, j, i + 1, j, dx)
    for j in range(ny - 1):
        for i in range(nx):
            add_tendon(i, j, i, j + 1, dy)

    xml.append('  </tendon>')
    xml.append('</mujoco>')
    return "\n".join(xml)


def ensure_xml(force=False):
    if (not force) and os.path.exists(XML_PATH):
        return
    with open(XML_PATH, "w", encoding="utf-8") as f:
        f.write(build_net_xml())
    print("Generated:", XML_PATH)


# ----------------------------
# GLFW interaction
# ----------------------------
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
        lastx, lasty = xpos, ypos
        return

    dx = xpos - lastx
    dy = ypos - lasty
    lastx, lasty = xpos, ypos

    width, height = glfw.get_window_size(window)
    if height == 0:
        height = 1

    if button_left:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    elif button_right:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS else mujoco.mjtMouse.mjMOUSE_MOVE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll_callback(window, xoffset, yoffset):
    global cam, scene, model
    mujoco.mjv_moveCamera(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05 * yoffset, scene, cam)


def key_callback(window, key, scancode, action, mods):
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)


def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))


# ----------------------------
# Main sim
# ----------------------------
def run_sim(on_render=None):
    global model, data, cam, opt, scene, context

    ensure_xml(force=XML_FORCE_REBUILD_ON_RUN)
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if not glfw.init():
        raise RuntimeError("GLFW init failed")

    window = glfw.create_window(VIEWER_WIDTH, VIEWER_HEIGHT, VIEWER_TITLE, None, None)
    glfw.make_context_current(window)
    glfw.swap_interval(VIEWER_SWAP_INTERVAL)  # 0=no vsync / 0=关闭垂直同步

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

    cam.distance = CAM_DISTANCE
    cam.elevation = CAM_ELEVATION
    cam.azimuth = CAM_AZIMUTH
    cam.lookat[:] = np.array(CAM_LOOKAT, dtype=float)

    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(VIS_CONTACT_POINT)
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = int(VIS_CONTACT_FORCE)

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

    # Drone set: infer 4 corners + center from model names
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
        drone_entries[(i, j)] = dict(
            body=body_id,
            qadr=qadr,
            dadr=dadr,
            quat0=np.array(data.qpos[qadr+3:qadr+7], copy=True),
        )

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

    drone_body = np.array([drone_entries[k]["body"] for k in control_keys], dtype=int)
    drone_qadr = np.array([drone_entries[k]["qadr"] for k in control_keys], dtype=int)
    drone_dadr = np.array([drone_entries[k]["dadr"] for k in control_keys], dtype=int)
    drone_quat0 = np.array([drone_entries[k]["quat0"] for k in control_keys], dtype=float)
    corner_body_ids = [int(x) for x in drone_body[:4]]
    corner_body_to_idx = {bid: idx for idx, bid in enumerate(corner_body_ids)}

    # Debris fixed
    debris_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "debris")
    jid_debris = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "j_debris")
    q_debris = int(model.jnt_qposadr[jid_debris])
    d_debris = int(model.jnt_dofadr[jid_debris])
    debris_pos = np.array(XML_DEBRIS_POS, dtype=float)
    debris_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    debris_gid = -1
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == debris_body:
            debris_gid = gid
            break
    if debris_gid < 0:
        raise RuntimeError("Debris geom not found.")
    debris_half_xyz = np.array(model.geom_size[debris_gid][:3], dtype=float)
    debris_half_z = float(debris_half_xyz[2])

    def body_box_half_extents(body_id):
        half = np.zeros(3, dtype=float)
        for gid in range(model.ngeom):
            if int(model.geom_bodyid[gid]) != int(body_id):
                continue
            if int(model.geom_type[gid]) != int(mujoco.mjtGeom.mjGEOM_BOX):
                continue
            half = np.maximum(half, np.array(model.geom_size[gid][:3], dtype=float))
        if np.all(half == 0.0):
            half[:] = BOX_HALF_EXTENTS_FALLBACK
        return half

    drone_half_extents = np.array([body_box_half_extents(int(b)) for b in drone_body], dtype=float)

    def project_drone_outside_debris(pos_xyz, drone_half_xyz, debris_center):
        rel = np.array(pos_xyz, dtype=float) - debris_center
        lim = debris_half_xyz + drone_half_xyz + DRONE_DEBRIS_CLEARANCE
        if np.any(np.abs(rel) >= lim):
            return np.array(pos_xyz, dtype=float)
        pen = lim - np.abs(rel)
        axis = int(np.argmin(pen))
        sgn = 1.0 if rel[axis] >= 0.0 else -1.0
        rel[axis] = sgn * lim[axis]
        return debris_center + rel

    locked_corner_pairs = set()

    # Initial geometry / axes
    p00 = data.xpos[drone_body[0], :2].copy()
    p40 = data.xpos[drone_body[1], :2].copy()
    p04 = data.xpos[drone_body[2], :2].copy()
    p44 = data.xpos[drone_body[3], :2].copy()
    c0 = data.xpos[drone_body[4], :2].copy()

    u1 = (p40 + p44 - p00 - p04) * 0.5
    u2 = (p04 + p44 - p00 - p40) * 0.5
    u1 = u1 / (np.linalg.norm(u1) + EPS_ORTHO)
    u2 = u2 - (u2 @ u1) * u1
    u2 = u2 / (np.linalg.norm(u2) + EPS_ORTHO)

    halfL0 = 0.5 * float(np.linalg.norm(p40 - p00))
    halfW0 = 0.5 * float(np.linalg.norm(p04 - p00))
    z0 = float(np.mean([data.qpos[q+2] for q in drone_qadr]))

    # State controller
    mode = "APPROACH"
    capture_enter_dist = CAPTURE_ENTER_DIST
    capture_exit_dist = CAPTURE_EXIT_DIST
    final_enter_center_dist = FINAL_ENTER_CENTER_DIST
    final_exit_center_dist = FINAL_EXIT_CENTER_DIST
    corner_lock_trigger_dist = CORNER_LOCK_TRIGGER_DIST

    fly_speed = FLY_SPEED
    c_cmd = c0.copy()
    target_xy = debris_pos[:2].copy()

    aspect_ref = ASPECT_REF  # try to keep square-ish
    k_widen = K_WIDEN
    k_relaxL = K_RELAX_L
    W_cap = W_CAP
    L_cap = L_CAP

    alpha = 0.0
    alpha_tau = ALPHA_TAU
    alpha_tau_final = ALPHA_TAU_FINAL
    close_ratio = CLOSE_RATIO
    close_ratio_final = CLOSE_RATIO_FINAL
    corner_pair_lock_blend = CORNER_PAIR_LOCK_BLEND
    corner_pair_lock_gain = CORNER_PAIR_LOCK_GAIN
    corner_guard_scale_final = close_ratio_final * CORNER_GUARD_VS_CLOSE_GAIN
    corner_guard_margin_final = CORNER_GUARD_MARGIN_FINAL

    # Z targets are anchored to debris top, so the net actively descends to capture.
    corner_hover_offset = CORNER_HOVER_OFFSET
    corner_capture_offset = CORNER_CAPTURE_OFFSET
    center_hover_offset = CENTER_HOVER_OFFSET
    center_capture_offset = CENTER_CAPTURE_OFFSET
    final_corner_under_clearance = FINAL_CORNER_UNDER_CLEARANCE
    final_center_top_clearance = FINAL_CENTER_TOP_CLEARANCE

    vxy_nom = VXY_NOM
    vz_nom = VZ_NOM
    sim_substeps_per_frame = SIM_SUBSTEPS_PER_FRAME

    last_print = -1
    final3_hold_targets_xy = None
    final3_hold_targets_z = None
    final3_enter_time = None
    move_enter_after_final3_sec = MOVE_ENTER_AFTER_FINAL3_SEC
    move_target_xyz = np.array(MOVE_TARGET_POS, dtype=float)
    move_center_speed_xy = MOVE_CENTER_SPEED_XY
    move_center_speed_z = MOVE_CENTER_SPEED_Z
    move_formation_hold_gain = clamp(MOVE_FORMATION_HOLD_GAIN, 0.0, 1.0)
    move_stop_center_dist = MOVE_STOP_CENTER_DIST
    move_stop_max_drone_speed = MOVE_STOP_MAX_DRONE_SPEED
    move_offsets_xyz = None
    stop_hold_targets_xy = None
    stop_hold_targets_z = None
    prev_drone_xyz = np.array([data.qpos[int(q):int(q)+3] for q in drone_qadr], dtype=float)

    # Ensure debris starts from configured pose; keep free unless DEBRIS_LOCKED=True.
    data.qpos[q_debris:q_debris+3] = debris_pos
    data.qpos[q_debris+3:q_debris+7] = debris_quat
    if DEBRIS_LOCKED:
        data.qvel[d_debris:d_debris+6] = 0.0
    mujoco.mj_forward(model, data)

    update_net_center_marker()
    mujoco.mj_forward(model, data)

    while not glfw.window_should_close(window):
        mode_prev = mode

        # Debris can be free-moving; optional lock for debugging.
        if DEBRIS_LOCKED:
            data.qpos[q_debris:q_debris+3] = debris_pos
            data.qpos[q_debris+3:q_debris+7] = debris_quat
            data.qvel[d_debris:d_debris+6] = 0.0
        debris_center = np.array(data.qpos[q_debris:q_debris+3], dtype=float)
        target_xy = debris_center[:2].copy()

        # command center toward target (frame-time based, substep-count invariant)
        vec_cmd = target_xy - c_cmd
        cmd_dist = float(np.linalg.norm(vec_cmd))
        dt_ctrl = dt * sim_substeps_per_frame
        if cmd_dist > EPS_NORM:
            step = min(fly_speed * dt_ctrl, cmd_dist)
            c_cmd = c_cmd + (step / cmd_dist) * vec_cmd
        debris_top = debris_center[2] + debris_half_z
        net_center = np.mean(data.xpos[node_body_ids, :3], axis=0)
        net_center_xy_dist = float(np.linalg.norm(net_center[:2] - target_xy))
        center_dist = float(np.linalg.norm(net_center - debris_center))
        goal_dist = float(np.linalg.norm(debris_center - move_target_xyz))
        dt_frame = max(dt_ctrl, EPS_NORM)
        drone_xyz_now = np.array([data.qpos[int(q):int(q)+3] for q in drone_qadr], dtype=float)
        drone_speeds = np.linalg.norm((drone_xyz_now - prev_drone_xyz) / dt_frame, axis=1)
        max_drone_speed = float(np.max(drone_speeds))

        # current aspect from corner spans
        L_now = 0.5 * float(np.linalg.norm(data.xpos[drone_body[1], :2] - data.xpos[drone_body[0], :2]))
        W_now = 0.5 * float(np.linalg.norm(data.xpos[drone_body[2], :2] - data.xpos[drone_body[0], :2]))
        aspect = W_now / max(L_now, EPS_NORM)

        center_pos = np.array(data.qpos[drone_qadr[4]:drone_qadr[4] + 3], dtype=float)
        corner_now = [np.array(data.qpos[drone_qadr[i]:drone_qadr[i] + 3], dtype=float) for i in range(4)]
        pair_dists = {}
        pair_dists_sum = 0.0
        pair_dists_cnt = 0
        for i in range(4):
            for j in range(i + 1, 4):
                dij = float(np.linalg.norm(corner_now[i] - corner_now[j]))
                pair_dists[(i, j)] = dij
                pair_dists_sum += dij
                pair_dists_cnt += 1

        center_corner_dists = [float(np.linalg.norm(center_pos - corner_now[i])) for i in range(4)]
        mean_corner_pair_dist = pair_dists_sum / max(pair_dists_cnt, 1)
        mean_center_corner_dist = sum(center_corner_dists) / max(len(center_corner_dists), 1)
        final2_ready = mean_corner_pair_dist <= mean_center_corner_dist

        # "Contact-lock" trigger replaced by corner-center distance~=0 trigger.
        if mode in ("FINAL_1", "FINAL_2", "FINAL_3"):
            for key, dij in pair_dists.items():
                if dij <= corner_lock_trigger_dist:
                    locked_corner_pairs.add(key)

        # switch modes
        if mode == "STOP":
            mode = "STOP"
        elif mode == "MOVE":
            if (goal_dist <= move_stop_center_dist) and (max_drone_speed <= move_stop_max_drone_speed):
                mode = "STOP"
            else:
                mode = "MOVE"
        elif mode not in ("FINAL_1", "FINAL_2", "FINAL_3") and center_dist <= final_enter_center_dist:
            mode = "FINAL_1"
        elif mode == "FINAL_3":
            # FINAL_3 is terminal until timeout; then switch to MOVE.
            if (final3_enter_time is not None) and ((data.time - final3_enter_time) >= move_enter_after_final3_sec):
                mode = "MOVE"
            else:
                mode = "FINAL_3"
        elif mode in ("FINAL_1", "FINAL_2"):
            if center_dist >= final_exit_center_dist:
                mode = "CAPTURE" if net_center_xy_dist < capture_exit_dist else "APPROACH"
            elif mode in ("FINAL_1", "FINAL_2") and locked_corner_pairs:
                mode = "FINAL_3"
            elif mode == "FINAL_1" and final2_ready:
                mode = "FINAL_2"
            elif mode == "FINAL_2" and (not final2_ready):
                mode = "FINAL_1"
        else:
            if mode == "APPROACH":
                if net_center_xy_dist <= capture_enter_dist and aspect >= CAPTURE_ASPECT_MIN:
                    mode = "CAPTURE"
            else:
                if net_center_xy_dist >= capture_exit_dist:
                    mode = "APPROACH"

        # FINAL_3: freeze all drone targets at the moment of entering FINAL_3.
        if mode == "FINAL_3" and mode_prev != "FINAL_3":
            hold_xyz = np.array([data.qpos[int(q):int(q)+3] for q in drone_qadr], dtype=float)
            final3_hold_targets_xy = hold_xyz[:, :2].copy()
            final3_hold_targets_z = hold_xyz[:, 2].copy()
            final3_enter_time = float(data.time)
        elif mode != "FINAL_3":
            final3_hold_targets_xy = None
            final3_hold_targets_z = None

        if mode == "MOVE" and mode_prev != "MOVE":
            move_xyz = np.array([data.qpos[int(q):int(q)+3] for q in drone_qadr], dtype=float)
            move_center = np.mean(move_xyz, axis=0)
            move_offsets_xyz = move_xyz - move_center[None, :]
        if mode == "STOP" and mode_prev != "STOP":
            hold_xyz = np.array([data.qpos[int(q):int(q)+3] for q in drone_qadr], dtype=float)
            stop_hold_targets_xy = hold_xyz[:, :2].copy()
            stop_hold_targets_z = hold_xyz[:, 2].copy()

        # alpha
        if mode == "CAPTURE":
            alpha_raw = clamp(1.0 - net_center_xy_dist / CAPTURE_ALPHA_DIST_SCALE, 0.0, 1.0)
            if aspect < CAPTURE_ASPECT_MIN:
                alpha_raw *= CAPTURE_ALPHA_ASPECT_PENALTY
            alpha += (dt / alpha_tau) * (alpha_raw - alpha)
            alpha = clamp(alpha, 0.0, 1.0)
        elif mode in ("FINAL_1", "FINAL_2", "FINAL_3", "MOVE", "STOP"):
            alpha += (dt / alpha_tau_final) * (1.0 - alpha)
            alpha = clamp(alpha, 0.0, 1.0)
        else:
            alpha *= ALPHA_DECAY_NON_CAPTURE
            alpha = clamp(alpha, 0.0, 1.0)

        aspect_err = clamp(aspect_ref - aspect, 0.0, 1.0)

        if mode == "APPROACH":
            widen = min(1.0 + k_widen * aspect_err, W_cap)
            relax = min(1.0 + k_relaxL * aspect_err, L_cap)
            halfL = halfL0 * relax
            halfW = halfW0 * widen
        elif mode == "CAPTURE":
            halfL = halfL0 * (1.0 - close_ratio * alpha)
            halfW = halfW0 * max(CAPTURE_HALFW_MIN_SCALE, min(1.0 + CAPTURE_HALFW_ASPECT_GAIN * aspect_err, W_cap))
            if aspect < CAPTURE_ASPECT_MIN:
                halfL = max(halfL, CAPTURE_HALF_L_NARROW_FLOOR * halfL0)
        else:
            halfL = max(FINAL_MIN_SCALE * halfL0, halfL0 * (1.0 - close_ratio_final * alpha))
            halfW = max(FINAL_MIN_SCALE * halfW0, halfW0 * (1.0 - close_ratio_final * alpha))

        c_des = c_cmd
        t00 = c_des + (-halfL) * u1 + (-halfW) * u2
        t40 = c_des + (+halfL) * u1 + (-halfW) * u2
        t04 = c_des + (-halfL) * u1 + (+halfW) * u2
        t44 = c_des + (+halfL) * u1 + (+halfW) * u2
        tc = c_des

        if mode in ("FINAL_1", "FINAL_2", "FINAL_3"):
            corners = [t00, t40, t04, t44]
            # Priority (low->high): close-in < unified keep-away (FINAL_1 only) < lock-chain.
            guard_radius = (
                corner_guard_scale_final * float(np.linalg.norm(center_pos - debris_center))
                + corner_guard_margin_final
            )
            corner_z_for_constraint = debris_top + (1.0 - alpha) * corner_hover_offset + alpha * corner_capture_offset
            for idx in range(4):
                corner3 = np.array([corners[idx][0], corners[idx][1], corner_z_for_constraint], dtype=float)
                if mode == "FINAL_1":
                    for ref in (debris_center, center_pos):
                        r = corner3 - ref
                        rn = float(np.linalg.norm(r))
                        if rn < max(guard_radius, EPS_GUARD):
                            if rn < EPS_NORM:
                                r = np.array([1.0, 0.0, 0.0], dtype=float)
                                rn = 1.0
                            corner3 = ref + (r / rn) * guard_radius
                corners[idx] = corner3[:2]

            # Once a corner pair is latched, pull the pair together permanently.
            if locked_corner_pairs:
                parent = list(range(4))

                def findp(x):
                    while parent[x] != x:
                        parent[x] = parent[parent[x]]
                        x = parent[x]
                    return x

                def unionp(a, b):
                    ra, rb = findp(a), findp(b)
                    if ra != rb:
                        parent[rb] = ra

                for a, b in locked_corner_pairs:
                    unionp(a, b)

                groups = {}
                for i in range(4):
                    r0 = findp(i)
                    groups.setdefault(r0, []).append(i)

                for comp in groups.values():
                    if len(comp) <= 1:
                        continue
                    mean_xy = np.mean([corners[i] for i in comp], axis=0)
                    lock_blend_eff = clamp(corner_pair_lock_blend * corner_pair_lock_gain, 0.0, 1.0)
                    for i in comp:
                        corners[i] = (1.0 - lock_blend_eff) * corners[i] + lock_blend_eff * mean_xy

            t00, t40, t04, t44 = corners

        if mode == "APPROACH":
            zc = debris_top + center_hover_offset + CENTER_APPROACH_ASPECT_Z_GAIN * aspect_err
            zcorn = debris_top + corner_hover_offset
        elif mode == "CAPTURE":
            zcorn = debris_top + (1.0 - alpha) * corner_hover_offset + alpha * corner_capture_offset
            zc = debris_top + (1.0 - alpha) * center_hover_offset + alpha * center_capture_offset
        else:
            # FINAL: drive corners below debris while center stays on top (no interpenetration).
            zcorn = debris_center[2] - debris_half_z - final_corner_under_clearance
            zc = debris_top + drone_half_extents[4][2] + final_center_top_clearance

        targets_xy = np.array([t00, t40, t04, t44, tc], dtype=float)
        z_targets = np.array([zcorn, zcorn, zcorn, zcorn, zc], dtype=float)
        if mode == "FINAL_3" and final3_hold_targets_xy is not None and final3_hold_targets_z is not None:
            targets_xy = final3_hold_targets_xy.copy()
            z_targets = final3_hold_targets_z.copy()
        elif mode == "STOP" and stop_hold_targets_xy is not None and stop_hold_targets_z is not None:
            targets_xy = stop_hold_targets_xy.copy()
            z_targets = stop_hold_targets_z.copy()
        elif mode == "MOVE" and move_offsets_xyz is not None:
            # Priority 1: keep relative formation; Priority 2: move net center toward goal.
            dt_ctrl = dt * sim_substeps_per_frame
            move_xyz_now = np.array([data.qpos[int(q):int(q)+3] for q in drone_qadr], dtype=float)
            move_center_now = np.mean(move_xyz_now, axis=0)
            move_err = move_target_xyz - net_center
            err_xy = move_err[:2]
            err_xy_norm = float(np.linalg.norm(err_xy))
            center_step = np.zeros(3, dtype=float)
            if err_xy_norm > EPS_NORM:
                step_xy = min(move_center_speed_xy * dt_ctrl, err_xy_norm)
                center_step[:2] = (step_xy / err_xy_norm) * err_xy
            center_step[2] = np.clip(move_err[2], -move_center_speed_z * dt_ctrl, move_center_speed_z * dt_ctrl)
            desired_center = move_center_now + center_step
            rigid_targets = desired_center[None, :] + move_offsets_xyz
            shifted_now = move_xyz_now + center_step[None, :]
            move_targets = shifted_now + move_formation_hold_gain * (rigid_targets - shifted_now)
            targets_xy = move_targets[:, :2]
            z_targets = move_targets[:, 2]

        slow = clamp(aspect / max(aspect_ref, EPS_GUARD), SLOW_MIN_SCALE, 1.0)
        vxy = vxy_nom * slow
        vz = vz_nom * slow

        for _ in range(sim_substeps_per_frame):
            # apply 5 drones kinematically
            for k in range(5):
                q = int(drone_qadr[k])
                d = int(drone_dadr[k])

                x, y, z = data.qpos[q+0], data.qpos[q+1], data.qpos[q+2]
                tx, ty = targets_xy[k]
                tz = float(z_targets[k])

                if (
                    (mode == "FINAL_3" and final3_hold_targets_xy is not None and final3_hold_targets_z is not None)
                    or (mode == "STOP" and stop_hold_targets_xy is not None and stop_hold_targets_z is not None)
                ):
                    # Hard hold in FINAL_3/STOP: directly pin drones to frozen world-frame targets.
                    p_next = np.array([tx, ty, tz], dtype=float)
                else:
                    dx = np.clip(tx - x, -vxy * dt, vxy * dt)
                    dy = np.clip(ty - y, -vxy * dt, vxy * dt)
                    dz = np.clip(tz - z, -vz * dt, vz * dt)

                    p_next = np.array([x + dx, y + dy, z + dz], dtype=float)
                    debris_center_sub = np.array(data.qpos[q_debris:q_debris+3], dtype=float)
                    p_next = project_drone_outside_debris(p_next, drone_half_extents[k], debris_center_sub)
                data.qpos[q+0] = p_next[0]
                data.qpos[q+1] = p_next[1]
                data.qpos[q+2] = p_next[2]
                data.qpos[q+3:q+7] = drone_quat0[k]
                data.qvel[d:d+6] = 0.0

            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

        update_net_center_marker()
        mujoco.mj_forward(model, data)
        prev_drone_xyz = np.array([data.qpos[int(q):int(q)+3] for q in drone_qadr], dtype=float)

        bucket = int(data.time * 2)
        if bucket != last_print:
            last_print = bucket
            print(f"t={data.time:6.2f}s mode={mode:8s} dist={net_center_xy_dist:5.2f} cmd={cmd_dist:5.2f} goal={goal_dist:5.2f} alpha={alpha:5.2f} aspect={aspect:5.2f} "
                  f"halfL={halfL:4.2f} halfW={halfW:4.2f} cdist={center_dist:4.2f} vmax={max_drone_speed:4.2f} lock={len(locked_corner_pairs)} vxy={vxy:4.2f}")

        w, h = glfw.get_framebuffer_size(window)
        viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjv_updateScene(model, data, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scene)
        mujoco.mjr_render(viewport, scene, context)
        if on_render is not None:
            try:
                should_close = bool(on_render(viewport, context, data))
            except Exception as exc:
                print(f"[on_render] error: {exc}")
                should_close = True
            if should_close:
                glfw.set_window_should_close(window, True)

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    run_sim()
