import math
import os


HERE = os.path.dirname(os.path.abspath(__file__))


def _key3(p, ndigits=9):
    return (round(float(p[0]), ndigits), round(float(p[1]), ndigits), round(float(p[2]), ndigits))


def _linspace(a, b, n):
    if n <= 1:
        return [float(a)]
    step = (float(b) - float(a)) / float(n - 1)
    return [float(a) + i * step for i in range(n)]


def _add3(a, b):
    return (float(a[0]) + float(b[0]), float(a[1]) + float(b[1]), float(a[2]) + float(b[2]))


def _dist3(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    dz = float(a[2]) - float(b[2])
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def build_open_cube_net_xml(
    n_edge=7,
    cube_size=4.0,
    center=(0.0, 0.0, 1.5),
    timestep=0.001,
    gravity=(0.0, 0.0, 0.0),
    node_radius=0.08,
    node_density=40.0,
    drone_size=(0.16, 0.16, 0.10),
    drone_density=120.0,
    tendon_stiffness=1600.0,
    tendon_damping=8.0,
    floor_z=-2.0,
    checker_tile_size=1.0,
    checker_half_tiles_x=12,
    checker_half_tiles_y=12,
    checker_tile_half_thickness=0.015,
    checker_color_a=(0.92, 0.92, 0.92, 1.0),
    checker_color_b=(0.10, 0.10, 0.10, 1.0),
    payload_pos=(4.0, -2.5, 0.6),
    payload_half_size=(0.65, 0.65, 0.65),
    payload_density=300.0,
    payload_rgba=(0.2, 0.75, 0.25, 1.0),
    payload2_pos=(-3.6, 2.8, 0.6),
    payload2_half_size=(0.60, 0.60, 0.60),
    payload2_density=300.0,
    payload2_rgba=(0.70, 0.20, 0.90, 1.0),
):
    """
    Open-cube net:
    - 5 faces (z+, opening face removed)
    - 4 drones at opening corners
    - 1 drone at opposite face center
    - no extra constraints beyond base mesh tendons
    - two free box payloads in scene (green + purple)
    """
    if int(n_edge) < 3:
        raise ValueError("n_edge must be >= 3.")
    if int(n_edge) % 2 == 0:
        raise ValueError("n_edge must be odd so opposite-face center exists on the grid.")

    n_edge = int(n_edge)
    cube_size = float(cube_size)
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    half = 0.5 * cube_size

    # 5 faces: z-, x-, x+, y-, y+ (z+ is open)
    face_defs = {
        "z_neg": lambda u, v: (u, v, -half),
        "x_neg": lambda u, v: (-half, u, v),
        "x_pos": lambda u, v: (+half, u, v),
        "y_neg": lambda u, v: (u, -half, v),
        "y_pos": lambda u, v: (u, +half, v),
    }

    vals = _linspace(-half, half, n_edge)
    key_to_idx = {}
    node_xyz = []
    face_grids = {}

    # Build and deduplicate face nodes.
    for fname, fpos in face_defs.items():
        grid = [[0 for _ in range(n_edge)] for _ in range(n_edge)]
        for i, u in enumerate(vals):
            for j, v in enumerate(vals):
                p_local = fpos(u, v)
                p_world = _add3(p_local, (cx, cy, cz))
                k = _key3(p_world)
                idx = key_to_idx.get(k)
                if idx is None:
                    idx = len(node_xyz)
                    key_to_idx[k] = idx
                    node_xyz.append(p_world)
                grid[i][j] = idx
        face_grids[fname] = grid

    # Build structural edges from 4-neighbor connectivity on each face.
    edges = set()

    def add_edge(a, b):
        if a == b:
            return
        if a > b:
            a, b = b, a
        edges.add((a, b))

    for grid in face_grids.values():
        for i in range(n_edge - 1):
            for j in range(n_edge):
                add_edge(int(grid[i][j]), int(grid[i + 1][j]))
        for i in range(n_edge):
            for j in range(n_edge - 1):
                add_edge(int(grid[i][j]), int(grid[i][j + 1]))

    # Drone anchors.
    opening_corners = [
        ("drone_0_0", (-half, -half, +half)),
        ("drone_10_0", (+half, -half, +half)),
        ("drone_0_10", (-half, +half, +half)),
        ("drone_10_10", (+half, +half, +half)),
    ]
    opposite_center = ("drone_5_5", (0.0, 0.0, -half))
    drone_specs = opening_corners + [opposite_center]

    drone_node_to_name = {}
    for dname, p_local in drone_specs:
        p_world = _add3(p_local, (cx, cy, cz))
        idx = key_to_idx.get(_key3(p_world))
        if idx is None:
            raise RuntimeError(f"Missing drone anchor point: {dname} at {p_world}.")
        drone_node_to_name[idx] = dname

    if len(drone_node_to_name) != 5:
        raise RuntimeError("Expected exactly 5 distinct drone anchor nodes.")

    xml = []
    xml.append('<mujoco model="open_cube_net_fishing">')
    xml.append('  <compiler angle="degree" coordinate="local"/>')
    xml.append(
        f'  <option timestep="{timestep}" gravity="{gravity[0]} {gravity[1]} {gravity[2]}" integrator="RK4"/>'
    )
    xml.append('  <size nconmax="20000" njmax="20000"/>')
    xml.append('  <visual>')
    xml.append('    <headlight active="1" ambient="0.42 0.42 0.42" diffuse="0.70 0.70 0.70" specular="0.20 0.20 0.20"/>')
    xml.append('  </visual>')
    xml.append('  <worldbody>')
    xml.append('    <light name="key_top" directional="true" castshadow="false" pos="0 0 14" dir="0 0 -1" diffuse="0.95 0.95 0.95" specular="0.25 0.25 0.25"/>')
    xml.append('    <light name="fill_posx" directional="true" castshadow="false" pos="9 6 9" dir="-0.7 -0.4 -0.6" diffuse="0.55 0.55 0.55" specular="0.10 0.10 0.10"/>')
    xml.append('    <light name="fill_negx" directional="true" castshadow="false" pos="-9 -6 9" dir="0.7 0.4 -0.6" diffuse="0.50 0.50 0.50" specular="0.08 0.08 0.08"/>')
    xml.append('    <light name="back_rim" directional="true" castshadow="false" pos="0 -10 7" dir="0 0.85 -0.5" diffuse="0.40 0.40 0.42" specular="0.06 0.06 0.07"/>')
    # Visual checkerboard floor (black/white style)
    tile_size = max(float(checker_tile_size), 1e-6)
    half_x = max(int(checker_half_tiles_x), 1)
    half_y = max(int(checker_half_tiles_y), 1)
    floor_z = float(floor_z)
    half_t = max(float(checker_tile_half_thickness), 1e-4)
    half_s = 0.5 * tile_size
    c1 = checker_color_a
    c2 = checker_color_b
    for ix in range(-half_x, half_x):
        for iy in range(-half_y, half_y):
            cx_tile = (ix + 0.5) * tile_size
            cy_tile = (iy + 0.5) * tile_size
            rgba = c1 if ((ix + iy) & 1) == 0 else c2
            xml.append(
                f'    <geom name="checker_{ix + half_x}_{iy + half_y}" type="box" '
                f'pos="{cx_tile:.4f} {cy_tile:.4f} {floor_z:.4f}" '
                f'size="{half_s:.4f} {half_s:.4f} {half_t:.4f}" '
                f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" contype="0" conaffinity="0"/>'
            )

    xml.append(f'    <body name="net_center_marker" mocap="true" pos="{cx:.5f} {cy:.5f} {cz:.5f}">')
    xml.append('      <geom name="net_center_geom" type="sphere" size="0.10" rgba="1 1 1 0.28" contype="0" conaffinity="0"/>')
    xml.append('    </body>')

    # Small block payload for interaction tests.
    ppx, ppy, ppz = float(payload_pos[0]), float(payload_pos[1]), float(payload_pos[2])
    psx, psy, psz = float(payload_half_size[0]), float(payload_half_size[1]), float(payload_half_size[2])
    pr, pg, pb, pa = float(payload_rgba[0]), float(payload_rgba[1]), float(payload_rgba[2]), float(payload_rgba[3])
    xml.append(f'    <body name="payload" pos="{ppx:.5f} {ppy:.5f} {ppz:.5f}">')
    xml.append('      <joint name="j_payload" type="free"/>')
    xml.append(
        f'      <geom name="payload_geom" type="box" '
        f'size="{psx:.5f} {psy:.5f} {psz:.5f}" density="{float(payload_density):.5f}" '
        f'rgba="{pr:.5f} {pg:.5f} {pb:.5f} {pa:.5f}" '
        f'contype="1" conaffinity="1" friction="1.1 0.2 0.2"/>'
    )
    xml.append('    </body>')

    # Second (purple) payload.
    p2x, p2y, p2z = float(payload2_pos[0]), float(payload2_pos[1]), float(payload2_pos[2])
    s2x, s2y, s2z = float(payload2_half_size[0]), float(payload2_half_size[1]), float(payload2_half_size[2])
    r2, g2, b2, a2 = float(payload2_rgba[0]), float(payload2_rgba[1]), float(payload2_rgba[2]), float(payload2_rgba[3])
    xml.append(f'    <body name="payload_purple" pos="{p2x:.5f} {p2y:.5f} {p2z:.5f}">')
    xml.append('      <joint name="j_payload_purple" type="free"/>')
    xml.append(
        f'      <geom name="payload_purple_geom" type="box" '
        f'size="{s2x:.5f} {s2y:.5f} {s2z:.5f}" density="{float(payload2_density):.5f}" '
        f'rgba="{r2:.5f} {g2:.5f} {b2:.5f} {a2:.5f}" '
        f'contype="1" conaffinity="1" friction="1.1 0.2 0.2"/>'
    )
    xml.append('    </body>')

    sx, sy, sz = float(drone_size[0]), float(drone_size[1]), float(drone_size[2])
    for idx, p in enumerate(node_xyz):
        x, y, z = float(p[0]), float(p[1]), float(p[2])
        if idx in drone_node_to_name:
            bname = drone_node_to_name[idx]
            parts = bname.split("_")
            if len(parts) != 3:
                raise RuntimeError(f"Drone name format must be drone_i_j: {bname}")
            joint_name = f"j_{parts[1]}_{parts[2]}"
            geom = (
                f'<geom name="dronegeom_{parts[1]}_{parts[2]}" type="box" '
                f'size="{sx:.5f} {sy:.5f} {sz:.5f}" density="{float(drone_density):.5f}" '
                f'rgba="0.2 0.2 0.85 1" contype="2" conaffinity="1" friction="1.1 0.2 0.2"/>'
            )
            body_name = bname
        else:
            body_name = f"n_{idx}"
            joint_name = f"j_n_{idx}"
            geom = (
                f'<geom name="nodegeom_{idx}" type="sphere" size="{float(node_radius):.5f}" '
                f'density="{float(node_density):.5f}" rgba="0.85 0.2 0.2 1" '
                f'contype="1" conaffinity="1" friction="1.1 0.2 0.2"/>'
            )

        xml.append(f'    <body name="{body_name}" pos="{x:.5f} {y:.5f} {z:.5f}">')
        xml.append(f'      <joint name="{joint_name}" type="free"/>')
        xml.append(f'      {geom}')
        xml.append(f'      <site name="s_{idx}" size="0.01"/>')
        xml.append('    </body>')

    xml.append('  </worldbody>')
    xml.append('  <tendon>')

    # Base mesh tendons.
    for a, b in sorted(edges):
        rest = _dist3(node_xyz[a], node_xyz[b])
        xml.append(
            f'    <spatial name="t_{a}_{b}" stiffness="{float(tendon_stiffness):.5f}" '
            f'damping="{float(tendon_damping):.5f}" springlength="{rest:.6f}" width="0.01">'
        )
        xml.append(f'      <site site="s_{a}"/>')
        xml.append(f'      <site site="s_{b}"/>')
        xml.append('    </spatial>')

    xml.append('  </tendon>')
    xml.append('</mujoco>')

    return "\n".join(xml), node_xyz, edges, drone_node_to_name


def write_xml(output_name="net_drone_fishing.xml", **kwargs):
    xml_str, node_xyz, edges, drone_node_to_name = build_open_cube_net_xml(**kwargs)
    out_path = os.path.join(HERE, output_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(xml_str)
    print(
        f"Generated: {out_path}\n"
        f"nodes={len(node_xyz)}, edges={len(edges)}, drones={len(drone_node_to_name)}"
    )


if __name__ == "__main__":
    write_xml()
