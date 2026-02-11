import numpy as np


def build_cloth_xml(
    nx=5,
    ny=5,
    width=6.0,
    height=6.0,
    base_z=2.0,
    # --- “线”离散参数（让连接线也能碰撞） ---
    edge_subdiv=4,
    edge_node_radius=0.08,
    edge_node_density=1.0,
    # --- payload 参数 ---
    payload_radius=1.55,
    payload_mass=960.0,
    payload_pos=None,  # None -> (0,0, base_z + payload_radius + 0.8)
):
    '''
    v4:
    - 给所有与“网”相关的 geom 自动加 name（便于 contact 统计）
      * 主节点：nodegeom_n_i_j / nodegeom_drone_i_j
      * 边小球链：edgegeom_edge_h_i_j_k / edgegeom_edge_v_i_j_k
    - thrZ 支持正负推力（允许下压/抬起）
    - payload 默认生成在网的上方，geom 名为 payload_geom
    - 连接线用 edge_subdiv 小球链实现碰撞
    输出：cloth_drones.xml
    '''

    assert nx == 5 and ny == 5

    dx = width / (nx - 1)
    dy = height / (ny - 1)

    cx, cy = nx // 2, ny // 2

    def node_pos(i, j):
        x = -width / 2 + i * dx
        y = -height / 2 + j * dy
        z = base_z
        return x, y, z

    if payload_pos is None:
        payload_pos = (0.0, 0.0, float(base_z) + float(payload_radius) + 0.8)

    r = float(payload_radius)
    vol = (4.0 / 3.0) * np.pi * (r ** 3)
    payload_density = float(payload_mass) / vol
    px, py, pz = payload_pos

    xml = []
    xml.append('<mujoco model="net_wrap_5x5_contactdriven_v4">')
    xml.append('  <compiler angle="degree" autolimits="true"/>')
    xml.append('  <option timestep="0.002" gravity="0 0 -9.81"/>')
    xml.append('  <default>')
    xml.append('    <geom friction="1.4 0.01 0.001" solref="0.002 1" solimp="0.9 0.95 0.001"/>')
    xml.append('  </default>')

    xml.append('  <worldbody>')
    xml.append('    <light diffuse="0.8 0.8 0.8" pos="0 0 50"/>')
    xml.append('    <geom type="plane" size="50 50 0.1" rgba="0.92 0.92 0.92 1"/>')

    # payload
    xml.append(f'    <body name="payload" pos="{px:.3f} {py:.3f} {pz:.3f}">')
    xml.append('      <joint name="payload_free" type="free" damping="0.03"/>')
    xml.append(
        f'      <geom name="payload_geom" type="sphere" size="{r:.3f}" '
        f'density="{payload_density:.3f}" rgba="0.2 0.8 0.2 1"/>'
    )
    xml.append('    </body>')

    # main nodes
    for j in range(ny):
        for i in range(nx):
            x, y, z = node_pos(i, j)
            is_corner = (i in (0, nx - 1)) and (j in (0, ny - 1))
            #is_center = (i == cx) and (j == cy)
            is_drone = is_corner #or is_center

            if is_drone:
                bname = f"drone_{i}_{j}"
                jnt = f'<joint name="j_{i}_{j}" type="free" damping="0.03"/>'
                gname = f'nodegeom_drone_{i}_{j}'
                geom = f'<geom name="{gname}" type="box" size="0.25 0.25 0.15" density="40" rgba="0.2 0.2 0.8 1"/>'
            else:
                bname = f"n_{i}_{j}"
                jnt = f'<joint name="j_{i}_{j}" type="free" damping="0.03"/>'
                gname = f'nodegeom_n_{i}_{j}'
                geom = f'<geom name="{gname}" type="sphere" size="0.15" density="18" rgba="0.85 0.25 0.25 1"/>'

            xml.append(f'    <body name="{bname}" pos="{x:.3f} {y:.3f} {z:.3f}">')
            xml.append(f'      {jnt}')
            xml.append(f'      {geom}')
            xml.append(f'      <site name="s_{i}_{j}" size="0.04"/>')
            xml.append('    </body>')

    # edge nodes
    def add_edge_node(name, x, y, z):
        xml.append(f'    <body name="{name}" pos="{x:.3f} {y:.3f} {z:.3f}">')
        xml.append(f'      <joint name="j_{name}" type="free" damping="0.03"/>')
        xml.append(
            f'      <geom name="edgegeom_{name}" type="sphere" size="{edge_node_radius:.3f}" '
            f'density="{edge_node_density:.3f}" rgba="0.75 0.75 0.75 1"/>'
        )
        xml.append(f'      <site name="s_{name}" size="0.03"/>')
        xml.append('    </body>')

    for j in range(ny):
        for i in range(nx - 1):
            x0, y0, z0 = node_pos(i, j)
            x1, y1, z1 = node_pos(i + 1, j)
            for k in range(1, edge_subdiv + 1):
                a = k / (edge_subdiv + 1)
                add_edge_node(f"edge_h_{i}_{j}_{k}",
                              (1 - a) * x0 + a * x1,
                              (1 - a) * y0 + a * y1,
                              (1 - a) * z0 + a * z1)

    for j in range(ny - 1):
        for i in range(nx):
            x0, y0, z0 = node_pos(i, j)
            x1, y1, z1 = node_pos(i, j + 1)
            for k in range(1, edge_subdiv + 1):
                a = k / (edge_subdiv + 1)
                add_edge_node(f"edge_v_{i}_{j}_{k}",
                              (1 - a) * x0 + a * x1,
                              (1 - a) * y0 + a * y1,
                              (1 - a) * z0 + a * z1)

    xml.append('  </worldbody>')

    # tendons
    xml.append('  <tendon>')
    stiffness = 1400.0
    damping = 6.0

    def add_edge_tendon(tname, site_list, rest_len):
        xml.append(
            f'    <spatial name="{tname}" stiffness="{stiffness}" damping="{damping}" '
            f'springlength="{rest_len:.4f}" width="0.02">'
        )
        for s in site_list:
            xml.append(f'      <site site="{s}"/>')
        xml.append('    </spatial>')

    for j in range(ny):
        for i in range(nx - 1):
            sites = [f"s_{i}_{j}"] + [f"s_edge_h_{i}_{j}_{k}" for k in range(1, edge_subdiv + 1)] + [f"s_{i+1}_{j}"]
            add_edge_tendon(f"t_h_{i}_{j}", sites, dx)

    for j in range(ny - 1):
        for i in range(nx):
            sites = [f"s_{i}_{j}"] + [f"s_edge_v_{i}_{j}_{k}" for k in range(1, edge_subdiv + 1)] + [f"s_{i}_{j+1}"]
            add_edge_tendon(f"t_v_{i}_{j}", sites, dy)

    xml.append('  </tendon>')

    # actuators
    xml.append('  <actuator>')
    drone_points = [(0, 0), (nx - 1, 0), (0, ny - 1), (nx - 1, ny - 1), (cx, cy)]
    for (i, j) in drone_points:
        xml.append(f'    <general name="thrX_{i}_{j}" joint="j_{i}_{j}" gear="1 0 0 0 0 0" forcerange="-250 250"/>')
        xml.append(f'    <general name="thrY_{i}_{j}" joint="j_{i}_{j}" gear="0 1 0 0 0 0" forcerange="-250 250"/>')
        xml.append(f'    <general name="thrZ_{i}_{j}" joint="j_{i}_{j}" gear="0 0 1 0 0 0" forcerange="-600 600"/>')
    xml.append('  </actuator>')

    xml.append('</mujoco>')
    return "\n".join(xml)


if __name__ == "__main__":
    xml_str = build_cloth_xml(edge_subdiv=4, edge_node_radius=0.08)
    with open("cloth_drones.xml", "w", encoding="utf-8") as f:
        f.write(xml_str)
    print("✅ cloth_drones.xml 已生成：geom 已命名（便于接触统计）；thrZ ±；payload 在网的上方")
