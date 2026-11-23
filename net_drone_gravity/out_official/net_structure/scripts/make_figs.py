"""Figures of the net's force structure as it is in the simulation (net_sim.py, unchanged), off-screen (matplotlib Agg):
  A  top view of the flat net to scale: knots, every tendon segment of the model, corner strings, drones, a 40 mm ball
  B  the real state of catch seed 206 (x5, tier 0, hard, 2 s, 10 balls; out_official/best_video): hovering with the
     empty net, and the end with all ten balls in the net; 3-D view and side view, force structure only
  C  the same 3-D view at the end: force structure alone / with the 128 contact triangles (translucent)
The structure facts are read from the compiled MuJoCo model (built by net_sim.build_xml / net_sim.Sim with the trial's
parameters; nothing is simulated) and written to structure_facts.json.
python out_official/net_structure/scripts/make_figs.py"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(OUT, "..", ".."))
sys.path.insert(0, REPO)
import mujoco
import net_sim as N

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

TRIAL = os.path.join(REPO, "out_official", "best_video", "x5_tier0_hard2s_seed206", "trials")
SUMMARY = os.path.join(TRIAL, "catch_hard_sequence_seed206_summary.json")
LOG = os.path.join(TRIAL, "catch_hard_sequence_seed206_log.npz")

C_KNOT, C_LINK, C_STR, C_DRONE, C_BALL, C_TRI = "#1f3b73", "#4a6fa5", "#c0392b", "#333333", "#f39c12", "#2e8b57"


# ---------------------------------------------------------------- facts from the model
def model_facts():
    s = json.load(open(SUMMARY))
    p = N.catch_params({"p_over": s["cfg"]["p_over"], "seed": s["cfg"]["seed"]})
    p.n_balls = 10
    diffs = {k: (v, s["params"].get(k)) for k, v in N.asdict(p).items()
             if k in s["params"] and json.dumps(v) != json.dumps(s["params"][k])}
    sim = N.Sim(p)                       # compiles the model and applies the run-time settings (no stepping)
    m = sim.model
    n, h = p.n, p.L / (p.n - 1)
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TENDON, i) for i in range(m.ntendon)]
    net_t = [i for i, nm in enumerate(names) if nm[0] in "ab"]
    str_t = [i for i, nm in enumerate(names) if nm.startswith("str")]
    # which knot pairs each net tendon joins (from its wrap sites)
    site_knot = {int(sim.node_site[i, j]): i * n + j for i in range(n) for j in range(n)}
    pairs = []
    for i in net_t:
        adr, num = m.tendon_adr[i], m.tendon_num[i]
        ks = [site_knot[int(m.wrap_objid[adr + q])] for q in range(num)]
        pairs.append(tuple(ks))
    kinds = {"horizontal/vertical neighbour": 0, "other": 0}
    for a, b in pairs:
        ia, ja, ib, jb = a // n, a % n, b // n, b % n
        kinds["horizontal/vertical neighbour" if abs(ia - ib) + abs(ja - jb) == 1 else "other"] += 1
    kb = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"n{i}_{j}") for i in range(n) for j in range(n)]
    km = m.body_mass[kb]
    deg = np.array([(i > 0) + (i < n - 1) + (j > 0) + (j < n - 1) for i in range(n) for j in range(n)])
    hook_body = []
    for i in str_t:
        adr = m.tendon_adr[i]
        sid = int(m.wrap_objid[adr])
        hook_body.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.site_bodyid[sid])))
    solref, solimp = m.tendon_solref_lim[net_t[0]], m.tendon_solimp_lim[net_t[0]]
    tc, dr, dmax = float(solref[0]), float(solref[1]), float(solimp[1])
    cq = N.net_drag_coefficients(p)
    kgeom = [int(m.body_geomadr[b]) for b in kb]
    tri = N.net_triangles(n)
    F = {
        "source": "net_sim.build_xml / net_sim.Sim with the parameters of catch seed 206 (catch_params, p_over "
                  f"{s['cfg']['p_over']}); params differing from the trial summary: {diffs or 'none'}",
        "n_knots_per_side": n, "n_knots": n * n, "cell_m": h, "net_side_m": p.L, "net_mass_kg": p.m_net,
        "knot_mass_kg": {"corner (2 links)": float(km[deg == 2][0]), "edge (3 links)": float(km[deg == 3][0]),
                         "interior (4 links)": float(km[deg == 4][0]), "sum": float(km.sum()),
                         "rule": "degree / 2 x link mass, link mass = m_net / (2 n (n-1)) = "
                                 f"{p.m_net / (2 * n * (n - 1)):.4e} kg"},
        "knot_joints": "3 slide joints along the world axes (no rotation)",
        "knot_dof_damping_N_s_m": float(m.dof_damping[sim.knot_dof.ravel()].max()),
        "knot_geom": {"type": "sphere", "radius_m": float(m.geom_size[kgeom[0]][0]),
                      "contype": int(m.geom_contype[kgeom[0]]), "conaffinity": int(m.geom_conaffinity[kgeom[0]])},
        "net_tendons": {"count": len(net_t), "pairs": kinds, "diagonals": kinds["other"],
                        "type": "spatial tendon between the sites of two knots",
                        "limited": bool(m.tendon_limited[net_t[0]]), "range_m": m.tendon_range[net_t[0]].tolist(),
                        "stiffness": float(m.tendon_stiffness[net_t].max()),
                        "damping": float(m.tendon_damping[net_t].max()),
                        "solref_lim": [tc, dr], "solimp_lim": solimp[:3].tolist(),
                        "soft_constraint_ref_accel": {"k_1_s2": 1.0 / (dmax ** 2 * tc ** 2 * dr ** 2),
                                                      "b_1_s": 2.0 / (dmax * tc),
                                                      "note": "MuJoCo reference acceleration a = -b v - k r (per unit "
                                                              "effective mass; not a fixed N/m stiffness)"}},
        "corner_strings": {"count": len(str_t), "range_m": [m.tendon_range[i].tolist() for i in str_t],
                           "hook_body": hook_body, "hook_z_m": p.hook_z,
                           "corner_knot": [int(k) for k in sim.corner_knot],
                           "stiffness": float(m.tendon_stiffness[str_t].max()),
                           "damping": float(m.tendon_damping[str_t].max()),
                           "solref_lim": m.tendon_solref_lim[str_t[0]].tolist()},
        "thread": {"diameter_m": N.thread_diameter(p), "total_length_m": float(N.thread_length_per_knot(p).sum()),
                   "air_drag_c_N_s2_m2": {"corner": float(cq[deg == 2][0]), "edge": float(cq[deg == 3][0]),
                                          "interior": float(cq[deg == 4][0])},
                   "net_drag_on": bool(p.net_drag)},
        "drones": {"count": 4, "mass_kg": p.m_drone, "spacing_m": s["cfg"]["spacing"], "yaw_deg": p.yaw_deg,
                   "rotor_radius_m": p.prop_radius},
        "contact_triangles": {"count": len(tri), "split": "each cell (i,j)-(i+1,j)-(i+1,j+1)-(i,j+1) split along "
                                                           "the (i,j)-(i+1,j+1) diagonal",
                              **{k: v for k, v in N.NET_CONTACT.items()},
                              "force_to_knots": "minus the force on the ball, times the triangle's barycentric "
                                                "weights (clipped to [0, 1], normalised to sum 1)"},
        "ball": {"d_m": p.ball_d, "m_kg": p.ball_m, "contype": 2, "conaffinity": 3},
    }
    return p, sim, F


# ---------------------------------------------------------------- helpers
def quat_mat(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def links(n):
    L = []
    for i in range(n):
        for j in range(n):
            if i < n - 1:
                L.append((i * n + j, (i + 1) * n + j))
            if j < n - 1:
                L.append((i * n + j, i * n + j + 1))
    return L


def sphere(ax, c, r, color, alpha=1.0, res=14):
    u, v = np.linspace(0, 2 * np.pi, res), np.linspace(0, np.pi, res // 2 + 1)
    x = c[0] + r * np.outer(np.cos(u), np.sin(v))
    y = c[1] + r * np.outer(np.sin(u), np.sin(v))
    z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0, shade=True)


def rotor_circles(p, pos, R, res=24):
    lay = N.rotor_layout(p)
    a = np.linspace(0, 2 * np.pi, res)
    out = []
    for c in lay:
        ring = np.stack([c[0] + p.prop_radius * np.cos(a), c[1] + p.prop_radius * np.sin(a),
                         np.full(res, c[2])], 1)
        out.append(pos + ring @ R.T)
    return out


def state_at(Lg, t):
    ft = Lg["frames_t"]
    i = int(np.argmin(np.abs(ft - t)))
    fr = Lg["frames"][i]
    j = int(np.argmin(np.abs(Lg["t"] - ft[i])))
    return float(ft[i]), fr, Lg["frames_pos"][i], Lg["frames_quat"][i], Lg["ball_pos"][j], Lg["ball_contact"][j]


def draw3d(ax, p, st, tris=None, view=(16, -62), title="", zoom=1.35):
    t, fr, pos, quat, bp, bc = st
    n = p.n
    kx, hooks = fr[:n * n], fr[n * n:]
    if tris is not None:
        pc = Poly3DCollection([kx[tri] for tri in tris], facecolor=C_TRI, edgecolor=(0.18, 0.55, 0.34, 0.6),
                              linewidth=0.4, alpha=0.28)
        ax.add_collection3d(pc)
    ax.add_collection3d(Line3DCollection([kx[[a, b]] for a, b in links(n)], colors=C_LINK, linewidths=1.0))
    ax.scatter(*kx.T, s=9, color=C_KNOT, depthshade=False)
    corner = [(n - 1 if sx > 0 else 0) * n + (n - 1 if sy > 0 else 0) for sx, sy in N.CORNERS]
    for k in range(4):
        ax.plot(*np.stack([hooks[k], kx[corner[k]]]).T, color=C_STR, lw=1.6)
        R = quat_mat(quat[k])
        for ring in rotor_circles(p, pos[k], R):
            ax.plot(*ring.T, color=C_DRONE, lw=0.9)
        ax.scatter(*pos[k], s=14, color=C_DRONE, depthshade=False)
    for b in range(len(bp)):
        if np.all(np.isfinite(bp[b])) and bp[b][0] > -20:          # launched (parked balls sit at x = -30 m)
            sphere(ax, bp[b], p.ball_d / 2, C_BALL)
    allp = np.concatenate([kx, pos])
    c = 0.5 * (allp.min(0) + allp.max(0))
    half = 0.5 * (allp.max(0) - allp.min(0)) + 0.06
    ax.set_xlim(c[0] - half[0], c[0] + half[0]); ax.set_ylim(c[1] - half[1], c[1] + half[1])
    ax.set_zlim(c[2] - half[2], c[2] + half[2])
    ax.set_box_aspect(2 * half, zoom=zoom)
    ax.set_proj_type("ortho")
    ax.view_init(*view)
    ax.set_xlabel("x [m]", labelpad=-2); ax.set_ylabel("y [m]", labelpad=-2); ax.set_zlabel("z [m]", labelpad=-4)
    ax.tick_params(labelsize=7, pad=0)
    ax.set_title(title, fontsize=11)
    return c, half


def side_view(ax, p, st, title=""):
    t, fr, pos, quat, bp, bc = st
    n = p.n
    kx, hooks = fr[:n * n], fr[n * n:]
    for a, b in links(n):
        ax.plot(kx[[a, b], 0], kx[[a, b], 2], color=C_LINK, lw=0.8, alpha=0.7)
    ax.scatter(kx[:, 0], kx[:, 2], s=6, color=C_KNOT, zorder=3)
    corner = [(n - 1 if sx > 0 else 0) * n + (n - 1 if sy > 0 else 0) for sx, sy in N.CORNERS]
    for k in range(4):
        ax.plot([hooks[k][0], kx[corner[k]][0]], [hooks[k][2], kx[corner[k]][2]], color=C_STR, lw=1.6)
        for ring in rotor_circles(p, pos[k], quat_mat(quat[k])):
            ax.plot(ring[:, 0], ring[:, 2], color=C_DRONE, lw=0.9)
    for b in range(len(bp)):
        if np.all(np.isfinite(bp[b])) and bp[b][0] > -20:
            ax.add_patch(Circle((bp[b][0], bp[b][2]), p.ball_d / 2, color=C_BALL, zorder=4))
    zc = kx[(n // 2) * n + n // 2, 2]
    zd = pos[:, 2].mean()
    ax.annotate("", xy=(pos[:, 0].max() + 0.08, zc), xytext=(pos[:, 0].max() + 0.08, zd),
                arrowprops=dict(arrowstyle="<->", color="k", lw=0.8))
    ax.text(pos[:, 0].max() + 0.1, 0.5 * (zc + zd), f"无人机均高 − 网心\n{1e3 * (zd - zc):.0f} mm", fontsize=8,
            va="center")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]"); ax.set_ylabel("z [m]")
    ax.grid(alpha=0.25)
    ax.set_title(title, fontsize=11)
    x0 = pos[:, 0].mean()
    ax.set_xlim(x0 - 0.85, x0 + 1.05)


# ---------------------------------------------------------------- figure A
def fig_a(p, F):
    n, h, L = p.n, F["cell_m"], p.L
    D = F["drones"]["spacing_m"]
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 8.6), gridspec_kw={"width_ratios": [2.15, 1]})
    g = np.array([[i * h - L / 2, j * h - L / 2] for i in range(n) for j in range(n)])
    for a, b in links(n):
        ax.plot(g[[a, b], 0], g[[a, b], 1], color=C_LINK, lw=1.3, zorder=1)
    ax.scatter(*g.T, s=22, color=C_KNOT, zorder=3)
    lay = N.rotor_layout(p)
    for k, (sx, sy) in enumerate(N.CORNERS):
        dc = np.array([sx * D / 2, sy * D / 2])
        ck = np.array([sx * L / 2, sy * L / 2])
        ax.plot([dc[0], ck[0]], [dc[1], ck[1]], color=C_STR, lw=2.2, zorder=2)
        yaw = np.radians(p.yaw_deg)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        for c in lay:
            ax.add_patch(Circle(dc + Rz @ c[:2], p.prop_radius, fill=False, color=C_DRONE, lw=1.0))
        ax.scatter(*dc, s=16, color=C_DRONE, zorder=4)
    # ball in the cell left of / below the centre knot, to scale
    bc = np.array([-h / 2, -h / 2])
    ax.add_patch(Circle(bc, p.ball_d / 2, color=C_BALL, zorder=5))
    # dimensions
    def dim(p0, p1, txt, off, rot=0, fs=9):
        ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="<->", lw=0.9, color="k"))
        mid = 0.5 * (np.array(p0) + np.array(p1)) + np.array(off)
        ax.text(*mid, txt, ha="center", va="center", fontsize=fs, rotation=rot,
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    dim((-L / 2, -L / 2 - 0.09), (L / 2, -L / 2 - 0.09), f"网边长 {L:.2f} m（{n} × {n} 个结）", (0, 0))
    dim((-D / 2, D / 2 + 0.11), (D / 2, D / 2 + 0.11), f"无人机间距（正方形边长）{D:.2f} m", (0, 0))
    dim((L / 2 + 0.03, 0.0), (L / 2 + 0.03, h), f"格距\n{1e3 * h:.0f} mm", (0.075, 0), fs=8)
    ax.annotate(f"角绳长 {p.string:.2f} m\n（平铺时俯视投影 {np.hypot(*(np.array([D, D]) - L) / 2) * 1e3:.0f} mm）",
                xy=(0.5 * (D + L) / 2, -0.5 * (D + L) / 2), xytext=(0.70, -0.40), fontsize=8.5, color=C_STR,
                ha="left", va="center", arrowprops=dict(arrowstyle="-", color=C_STR, lw=0.7))
    nt, ns = F["net_tendons"]["count"], F["corner_strings"]["count"]
    ax.set_title(f"图 A  平铺的网（俯视，真实比例）：{n * n} 个结、{nt} 段线（只连上下左右相邻结，无斜线、无单独边绳）、"
                 f"{ns} 根角绳", fontsize=11.5)
    ax.set_aspect("equal")
    ax.set_xlim(-0.82, 1.08); ax.set_ylim(-0.78, 0.85)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.grid(alpha=0.2)
    # zoom of the cell with the ball
    i0 = n // 2 - 1
    x0, y0 = -h, -h
    sub = [(a, b) for a, b in links(n) if all(x0 - 1e-9 <= g[q][0] <= x0 + h + 1e-9 and y0 - 1e-9 <= g[q][1] <= y0 + h + 1e-9 for q in (a, b))]
    for a, b in sub:
        ax2.plot(g[[a, b], 0], g[[a, b], 1], color=C_LINK, lw=2.5)
    pts = g[[q for q in range(n * n) if x0 - 1e-9 <= g[q][0] <= x0 + h + 1e-9 and y0 - 1e-9 <= g[q][1] <= y0 + h + 1e-9]]
    ax2.scatter(*pts.T, s=90, color=C_KNOT, zorder=3)
    for q in pts:
        ax2.add_patch(Circle(q, F["knot_geom"]["radius_m"], fill=False, color=C_KNOT, lw=0.8, ls=":"))
    ax2.add_patch(Circle(bc, p.ball_d / 2, color=C_BALL, zorder=5))
    ax2.annotate("", xy=(bc[0] + p.ball_d / 2, bc[1] - 0.03), xytext=(bc[0] - p.ball_d / 2, bc[1] - 0.03),
                 arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax2.text(bc[0], bc[1] - 0.04, f"球 {1e3 * p.ball_d:.0f} mm", ha="center", va="top", fontsize=10)
    ax2.annotate("", xy=(x0 + h, y0 - 0.012), xytext=(x0, y0 - 0.012), arrowprops=dict(arrowstyle="<->", lw=0.9))
    ax2.text(x0 + h / 2, y0 - 0.017, f"{1e3 * h:.0f} mm", ha="center", va="top", fontsize=10)
    ax2.set_aspect("equal")
    ax2.set_xlim(x0 - 0.02, x0 + h + 0.02); ax2.set_ylim(y0 - 0.03, y0 + h + 0.02)
    ax2.set_title("放大：中心结左下的一格和一个 40 mm 球\n（虚线圈＝结的碰撞小球，半径 4 mm，只与地面碰撞）", fontsize=10.5)
    ax2.set_xticks([]); ax2.set_yticks([])
    km = F["knot_mass_kg"]
    T = F["net_tendons"]
    txt = (f"结：{n * n} 个质点，每个用 3 个沿世界轴的滑动关节（不转动）。质量按连到它的线段数分摊，"
           f"每段 {1e3 * F['net_mass_kg'] / T['count']:.3f} g 的一半："
           f"角结 {1e3 * km['corner (2 links)']:.3f} g，边结 {1e3 * km['edge (3 links)']:.3f} g，内结 {1e3 * km['interior (4 links)']:.3f} g，"
           f"合计 {1e3 * km['sum']:.1f} g。\n"
           f"线段：{T['count']} 段 MuJoCo 空间腱，只有长度上限 {1e3 * T['range_m'][1]:.0f} mm（超过即软约束顶住），"
           f"弹簧刚度 {T['stiffness']:g}、阻尼 {T['damping']:g}；所以只能受拉，松了不传力；线本身无质量。"
           f"角绳：同样的腱，上限 {p.string:.2f} m，从无人机机体下 {-1e3 * p.hook_z:.0f} mm 的挂点连到角结。\n"
           f"示意图按平铺画（线全部拉直）；真实悬停时网下垂、角结被拉向内（见图 B）。")
    fig.text(0.015, 0.012, txt, fontsize=9.2, va="bottom", ha="left", wrap=True)
    fig.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.17, wspace=0.08)
    fig.savefig(os.path.join(OUT, "A_top_view_flat.png"), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- figures B and C
def fig_b(p, F, Lg, s):
    t_h = s["t_launch0_s"] - 1.0
    st_h, st_e = state_at(Lg, t_h), state_at(Lg, Lg["frames_t"][-1])
    fig = plt.figure(figsize=(16, 10.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1])
    for r, (st, lab) in enumerate(((st_h, "空网悬停"), (st_e, "末尾：10 个球都在网里"))):
        ax = fig.add_subplot(gs[r, 0], projection="3d")
        draw3d(ax, p, st, title=f"{lab}（t = {st[0]:.2f} s）", zoom=1.0)
        ax2 = fig.add_subplot(gs[r, 1])
        nb = int(np.sum(st[5])) if r else 0
        side_view(ax2, p, st, title=f"{lab}：侧视（沿 y 方向看，真实比例）")
    fig.suptitle("图 B  种子 206 那次仿真里的真实形状（只画受力结构：结、线段、角绳、无人机；球按真实大小）",
                 fontsize=12.5, y=0.985)
    fig.text(0.015, 0.01, "数据：out_official/best_video/x5_tier0_hard2s_seed206/trials 的日志（结和挂点位置每 0.05 s 存一帧，"
             "球位置 100 Hz）。深蓝点＝结，蓝线＝结间的腱，红线＝角绳，黑圈＝旋翼盘（半径 27.5 mm），橙球＝球。",
             fontsize=9.2)
    fig.subplots_adjust(left=0.0, right=0.97, top=0.92, bottom=0.04, wspace=0.08, hspace=0.12)
    fig.savefig(os.path.join(OUT, "B_real_state_seed206.png"), dpi=140)
    plt.close(fig)
    return st_h, st_e


def fig_c(p, F, st_e):
    tri = N.net_triangles(p.n)
    fig = plt.figure(figsize=(15, 8.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    draw3d(ax, p, st_e, view=(22, -62), title="只有受力结构（结 + 腱 + 角绳）", zoom=1.0)
    ax = fig.add_subplot(1, 2, 2, projection="3d")
    draw3d(ax, p, st_e, tris=tri, view=(22, -62), title=f"叠加接触面：{len(tri)} 片三角形（半透明绿）", zoom=1.0)
    C = F["contact_triangles"]
    fig.suptitle(f"图 C  同一时刻（t = {st_e[0]:.2f} s，10 个球在网里），同一视角：球实际是被这层三角面挡住的",
                 fontsize=12.5)
    txt = (f"接触面：每格沿同一条对角线切成两片三角形，顶点就是该格的 4 个结（共 {len(tri)} 片，面本身无质量、不参与受力结构；"
           f"对角线方向并没有线）。球与腱、与结的小球都不碰撞（碰撞分组），只与这层面、地面、其他球和无人机外壳接触。\n"
           f"罚函数接触：球进到某片三角面半径以内时，法向力 = {C['k_N_m']:g} N/m × 压入深度 − {C['c_N_s_m']:g} N·s/m × 法向相对速度（不小于 0），"
           f"切向粘性摩擦 {C['ct_N_s_m']:g} N·s/m、上限 {C['mu']:g} × 法向力；\n"
           f"三角形边缘 ±{1e3 * C['edge_margin_m']:.0f} mm 内与邻片平滑过渡；"
           f"球在面另一侧超过 {1e3 * C['depth_max_m']:.0f} mm 时该面不再起作用。\n"
           f"力怎么回到受力结构：作用在球上的力取反，按球在该三角形里的重心坐标（截到 0–1 后归一）分给三角形的 3 个结，"
           f"再由腱、角绳传到无人机。")
    fig.text(0.015, 0.012, txt, fontsize=9.2, va="bottom", wrap=True)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.15, wspace=0.02)
    fig.savefig(os.path.join(OUT, "C_structure_vs_contact_faces.png"), dpi=140)
    plt.close(fig)


def main():
    p, sim, F = model_facts()
    s = json.load(open(SUMMARY))
    Lg = dict(np.load(LOG))
    fig_a(p, F)
    st_h, st_e = fig_b(p, F, Lg, s)
    fig_c(p, F, st_e)
    n = p.n
    for name, st in (("hover", st_h), ("end", st_e)):
        kx, pos = st[1][:n * n], st[2]
        F[f"state_{name}"] = {"t_s": st[0], "drone_mean_z_m": float(pos[:, 2].mean()),
                              "net_centre_z_m": float(kx[(n // 2) * n + n // 2, 2]),
                              "net_lowest_z_m": float(kx[:, 2].min()),
                              "drone_spacing_mean_m": float(np.mean([np.linalg.norm(pos[a, :2] - pos[(a + 1) % 4, :2])
                                                                     for a in range(4)])),
                              "balls_in_contact": int(np.sum(st[5]))}
    json.dump(F, open(os.path.join(OUT, "structure_facts.json"), "w"), indent=1, ensure_ascii=False,
              default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(json.dumps(F, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()
