"""Four Crazyflie 2.1 Brushless quadrotors holding a net by its corners (MuJoCo).

Fully headless: no viewer and no OpenGL context is ever created. Figures are
drawn with matplotlib's Agg backend from the logged state.

    python net_sim.py --mode sweep                    # quasi-static spacing sweep (run this first)
    python net_sim.py --mode hold --ff --gif          # take off at 1.20 m, spread to 1.25 m, hold, gust, land
    python net_sim.py --mode hold                     # same, PID only (no feedforward)
    python net_sim.py --mode hold --ff --spacing 1.20 # hold at another spacing (default 1.25 m)
    python net_sim.py --mode hold --ff --drone-model simple   # earlier simplified drone (needs its own sweep)
    python net_sim.py --mode hold --ff --takeoff-spacing 0.80 # old flight plan: take off sitting on the net
    python net_sim.py --mode hold --ff --yaw-deg 0 --hook-z -0.010    # earlier default heading / hook height
                                                      # (needs a sweep table made with that hook height)

Two drone models (--drone-model, default "official"):
  * official : Crazyflie 2.1 Brushless with the firmware's platform defaults and the
               system-identified Crazyflow parameter set (see OfficialParams): quadratic
               rpm -> thrust / torque maps, asymmetric rotor speed ODE (~60 ms), body drag
  * simple   : the earlier hand-assumed model (first-order 20 ms thrust lag, 30 gf per
               motor, no drag), kept unchanged for comparison

Runs from any directory: results go to an "out_official" (official) or "out" (simple)
folder next to this file unless --out is given, and a bare --ff uses the sweep table
found in that folder (so each drone model needs its own sweep).
All documented results use the default hold spacing of 1.25 m.

Model
  * drone : free rigid body; simple: 4 rotor thrust actuators (first-order lag,
            saturating); official: rotor speeds integrated by this script every step
            and applied to the body as a thrust / torque / drag wrench
  * net   : N x N knots (point masses) joined by inextensible strings, modelled
            as length-limited spatial tendons (slack in compression, taut in tension)
  * each net corner hangs from a drone by a short string
  * no collision between the net and the drones (knots and drones only touch the floor);
    instead every hold run reports the clearance from the net and strings to the rotor discs
    (blade strike) and, separately, to the real propeller guards (contact / snag; Crazyflow's guard
    mesh, embedded as a point set, see GUARD_STL), per flight phase (see clearance_analysis)
  * --yaw-deg turns all four drones about the vertical (initial attitude and heading reference);
    --hook-z moves the string hook (a hook below the guard feet, -15.4 mm, needs longer landing legs)
Flight plan (hold): the net lies flat on the floor (corners at about +-0.49 m); the drones
  start outside its corners at the takeoff spacing (default 1.20 m, strings slack), climb at
  that spacing, spread to the hold spacing, hold, take a gust, relax to 1.20 m and land.
Control (per drone, no communication between drones)
  * 100 Hz position PID -> desired thrust vector
  * 500 Hz geometric attitude PD with a small integrator (removes the steady tilt
    error from the string pulling below the centre of mass) -> body torques -> rotor mixer
  Net load:
  * without --ff the controller knows nothing about the net; the position
    integrator absorbs its pull and weight
  * with --ff the position loop also adds a feedforward force: the net's horizontal
    pull per drone, looked up by the reference spacing in the sweep table (linear
    interpolation, edge value held outside the table), plus a quarter of the net
    weight, both scaled by the share of the net that is off the floor (reference
    height / height at which the whole net hangs free); the PID corrects the rest
"""
import argparse
import json
import math
import os
import time
from dataclasses import dataclass, asdict, field

import numpy as np
import mujoco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

G = 9.81
HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Params:
    # --- Crazyflie 2.1 Brushless (bitcraze.io product page) ---
    m_drone: float = 0.037          # kg, takeoff weight with guards
    f_max: float = 0.030 * G        # N per motor ("up to 30 grams thrust each")
    arm: float = 0.050              # m, centre to motor (100 mm diagonal frame)
    # --- assumed, not published ---
    inertia: tuple = (3.0e-5, 3.0e-5, 4.5e-5)   # kg m^2, scaled from the CF2.x model
    kappa: float = 0.006            # m, rotor drag-torque / thrust
    motor_tau: float = 0.020        # s, rotor thrust lag
    hook_z: float = -0.015          # m, string attachment relative to the centre of mass (--hook-z); a hook
                                    # below GUARD_FEET_Z (the guard feet, -15.4 mm) needs longer landing legs.
                                    # -0.015 (0.4 mm above the feet plane) clears the rotor discs by ~35 mm in
                                    # flight; the earlier -0.010 gave ~30 mm (--yaw-deg 0 --hook-z -0.010)
    # --- net ---
    L: float = 1.0                  # m, side of the (flat) net
    n: int = 9                      # knots per side
    m_net: float = 0.020            # kg
    string: float = 0.20            # m, corner string length
    # per-corner string lengths (m, drones 0-3) overriding `string` (robustness tests of a length error). A plain class
    # attribute, not a dataclass field: None (default) leaves the model and every saved parameter set unchanged.
    string_each = None
    node_drag: float = 2.0e-4       # N s/m per knot, stands in for air drag
    # quadratic air drag on the net threads (Sim, net_drag_coefficients). Plain class attributes, not dataclass
    # fields, so the parameter sets saved by the existing modes are unchanged; off in hold / sweep (their
    # reference numbers), on by default in travel mode (--net-drag), which records them in its JSON.
    # Per knot: F = -0.5 rho Cd d L_k |v| v, L_k = half of every link ending at the knot, v through still air.
    # Isotropic: the thread's orientation to the flow is ignored (every thread treated as in cross-flow),
    # which overestimates the drag of threads running along the flow.
    net_drag = False
    rho_air = 1.2                   # kg/m^3
    cd_thread = 1.2                 # cylinder in cross-flow
    rho_thread = 1150.0             # kg/m^3, nylon: sets the default thread diameter from m_net (thread_diameter)
    thread_d = None                 # m, thread diameter; None: derived from m_net, rho_thread and the thread length
    net_drag_keep_linear = False    # with net_drag on: False replaces the linear node_drag stand-in, True keeps both
    # payload fixed to the centre knot (--payload-mass, g): a small sphere that rests on the floor before takeoff,
    # collides with the floor, has its own quadratic air drag (always on, independent of net_drag) and adds to
    # the hanging weight in the feedforward and the static prediction. 0: no payload, the model is unchanged.
    payload_mass = 0.0              # kg
    payload_d = 0.015               # m, sphere diameter
    cd_payload = 0.47               # sphere drag coefficient
    # catch mode (--mode catch): free balls (table-tennis balls) in the model. 0: none, the model is unchanged.
    n_balls = 0
    ball_d = 0.040                  # m
    ball_m = 0.0027                 # kg
    ball_cd = 0.47                  # sphere drag coefficient (subcritical Re ~ 1e4-5e4)
    # --- flight plan ---
    z_hover: float = 1.2
    d_takeoff: float = 1.20         # drone spacing at rest and during the climb (--takeoff-spacing): the drones
                                    # start outside the net corners (+-0.49 m) with the strings slack. 0.80 is the
                                    # old plan, where each drone sat on the net (knots inside the rotor footprint)
    d_land: float = 1.20            # drone spacing during descent: net stays spread, drones land outside it
    z_touch: float = 0.015          # reference height at which the motors are cut (rest height is 0.010)
    yaw_deg: float = 45.0           # deg, heading of all four drones (--yaw-deg): initial attitude and the attitude
                                    # loop's heading reference. 0 (earlier default): rotors on the world diagonals,
                                    # so each corner string runs under the drone's inward rotor and its guard foot;
                                    # 45: rotors on the world axes and every string runs along a body axis,
                                    # between two rotors (guard clearance ~35 mm instead of ~18 mm)
    # --- rotor-to-net clearance check (analysis only, not used by the dynamics) ---
    prop_radius: float = 27.5e-3    # m, rotor disc radius
    prop_z: float = 0.012           # m, rotor disc plane above the centre of mass: ASSUMED, not published
    clearance_warn: float = 0.030   # m, warn for any flight phase whose minimum clearance is below this
    guard_warn: float = 0.010       # m, same for the propeller-guard clearance (contact, not a blade strike)
    # --- controller ---
    kp: tuple = (12.0, 12.0, 18.75)   # closed-loop poles at -2 rad/s (xy), -2.5 rad/s (z)
    kd: tuple = (6.0, 6.0, 7.5)
    ki: tuple = (8.0, 8.0, 15.6)
    i_clamp: float = 10.0           # m/s^2, limit on the integral term
    max_tilt_deg: float = 35.0
    att_wn: float = 18.0
    att_zeta: float = 1.0
    # attitude integrator gain kI = J * att_ki_factor * zeta * wn^3. A plain class attribute
    # (not a dataclass field) so the simple model's saved parameter set is unchanged;
    # OfficialParams overrides it as a field.
    att_ki_factor = 0.5
    # --- measurement noise (position system) ---
    sigma_p: float = 0.001          # m
    sigma_v: float = 0.005          # m/s
    # --- simulation ---
    dt: float = 0.001
    seed: int = 0


@dataclass
class OfficialParams(Params):
    """Crazyflie 2.1 Brushless, official numbers (no value here is tuned to this script).

    Firmware: bitcraze/crazyflie-firmware master, src/platform/interface/platform_defaults_cf21bl.h
      ARM_LENGTH, CF_MASS (2.1 Brushless with propeller guards, 350 mAh battery and
      Lighthouse deck), THRUST_MIN / THRUST_MAX (battery-compensated range per motor,
      i.e. 20.4 gf guaranteed, not the 30 gf peak of the product page), THRUST2TORQUE.
    Crazyflow (learnsyslab/crazyflow, high-fidelity simulator named by Bitcraze's blog,
      20 Oct 2025): system-identified set "cf21B_500" from learnsyslab/drone-models,
      drone_models/data/params.toml, equations from drone_models/first_principles/model.py.
      Their mass (0.04338 kg) is for their own configuration; the firmware's CF_MASS is used.
    Checked against the upstream files on 2026-11-21: all values below match.
    """
    drone_model: str = "official"
    m_drone: float = 0.0393                      # kg, firmware CF_MASS (override: --drone-mass)
    f_max: float = 0.2                           # N per motor, firmware THRUST_MAX
    f_min: float = 0.02136263065537499           # N per motor, firmware THRUST_MIN (clip while armed)
    arm: float = 0.050                           # m, firmware ARM_LENGTH, centre to motor
    inertia: tuple = (25e-6, 28e-6, 49e-6)       # kg m^2, Crazyflow J
    kappa: float = 0.00593893393599368           # m, Crazyflow thrust2torque (used by the mixer)
    kappa_firmware: float = 0.00569278844371417  # m, firmware THRUST2TORQUE (reference only)
    motor_tau: float = None                      # not used: rotor speed ODE below
    rotor_xy: float = 0.035355                   # m, Crazyflow L = rotor x/y offset (0.050 / sqrt 2)
    rpm2thrust: tuple = (0.0, -3.133427287299859e-7, 4.407354891648379e-10)    # F = c0 + c1 w + c2 w^2 [N], w in rpm
    rpm2torque: tuple = (0.0, 1.65886356219615e-9, 2.4693477924534137e-12)     # Q, same form [N m]
    rotor_dyn_coef: tuple = (13.996001897562685, 0.00011093207920685363,       # k0, k1 (spin-up)
                             5.933168530682111, 0.00031951312393561264)        # k2, k3 (spin-down)
    drag: tuple = (-0.02149163, -0.02149163, -0.02359736)                      # N s/m, body-frame drag_matrix diagonal
    prop_inertia: float = 38.93e-9               # kg m^2
    prop_radius: float = 27.5e-3                 # m (upstream marks it "TODO check"; clearance check only)
    # --- controller, retuned for the ~60 ms rotor response (simple model: zeta 1.0, ki factor 0.5) ---
    # att_wn stays 18 (tuned with the hook 1 cm below the centre of mass, yaw 0; unchanged for the
    # 1.5 cm / 45 deg default): the string pulls below the centre of mass (~1.4e-3 N m), and a
    # softer attitude loop (wn 6-14) lets the coupled drone-net system oscillate and flip.
    # More rate damping and a smaller attitude integrator cut the isolated 10 deg step
    # overshoot from 88 % to 30 % (simple model: 48 %); position gains are unchanged.
    att_zeta: float = 1.2
    att_ki_factor: float = 0.2


CORNERS = [(1, 1), (-1, 1), (-1, -1), (1, -1)]      # drone k sits at (sx, sy) * D/2
SPIN = [1, -1, 1, -1]                               # rotor yaw-torque signs (simple model)
# official model (Crazyflow): motor0 (+x,-y), motor1 (-x,-y), motor2 (-x,+y), motor3 (+x,+y)
ROTORS_OFFICIAL = [(1, -1), (-1, -1), (-1, 1), (1, 1)]
MIX = np.array([[-1.0, -1.0, 1.0, 1.0],      # roll  torque  = L * MIX[0] @ F
                [-1.0, 1.0, 1.0, -1.0],      # pitch torque  = L * MIX[1] @ F
                [-1.0, 1.0, -1.0, 1.0]])     # yaw   torque  =     MIX[2] @ Q


def _cross3(a, b):
    """np.cross of two 3-vectors with the same floating-point operations (a1 b2 - a2 b1, ...), without its overhead."""
    a0, a1, a2 = a.tolist()
    b0, b1, b2 = b.tolist()
    return np.array([a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0])


def _norm3(v):
    """np.linalg.norm of a 1-D float vector: sqrt(v . v), the same operations, without its overhead."""
    return math.sqrt(v.dot(v))


def hanging_mass(p: Params) -> float:
    """Net plus payload: the weight the four strings carry."""
    return p.m_net + getattr(p, "payload_mass", 0.0)


def payload_info(p: Params):
    """The payload for the JSON summaries (None without one: the summaries stay as they were)."""
    m = getattr(p, "payload_mass", 0.0)
    if not m:
        return None
    return {"mass_kg": m, "diameter_m": p.payload_d, "cd": p.cd_payload,
            "drag_coef_N_s2_m2": 0.5 * p.rho_air * p.cd_payload * np.pi * (p.payload_d / 2) ** 2,
            "hanging_mass_kg": hanging_mass(p)}


def apply_payload(p: Params, info):
    """Set the payload of p from a payload_info dict (summaries read back by render_scene / load_ff)."""
    if info:
        p.payload_mass, p.payload_d, p.cd_payload = info["mass_kg"], info["diameter_m"], info["cd"]
    return p


def is_official(p: Params) -> bool:
    return isinstance(p, OfficialParams)


class Rotors:
    """Rotor speeds of all four drones, Crazyflow first_principles model (cf21B_500).

    State w [rpm], shape (drone, motor). Per motor, with cmd and w in rpm:
        dw/dt = k0 (cmd - w) + k1 (cmd^2 - w^2)   if cmd > w   (spin-up)
                k2 (cmd - w) + k3 (cmd^2 - w^2)   otherwise    (spin-down)
        F = rpm2thrust(w), Q = rpm2torque(w)
    Body wrench (body frame):
        force  = [0, 0, sum F] (rotated to world) + R D R^T v   (linear body drag)
        torque = (MIX @ F) * [L, L, 0] + (MIX @ Q) * [0, 0, 1]
               + prop_inertia * [-omega_y S, -omega_x S, Sdot],  S = sum(MIX[2] * w), w in rad/s
    The gyroscopic term is copied as Crazyflow writes it (signs included), not re-derived.
    """

    def __init__(self, p: OfficialParams):
        self.p = p
        self.w = np.zeros((4, 4))
        self.cmd = np.zeros((4, 4))
        self.w_dot = np.zeros((4, 4))
        self._rxy = np.array([p.rotor_xy, p.rotor_xy, 0.0])     # constant arrays of wrench(), built once
        self._ez = np.array([0.0, 0.0, 1.0])
        self._drag = np.array(p.drag)

    def rpm_cmd(self, f):
        """rpm command that gives thrust f in steady state (inverse of rpm2thrust, upper root)."""
        c0, c1, c2 = self.p.rpm2thrust
        return (-c1 + np.sqrt(c1 * c1 + 4 * c2 * np.maximum(np.asarray(f) - c0, 0.0))) / (2 * c2)

    def thrust(self, w=None):
        c0, c1, c2 = self.p.rpm2thrust
        w = self.w if w is None else w
        return c0 + c1 * w + c2 * w * w

    def torque(self, w=None):
        c0, c1, c2 = self.p.rpm2torque
        w = self.w if w is None else w
        return c0 + c1 * w + c2 * w * w

    def derivative(self):
        k0, k1, k2, k3 = self.p.rotor_dyn_coef
        c, w = self.cmd, self.w
        self.w_dot = np.where(c > w, k0 * (c - w) + k1 * (c * c - w * w), k2 * (c - w) + k3 * (c * c - w * w))
        return self.w_dot

    def wrench(self, k, R, v, omega):
        """World-frame force and torque on drone k (R body to world, v world velocity, omega body rate)."""
        p = self.p
        F, Q = self.thrust(self.w[k]), self.torque(self.w[k])
        to_rads = 2 * np.pi / 60
        S = MIX[2] @ (self.w[k] * to_rads)
        Sdot = MIX[2] @ (self.w_dot[k] * to_rads)
        tau = ((MIX @ F) * self._rxy + (MIX @ Q) * self._ez
               + p.prop_inertia * np.array([-omega[1] * S, -omega[0] * S, Sdot]))
        force = R[:, 2] * F.sum() + R @ (self._drag * (R.T @ v))
        return force, R @ tau

    def step(self, dt):
        """Explicit Euler, using the derivative of the current step (also used for Sdot)."""
        self.w = np.maximum(self.w + dt * self.w_dot, 0.0)


def build_xml(p: Params) -> str:
    n, h = p.n, p.L / (p.n - 1)
    link_mass = p.m_net / (2 * n * (n - 1))
    a = p.arm / np.sqrt(2)
    out = []
    w = out.append
    w('<mujoco model="cf_net">')
    w(f'<option timestep="{p.dt}" integrator="implicitfast" gravity="0 0 {-G}"/>')
    w('<size memory="64M"/>')
    w('<worldbody>')
    w('<geom name="floor" type="plane" size="5 5 0.1" contype="1" conaffinity="0"/>')
    for k, (sx, sy) in enumerate(CORNERS):
        yaw = np.radians(p.yaw_deg) / 2
        quat = f' quat="{np.cos(yaw)} 0 0 {np.sin(yaw)}"' if p.yaw_deg else ""
        w(f'<body name="cf{k}" pos="{sx * p.d_takeoff / 2} {sy * p.d_takeoff / 2} 0.012"{quat}>')
        w('<freejoint/>')
        w(f'<inertial pos="0 0 0" mass="{p.m_drone}" diaginertia="{p.inertia[0]} {p.inertia[1]} {p.inertia[2]}"/>')
        w('<geom type="box" size="0.035 0.035 0.01" contype="0" conaffinity="1"/>')
        if getattr(p, "n_balls", 0):      # catch mode: the airframe with its guards as a cylinder, for the balls only
            w(f'<geom name="shell{k}" type="cylinder" size="{DRONE_R} {DRONE_HALF_H}" contype="0" conaffinity="2" '
              f'group="3" rgba="0.5 0.5 0.5 0.2"/>')
        w(f'<site name="hook{k}" pos="0 0 {p.hook_z}"/>')
        if is_official(p):
            for r, (rx, ry) in enumerate(ROTORS_OFFICIAL):
                w(f'<site name="rot{k}_{r}" pos="{rx * p.rotor_xy} {ry * p.rotor_xy} 0"/>')
        else:
            for r, (rx, ry) in enumerate(CORNERS):
                w(f'<site name="rot{k}_{r}" pos="{rx * a} {ry * a} 0"/>')
        w('</body>')
    shrink = 0.98       # start with every string just slack
    pm = getattr(p, "payload_mass", 0.0)
    for i in range(n):
        for j in range(n):
            deg = (i > 0) + (i < n - 1) + (j > 0) + (j < n - 1)
            x = (i * h - p.L / 2) * shrink
            y = (j * h - p.L / 2) * shrink
            centre = pm and i == n // 2 and j == n // 2
            z0 = max(0.004, p.payload_d / 2) if centre else 0.004
            w(f'<body name="n{i}_{j}" pos="{x} {y} {z0}">')
            for ax in ("1 0 0", "0 1 0", "0 0 1"):
                w(f'<joint type="slide" axis="{ax}" damping="{p.node_drag}"/>')
            w(f'<geom type="sphere" size="0.004" mass="{deg / 2 * link_mass}" contype="0" conaffinity="1"/>')
            if centre:          # payload: a sphere on the centre knot (floor contact like the knots)
                w(f'<geom name="payload" type="sphere" size="{p.payload_d / 2}" mass="{pm}" contype="0" '
                  f'conaffinity="1" rgba="0.85 0.10 0.10 1"/>')
            w(f'<site name="s{i}_{j}"/>')
            w('</body>')
    for b in range(getattr(p, "n_balls", 0)):   # catch mode: balls (parked far away until launched, see CatchPlan)
        # contype 2 / conaffinity 3: floor, other balls and the drone shells; never the knots (the net contact is
        # the penalty model of CatchPlan.pre_step)
        w(f'<body name="ball{b}" pos="{CATCH_PARK[0] - 0.2 * b} {CATCH_PARK[1]} {CATCH_PARK[2]}"><freejoint/>'
          f'<geom name="ball{b}" type="sphere" size="{p.ball_d / 2}" mass="{p.ball_m}" contype="2" conaffinity="3" '
          f'rgba="1 0.55 0.1 1"/></body>')
    w('</worldbody>')
    lim = 'limited="true" solreflimit="0.01 1" solimplimit="0.95 0.99 0.001"'
    w('<tendon>')
    for i in range(n):
        for j in range(n):
            if i < n - 1:
                w(f'<spatial name="a{i}_{j}" {lim} range="0 {h}"><site site="s{i}_{j}"/><site site="s{i + 1}_{j}"/></spatial>')
            if j < n - 1:
                w(f'<spatial name="b{i}_{j}" {lim} range="0 {h}"><site site="s{i}_{j}"/><site site="s{i}_{j + 1}"/></spatial>')
    for k, (sx, sy) in enumerate(CORNERS):
        i = n - 1 if sx > 0 else 0
        j = n - 1 if sy > 0 else 0
        Ls = p.string if p.string_each is None else float(p.string_each[k])
        w(f'<spatial name="str{k}" {lim} range="0 {Ls}"><site site="hook{k}"/><site site="s{i}_{j}"/></spatial>')
    w('</tendon>')
    if not is_official(p):      # the official rotors are integrated by Rotors, not by MuJoCo actuators
        w('<actuator>')
        for k in range(4):
            for r in range(4):
                w(f'<general name="m{k}_{r}" site="rot{k}_{r}" gear="0 0 1 0 0 {SPIN[r] * p.kappa}" '
                  f'ctrllimited="true" ctrlrange="0 {p.f_max}" dyntype="filter" dynprm="{p.motor_tau}" '
                  f'gaintype="fixed" gainprm="1" biastype="none"/>')
        w('</actuator>')
    w('<sensor>')
    for k in range(4):
        w(f'<tendonlimitfrc name="tension{k}" tendon="str{k}"/>')
    w('</sensor>')
    w('</mujoco>')
    return "\n".join(out)


# ---------------------------------------------------------------- flight plan
def minjerk(t, t0, t1, v0, v1):
    """Value and rate of a minimum-jerk ramp from v0 (at t0) to v1 (at t1)."""
    if t <= t0:
        return v0, 0.0
    if t >= t1:
        return v1, 0.0
    T = t1 - t0
    s = (t - t0) / T
    return (v0 + (v1 - v0) * (10 * s**3 - 15 * s**4 + 6 * s**5),
            (v1 - v0) * (30 * s**2 - 60 * s**3 + 30 * s**4) / T)


class Plan:
    """Piecewise min-jerk schedules for hover height z(t) and drone spacing D(t)."""

    def __init__(self, z_ramps, d_ramps, t_end, t_arm=0.5, t_off=np.inf):
        self.z_ramps, self.d_ramps, self.t_end, self.t_arm, self.t_off = z_ramps, d_ramps, t_end, t_arm, t_off

    @staticmethod
    def _eval(ramps, t):
        val, rate = ramps[0][2], 0.0
        for (t0, t1, v0, v1) in ramps:
            if t >= t0:
                val, rate = minjerk(t, t0, t1, v0, v1)
        return val, rate

    def ref(self, t):
        z, dz = self._eval(self.z_ramps, t)
        d, dd = self._eval(self.d_ramps, t)
        pos = np.array([[sx * d / 2, sy * d / 2, z] for sx, sy in CORNERS])
        vel = np.array([[sx * dd / 2, sy * dd / 2, dz] for sx, sy in CORNERS])
        return pos, vel, d

    def acc(self, t):
        """Reference acceleration for the acceleration feedforward (None: no feedforward term)."""
        return None


def minjerk_travel(t, t0, T, v):
    """Distance, speed and acceleration of a min-jerk speed ramp 0 -> v over [t0, t0 + T]."""
    s = np.clip((t - t0) / T, 0.0, 1.0)
    return (v * T * (2.5 * s**4 - 3 * s**5 + s**6), v * (10 * s**3 - 15 * s**4 + 6 * s**5),
            v / T * (30 * s**2 - 60 * s**3 + 30 * s**4) if 0.0 < s < 1.0 else 0.0)


class TravelPlan(Plan):
    """A hover plan (climb, spread) followed by a straight rigid-pattern trip of the whole formation.

    From t_go the formation's reference moves along the horizontal unit vector u (heading_deg from +x):
    min-jerk speed ramp 0 -> v_cruise over T_a = 1.875 v / a_max (its peak acceleration is a_max),
    cruise for t_cruise, the mirrored ramp down to rest, then hold. Distance to reach v and stop again
    (without the cruise) = v T_a = 1.875 v^2 / a_max (a trapezoid at a_max would need v^2 / a_max)."""

    def __init__(self, z_ramps, d_ramps, t_go, v_cruise, a_max, heading_deg, t_cruise, t_settle, t_arm=0.5):
        self.t_go, self.v, self.a_max, self.heading_deg = t_go, v_cruise, a_max, heading_deg
        self.T_a = 1.875 * v_cruise / a_max
        self.t_cruise, self.t_settle = t_cruise, t_settle
        h = np.radians(heading_deg)
        self.u = np.array([np.cos(h), np.sin(h)])
        self.t_a1 = t_go + self.T_a
        self.t_d0 = self.t_a1 + t_cruise
        self.t_d1 = self.t_d0 + self.T_a
        super().__init__(z_ramps, d_ramps, t_end=self.t_d1 + t_settle, t_arm=t_arm)

    def travel(self, t):
        """(distance along u, speed, acceleration) of the formation reference."""
        v, T = self.v, self.T_a
        if t < self.t_a1:
            return minjerk_travel(t, self.t_go, T, v)
        s_a = 0.5 * v * T
        if t < self.t_d0:
            return s_a + v * (t - self.t_a1), v, 0.0
        s, sd, sdd = minjerk_travel(t, self.t_d0, T, v)
        tt = min(t - self.t_d0, T)
        return s_a + v * self.t_cruise + v * tt - s, v - sd, -sdd

    def phases(self):
        return {"accelerate": (self.t_go, self.t_a1), "cruise": (self.t_a1, self.t_d0),
                "decelerate": (self.t_d0, self.t_d1), "settle": (self.t_d1, self.t_end)}

    def ref(self, t):
        pos, vel, d = super().ref(t)
        s, sd, _ = self.travel(t)
        pos[:, :2] += s * self.u
        vel[:, :2] += sd * self.u
        return pos, vel, d

    def acc(self, t):
        a = np.zeros((4, 3))
        a[:, :2] = self.travel(t)[2] * self.u
        return a


# ---------------------------------------------------------------- controller
def vee(M):
    return np.array([M[2, 1], M[0, 2], M[1, 0]])


_G_VEC = np.array([0, 0, G])


class DroneController:
    def __init__(self, p: Params):
        self.p = p
        self.kp, self.kd, self.ki = map(np.array, (p.kp, p.kd, p.ki))
        self.J = np.array(p.inertia)
        self.kR = self.J * p.att_wn**2
        self.kW = self.J * 2 * p.att_zeta * p.att_wn
        # attitude integrator (the firmware has one too): removes the steady tilt error
        # caused by the string pulling below the centre of mass
        self.kI = self.J * p.att_ki_factor * p.att_zeta * p.att_wn**3
        self.att_integ = np.zeros(3)
        if is_official(p):
            alloc = np.vstack([np.ones(4), p.rotor_xy * MIX[0], p.rotor_xy * MIX[1], p.kappa * MIX[2]])
            self.f_lo = p.f_min
        else:
            a = p.arm / np.sqrt(2)
            alloc = np.array([[1.0] * 4,
                              [ry * a for _, ry in CORNERS],
                              [-rx * a for rx, _ in CORNERS],
                              [s * p.kappa for s in SPIN]])
            self.f_lo = 0.0
        self.alloc_inv = np.linalg.inv(alloc)
        self.integ = np.zeros(3)
        self.F_des = np.array([0.0, 0.0, p.m_drone * G])
        self.tan_max = np.tan(np.radians(p.max_tilt_deg))
        yaw = np.radians(p.yaw_deg)
        self.heading = np.array([np.cos(yaw), np.sin(yaw), 0.0])   # desired body x axis, projected
        self.f_raw = np.zeros(4)

    def position_step(self, pos, vel, pos_ref, vel_ref, dt, f_ff=0.0, acc_ref=None):
        """acc_ref: optional reference acceleration, added as the feedforward m_drone * acc_ref (travel mode)."""
        e, ed = pos_ref - pos, vel_ref - vel
        self.integ = np.clip(self.integ + self.ki * e * dt, -self.p.i_clamp, self.p.i_clamp)
        acc = self.kp * e + self.kd * ed + self.integ + _G_VEC
        if acc_ref is not None:
            acc = acc + acc_ref
        F = self.p.m_drone * acc + f_ff
        F[2] = max(F[2], 0.1 * self.p.m_drone * G)
        fh = np.hypot(F[0], F[1])
        if fh > F[2] * self.tan_max:
            F[:2] *= F[2] * self.tan_max / fh
        fmax = 0.95 * 4 * self.p.f_max
        nF = _norm3(F)
        if nF > fmax:
            F *= fmax / nF
        self.F_des = F

    def attitude_step(self, R, omega, dt):
        F = self.F_des
        b3 = F / _norm3(F)
        b2 = _cross3(b3, self.heading)
        b2 /= _norm3(b2)
        Rd = np.column_stack([_cross3(b2, b3), b2, b3])
        eR = 0.5 * vee(Rd.T @ R - R.T @ Rd)
        self.att_integ = np.clip(self.att_integ + eR * dt, -0.5, 0.5)
        tau = -self.kR * eR - self.kW * omega - self.kI * self.att_integ + _cross3(omega, self.J * omega)
        thrust = F @ R[:, 2]
        f = self.alloc_inv @ np.array([thrust, *tau])
        self.f_raw = f                      # before the clip: > f_max means the motor command saturates
        return np.clip(f, self.f_lo, self.p.f_max)


# ---------------------------------------------------------------- net aerodynamics
def thread_length_per_knot(p: Params):
    """(n*n,) thread length attributed to each knot (flattened i * n + j): half of every link ending
    there (corner 1 h, edge 1.5 h, interior 2 h). Sums to the net's total thread length 2 n (n - 1) h."""
    n, h = p.n, p.L / (p.n - 1)
    i = np.arange(n)
    deg = (i > 0).astype(int) + (i < n - 1)
    return ((deg[:, None] + deg[None, :]) * h / 2).ravel()


def thread_diameter(p: Params):
    """p.thread_d, or the diameter at which the net's threads (total length 2 n (n - 1) h, a solid
    round nylon thread of density rho_thread) weigh m_net: d = sqrt(4 m_net / (pi rho L_total))."""
    if p.thread_d is not None:
        return float(p.thread_d)
    L_tot = 2 * p.n * (p.n - 1) * p.L / (p.n - 1)
    return float(np.sqrt(4 * p.m_net / (np.pi * p.rho_thread * L_tot)))


def net_drag_coefficients(p: Params):
    """(n*n,) c_k in F_k = -c_k |v_k| v_k [N s^2/m^2]: c_k = 0.5 rho Cd d L_k (isotropic, see Params)."""
    return 0.5 * p.rho_air * p.cd_thread * thread_diameter(p) * thread_length_per_knot(p)


def aero_info(p: Params):
    """The net-drag settings of a run, for its JSON."""
    L = thread_length_per_knot(p)
    c = 0.5 * p.rho_air * p.cd_thread * thread_diameter(p) * L.sum()
    return {"net_drag": bool(p.net_drag), "rho_air": p.rho_air, "cd_thread": p.cd_thread,
            "rho_thread": p.rho_thread, "thread_d_m": thread_diameter(p),
            "thread_d_derived": p.thread_d is None, "thread_length_total_m": float(L.sum()),
            "drag_area_cd_d_l_m2": float(p.cd_thread * thread_diameter(p) * L.sum()),
            "total_drag_at_1_m_s_N": float(c), "total_drag_at_2_m_s_N": float(4 * c),
            "linear_stand_in_N_s_m_per_knot": p.node_drag if (not p.net_drag or p.net_drag_keep_linear) else 0.0,
            "model": "per knot F = -0.5 rho Cd d L_k |v| v, still air, isotropic (thread orientation ignored: "
                     "overestimates the drag of threads along the flow); corner strings have no drag"}


# ---------------------------------------------------------------- simulation
class Sim:
    def __init__(self, p: Params):
        self.p = p
        self.model = mujoco.MjModel.from_xml_string(build_xml(p))
        self.data = mujoco.MjData(self.model)
        m = self.model
        nid = lambda t, name: mujoco.mj_name2id(m, t, name)
        self.qadr, self.vadr, self.hook = [], [], []
        for k in range(4):
            b = nid(mujoco.mjtObj.mjOBJ_BODY, f"cf{k}")
            j = m.body_jntadr[b]
            self.qadr.append(m.jnt_qposadr[j])
            self.vadr.append(m.jnt_dofadr[j])
            self.hook.append(nid(mujoco.mjtObj.mjOBJ_SITE, f"hook{k}"))
        self.body = [nid(mujoco.mjtObj.mjOBJ_BODY, f"cf{k}") for k in range(4)]
        n = p.n
        self.node_site = np.array([[nid(mujoco.mjtObj.mjOBJ_SITE, f"s{i}_{j}") for j in range(n)] for i in range(n)])
        self.corner_site = [self.node_site[n - 1 if sx > 0 else 0, n - 1 if sy > 0 else 0] for sx, sy in CORNERS]
        self.ctrl = [DroneController(p) for _ in range(4)]
        self.rng = np.random.default_rng(p.seed)
        self.rotors = Rotors(p) if is_official(p) else None
        # knots: three world-axis slide joints each, so a world force is a generalized force on them
        kb = [nid(mujoco.mjtObj.mjOBJ_BODY, f"n{i}_{j}") for i in range(n) for j in range(n)]
        self.knot_dof = np.array([[m.jnt_dofadr[m.body_jntadr[b] + a] for a in range(3)] for b in kb])
        self.knot_q = np.array([[m.jnt_qposadr[m.body_jntadr[b] + a] for a in range(3)] for b in kb])
        self.knot_pos0 = m.body_pos[kb].copy()
        self.knot_mass = m.body_mass[kb].copy()
        self.corner_knot = [(n - 1 if sx > 0 else 0) * n + (n - 1 if sy > 0 else 0) for sx, sy in CORNERS]
        self.knot_cq = None
        c_ = (n // 2) * n + n // 2
        self.c_idx, self.c_dof = c_, self.knot_dof[c_]
        pi_ = payload_info(p)
        self.payload_cq = pi_["drag_coef_N_s2_m2"] if pi_ else 0.0
        if p.net_drag:
            self.knot_cq = net_drag_coefficients(p)
            if not p.net_drag_keep_linear:
                m.dof_damping[self.knot_dof.ravel()] = 0.0      # the quadratic drag replaces the stand-in

    def knots_now(self):
        """Knot positions (n*n, 3) from qpos, i.e. at the same instant as drone_state."""
        return self.knot_pos0 + self.data.qpos[self.knot_q]

    def set_thrust(self, k, f):
        """Desired rotor thrusts of drone k (None: motors off)."""
        if self.rotors is None:
            self.data.ctrl[4 * k:4 * k + 4] = 0.0 if f is None else f
        else:
            self.rotors.cmd[k] = 0.0 if f is None else self.rotors.rpm_cmd(f)

    def motor_thrust(self):
        """Actual rotor thrusts, shape (drone, motor)."""
        if self.rotors is None:
            return self.data.actuator_force.reshape(4, 4).copy()
        return self.rotors.thrust().copy()

    def drone_state(self, k):
        d = self.data
        q, v = self.qadr[k], self.vadr[k]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[q + 3:q + 7])
        return d.qpos[q:q + 3].copy(), d.qvel[v:v + 3].copy(), R.reshape(3, 3), d.qvel[v + 3:v + 6].copy()

    def run(self, plan: Plan, gusts=(), noise=True, log_nodes_every=0.05, ff=None, acc_ff=0.0,
            extra_log=False, verbose=True, abort_err=None, snapshot=None, resume=None):
        """snapshot: optional (step, save_fn): at the start of that step the complete integration state (MjData,
        controllers, rotors, noise generator, the plan's own state, the log so far) is pickled and passed to save_fn;
        resume: such a pickled state, the run continues from its step (bit-identical to running from the start, provided
        everything before that step is the same: see catch_snapshot_key).
        ff: optional (pull, lift) feedforward functions of the drone spacing:
        pull = horizontal net force per drone [N], lift = drone height at which the whole
        net is off the floor [m].
        acc_ff: reference-acceleration feedforward, force = acc_ff * m_drone * plan.acc(t) (0: off;
        1: drone mass; used by travel mode). extra_log: also log the quantities the travel analysis
        needs (and take the net frames from qpos, the same instant as the drone pose). abort_err [m]:
        stop early once a drone is this far from its reference (log["aborted_at"])."""
        p, m, d = self.p, self.model, self.data
        steps = int(round(plan.t_end / p.dt))
        att_every, pos_every, log_every = 2, 10, 10          # 500 Hz, 100 Hz, 100 Hz
        node_every = int(round(log_nodes_every / p.dt))
        keys = ["t", "pos", "ref", "spacing_ref", "tilt", "tension", "theta",
                "motor", "sag_drone", "sag_net", "corner_span"]
        if extra_log:
            keys += ["vel", "ref_vel", "net_c", "net_com", "net_zmin", "ncon", "thrust_vec", "body_drag",
                     "str_dir", "net_qdrag", "net_ldrag", "motor_raw", "knot_vel_mean", "knot_zmax", "disc_zmin",
                     "knot_zmax_inner", "disc_czmin"]
        log = {k: [] for k in keys}
        frames_t, frames, frames_pos, frames_quat = [], [], [], []
        c = p.n // 2
        report_every = int(round(5.0 / p.dt))
        fd = np.zeros((p.n * p.n, 3))
        aborted = None
        pre_step = getattr(plan, "pre_step", None)      # catch mode: balls (launch, drag, net contact)
        ff_extra = getattr(plan, "ff_extra", None)      # catch mode: weight of the balls in the net
        step0 = 0
        if resume is not None:
            import pickle
            st = pickle.loads(resume)
            self.data = d = st["data"]
            self.ctrl = st["ctrl"]
            for cc in self.ctrl:
                cc.p = p
            if self.rotors is not None:
                self.rotors = st["rotors"]
                self.rotors.p = p
            self.rng.bit_generator.state = st["rng"]
            log, fd, aborted = st["log"], st["fd"], st["aborted"]
            frames_t, frames, frames_pos, frames_quat = st["frames"]
            plan.restore_state(st["plan"])
            step0 = st["step"]
        snap_step = snapshot[0] if snapshot is not None else -1
        for step in range(step0, steps):
            if step == snap_step:
                import pickle
                snapshot[1](pickle.dumps({"step": step, "data": d, "ctrl": self.ctrl, "rotors": self.rotors,
                                          "rng": self.rng.bit_generator.state, "log": log, "fd": fd,
                                          "aborted": aborted, "frames": (frames_t, frames, frames_pos, frames_quat),
                                          "plan": plan.snapshot_state()}, protocol=pickle.HIGHEST_PROTOCOL))
            t = step * p.dt
            if verbose and step % report_every == 0:
                print(f"  simulating: t = {t:4.0f} / {plan.t_end:.0f} s", flush=True)
            pos_ref, vel_ref, d_ref = plan.ref(t)
            acc_ref = plan.acc(t) if acc_ff else None
            armed = plan.t_arm <= t < plan.t_off
            if step % att_every == 0:
                for k in range(4):
                    pos, vel, R, om = self.drone_state(k)
                    if not armed:
                        self.set_thrust(k, None)
                        continue
                    if step % pos_every == 0:
                        if noise:
                            pos = pos + self.rng.normal(0, p.sigma_p, 3)
                            vel = vel + self.rng.normal(0, p.sigma_v, 3)
                        f_ff = 0.0
                        if ff is not None:
                            # scale with the share of the net that is off the floor
                            # (takeoff and landing both pass through this)
                            pull, lift = ff
                            gain = np.clip(pos_ref[k][2] / lift(d_ref), 0.0, 1.0)
                            sx, sy = CORNERS[k]
                            f_ff = gain * np.array([sx * pull(d_ref) / np.sqrt(2), sy * pull(d_ref) / np.sqrt(2),
                                                    hanging_mass(p) * G / 4])
                        if ff_extra is not None:
                            f_ff = f_ff + ff_extra(t, k)
                        if acc_ref is None:
                            self.ctrl[k].position_step(pos, vel, pos_ref[k], vel_ref[k], pos_every * p.dt, f_ff)
                        else:
                            self.ctrl[k].position_step(pos, vel, pos_ref[k], vel_ref[k], pos_every * p.dt, f_ff,
                                                       acc_ref=acc_ff * acc_ref[k])
                    self.set_thrust(k, self.ctrl[k].attitude_step(R, om, att_every * p.dt))
            d.xfrc_applied[:] = 0.0
            if self.rotors is not None:
                self.rotors.derivative()
                for k in range(4):
                    pos, vel, R, om = self.drone_state(k)
                    force, torque = self.rotors.wrench(k, R, vel, om)
                    d.xfrc_applied[self.body[k], :3] = force
                    d.xfrc_applied[self.body[k], 3:] = torque
            for (k, t0, t1, force) in gusts:
                if t0 <= t < t1:
                    d.xfrc_applied[self.body[k], :3] += force
            if self.knot_cq is not None:
                v = d.qvel[self.knot_dof]                         # (knot, 3): world velocity through still air
                fd = -(self.knot_cq * np.linalg.norm(v, axis=1))[:, None] * v
                d.qfrc_applied[self.knot_dof] = fd
            if self.payload_cq:                                   # payload sphere drag on the centre knot
                v = d.qvel[self.c_dof]
                d.qfrc_applied[self.c_dof] = ((fd[self.c_idx] if self.knot_cq is not None else 0.0)
                                              - self.payload_cq * np.linalg.norm(v) * v)
            if pre_step is not None:
                pre_step(t)
            mujoco.mj_step(m, d)
            if self.rotors is not None:
                self.rotors.step(p.dt)

            if extra_log and step % log_every == 0:
                kx = self.knots_now()
                st = [self.drone_state(k) for k in range(4)]
                P = np.array([s[0] for s in st])
                Rs = np.array([s[2] for s in st])
                hooks = P + p.hook_z * Rs[:, :, 2]
                sv = kx[self.corner_knot] - hooks
                log["vel"].append(np.array([s[1] for s in st]))
                log["ref_vel"].append(vel_ref.copy())
                log["net_c"].append(kx[c * p.n + c].copy())
                log["net_com"].append(self.knot_mass @ kx / self.knot_mass.sum())
                log["net_zmin"].append(kx[:, 2].min())
                log["ncon"].append(d.ncon)
                F = self.motor_thrust().sum(axis=1)
                log["thrust_vec"].append(Rs[:, :, 2] * F[:, None])
                if self.rotors is not None:
                    log["body_drag"].append(np.array([Rs[k] @ (np.array(p.drag) * (Rs[k].T @ st[k][1])) for k in range(4)]))
                else:
                    log["body_drag"].append(np.zeros((4, 3)))
                log["str_dir"].append(sv / np.linalg.norm(sv, axis=1, keepdims=True))
                log["net_qdrag"].append(fd.sum(axis=0) if self.knot_cq is not None else np.zeros(3))
                log["net_ldrag"].append(d.qfrc_damper[self.knot_dof].sum(axis=0))
                log["motor_raw"].append(np.array([cc.f_raw for cc in self.ctrl]))
                log["knot_vel_mean"].append(d.qvel[self.knot_dof].mean(axis=0))
                # highest knot and the lowest point of any rotor disc (disc tilted with its drone)
                Cd = P[:, None, :] + np.einsum("kij,rj->kri", Rs, rotor_layout(p))
                nz = np.clip(Rs[:, 2, 2], -1.0, 1.0)
                log["knot_zmax"].append(kx[:, 2].max())
                log["disc_zmin"].append(float((Cd[:, :, 2] - p.prop_radius * np.sqrt(1.0 - nz * nz)[:, None]).min()))
                log["disc_czmin"].append(float(Cd[:, :, 2].min()))                 # lowest rotor hub (disc centre)
                log["knot_zmax_inner"].append(float(np.delete(kx[:, 2], self.corner_knot).max()))
                if hasattr(plan, "observe"):            # mission plans decide departures from the live state
                    ck = kx[c * p.n + c]
                    plan.observe(t, P, ck, float(kx[self.corner_knot][:, 2].mean() - ck[2]))
                if abort_err is not None and np.linalg.norm(P - pos_ref, axis=1).max() > abort_err:
                    aborted = t

            if step % log_every == 0:
                P = np.array([self.drone_state(k)[0] for k in range(4)])
                tilt, theta = [], []
                for k in range(4):
                    R = self.drone_state(k)[2]
                    tilt.append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
                    v = d.site_xpos[self.corner_site[k]] - d.site_xpos[self.hook[k]]
                    theta.append(np.degrees(np.arctan2(-v[2], np.hypot(v[0], v[1]))))
                corners = np.array([d.site_xpos[s] for s in self.corner_site])
                zc = d.site_xpos[self.node_site[c, c]][2]
                log["t"].append(t)
                log["pos"].append(P)
                log["ref"].append(pos_ref)
                log["spacing_ref"].append(d_ref)
                log["tilt"].append(tilt)
                log["theta"].append(theta)
                log["tension"].append(np.abs(d.sensordata[:4]).copy())
                log["motor"].append(self.motor_thrust())
                log["sag_drone"].append(P[:, 2].mean() - zc)
                log["sag_net"].append(corners[:, 2].mean() - zc)
                log["corner_span"].append(np.mean([np.linalg.norm(corners[a] - corners[(a + 1) % 4]) for a in range(4)]))
            if node_every and step % node_every == 0:
                frames_t.append(t)
                frames.append(np.concatenate([d.site_xpos[self.node_site.ravel()],
                                              d.site_xpos[self.hook]]).copy())
                frames_pos.append(np.array([d.qpos[q:q + 3] for q in self.qadr]))          # drone CoM
                frames_quat.append(np.array([d.qpos[q + 3:q + 7] for q in self.qadr]))     # (w, x, y, z)
                if extra_log:       # knots and hooks from qpos: the same instant as frames_pos / frames_quat
                    Rq = quat_to_mat(frames_quat[-1])
                    frames[-1] = np.concatenate([self.knots_now(), frames_pos[-1] + p.hook_z * Rq[:, :, 2]])
            if aborted is not None or getattr(plan, "done", False):
                break
        log = {k: np.array(v) for k, v in log.items()}
        if extra_log:
            log["aborted_at"] = np.array(np.nan if aborted is None else aborted)
        log["frames_t"], log["frames"] = np.array(frames_t), np.array(frames)
        log["frames_pos"], log["frames_quat"] = np.array(frames_pos), np.array(frames_quat)
        return log


# ---------------------------------------------------------------- analysis
def predicted(p: Params, theta_deg):
    """Static force balance at one corner for a string angle theta below horizontal."""
    th = np.radians(theta_deg)
    w4 = hanging_mass(p) * G / 4
    tension = w4 / np.sin(th)
    f_h = w4 / np.tan(th)
    tilt = np.degrees(np.arctan(f_h / (p.m_drone * G + w4)))
    return tension, f_h, tilt


def window_stats(p: Params, log, t0, t1):
    s = (log["t"] >= t0) & (log["t"] < t1)
    err = log["pos"][s] - log["ref"][s]
    theta = float(log["theta"][s].mean())
    t_pred, fh_pred, tilt_pred = predicted(p, theta)
    return {
        "window_s": [t0, t1],
        "spacing_ref_m": float(log["spacing_ref"][s].mean()),
        "spacing_m": float(np.mean([np.linalg.norm(log["pos"][s][:, a, :2] - log["pos"][s][:, (a + 1) % 4, :2], axis=1).mean()
                                    for a in range(4)])),
        "pos_rms_mm": (1e3 * np.sqrt((err**2).mean(axis=(0, 1)))).round(3).tolist(),
        "pos_err_max_mm": float(1e3 * np.abs(err).max()),
        "pos_err_mean_mm": float(1e3 * np.linalg.norm(err.mean(axis=0), axis=1).max()),
        "sag_below_drones_m": float(log["sag_drone"][s].mean()),
        "sag_of_net_m": float(log["sag_net"][s].mean()),
        "sag_std_mm": float(1e3 * log["sag_net"][s].std()),
        "net_corner_span_m": float(log["corner_span"][s].mean()),
        "string_angle_deg": theta,
        "tilt_deg": float(log["tilt"][s].mean()),
        "tilt_pred_deg": float(tilt_pred),
        "tension_N": float(log["tension"][s].mean()),
        "tension_pred_N": float(t_pred),
        "pull_horizontal_gf": float(fh_pred / G * 1e3),
        "motor_frac_mean": float(log["motor"][s].mean() / p.f_max),
        "motor_frac_max": float(log["motor"][s].max() / p.f_max),
    }


# ---------------------------------------------------------------- plotting
def draw_net(ax, p, frame, proj=None):
    n = p.n
    nodes = frame[:n * n].reshape(n, n, 3)
    hooks = frame[n * n:]
    seg = lambda a, b, **kw: (ax.plot(*np.array([a, b]).T, **kw) if proj is None
                              else ax.plot(np.array([a, b])[:, proj[0]], np.array([a, b])[:, proj[1]], **kw))
    for i in range(n):
        for j in range(n):
            if i < n - 1:
                seg(nodes[i, j], nodes[i + 1, j], color="tab:blue", lw=0.8)
            if j < n - 1:
                seg(nodes[i, j], nodes[i, j + 1], color="tab:blue", lw=0.8)
    for k, (sx, sy) in enumerate(CORNERS):
        corner = nodes[n - 1 if sx > 0 else 0, n - 1 if sy > 0 else 0]
        seg(hooks[k], corner, color="k", lw=1.0)
    if proj is None:
        ax.scatter(*hooks.T, color="tab:red", s=40, depthshade=False)
    else:
        ax.scatter(hooks[:, proj[0]], hooks[:, proj[1]], color="tab:red", s=40, zorder=3)


def plot_snapshot(p, log, t_snap, path, title):
    i = int(np.argmin(np.abs(log["frames_t"] - t_snap)))
    frame = log["frames"][i]
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    draw_net(ax, p, frame)
    zc = frame[:, 2]
    ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7); ax.set_zlim(zc.max() - 0.9, zc.max() + 0.1)
    ax.set_box_aspect((1.4, 1.4, 1.0))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.view_init(elev=22, azim=-60)
    ax.set_title(f"t = {log['frames_t'][i]:.1f} s")
    ax2 = fig.add_subplot(1, 2, 2)
    draw_net(ax2, p, frame, proj=(0, 2))
    ax2.set_aspect("equal"); ax2.grid(alpha=0.3)
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("z [m]"); ax2.set_title("side view (projection on x-z)")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_timeseries(p, log, path, marks=(), clearance=None):
    """clearance: optional per-frame series from clearance_analysis (adds a fourth row)."""
    t = log["t"]
    rows = 3 if clearance is None else 4
    fig, axs = plt.subplots(rows, 2, figsize=(13, 3 * rows), sharex=True)
    ax = axs[0, 0]
    ax.plot(t, log["pos"][:, :, 2]); ax.plot(t, log["ref"][:, 0, 2], "k--", lw=1, label="reference")
    ax.set_ylabel("drone height [m]"); ax.legend()
    ax = axs[0, 1]
    span = [np.linalg.norm(log["pos"][:, a, :2] - log["pos"][:, (a + 1) % 4, :2], axis=1) for a in range(4)]
    ax.plot(t, np.array(span).T); ax.plot(t, log["spacing_ref"], "k--", lw=1, label="reference")
    ax.plot(t, log["corner_span"], color="tab:gray", lw=1, label="net corner span")
    ax.set_ylabel("spacing [m]"); ax.legend()
    ax = axs[1, 0]
    ax.plot(t, 1e3 * np.linalg.norm(log["pos"] - log["ref"], axis=2))
    ax.set_ylabel("position error [mm]"); ax.set_ylim(0, 60)
    ax = axs[1, 1]
    ax.plot(t, log["sag_net"], label="net centre below its corners")
    ax.plot(t, log["sag_drone"], label="net centre below drones")
    ax.set_ylabel("sag [m]"); ax.legend()
    ax = axs[2, 0]
    ax.plot(t, log["tilt"]); ax.set_ylabel("drone tilt [deg]")
    ax = axs[2, 1]
    ax.plot(t, log["tension"] / G * 1e3)
    ax.set_ylabel("string tension [gf]")
    ax3 = ax.twinx()
    ax3.plot(t, log["motor"].max(axis=(1, 2)) / p.f_max, color="tab:gray", lw=0.8)
    ax3.set_ylabel("max motor thrust / limit", color="tab:gray"); ax3.set_ylim(0, 1.05)
    if clearance is not None:
        ft = clearance["t"]
        ax = axs[3, 0]
        for k in range(4):
            ax.plot(ft, 1e3 * clearance["all"][:, k], color=f"C{k}", lw=1, label=f"drone {k}")
            ax.plot(ft, 1e3 * clearance["net"][:, k], color=f"C{k}", lw=0.7, ls="--")
            if "guard" in clearance and np.isfinite(clearance["guard"]).any():
                ax.plot(ft, 1e3 * clearance["guard"][:, k], color=f"C{k}", lw=0.7, ls=":")
        ax.axhline(1e3 * p.clearance_warn, color="k", lw=0.8, ls=":", label=f"warning {1e3 * p.clearance_warn:.0f} mm")
        ax.plot([], [], "k-", lw=1, label="net + taut strings")
        ax.plot([], [], "k--", lw=0.7, label="net only (knots, links)")
        ax.plot([], [], "k:", lw=0.7, label="prop guard, net + taut strings")
        if "guard" in clearance and np.isfinite(clearance["guard"]).any():
            ax.axhline(1e3 * p.guard_warn, color="tab:gray", lw=0.8, ls="-.", label=f"guard warning {1e3 * p.guard_warn:.0f} mm")
        ax.set_ylim(0, 200); ax.set_ylabel("rotor disc (solid) / guard (dotted)\nclearance [mm]")
        ax.legend(fontsize=7, loc="upper right")
        ax = axs[3, 1]
        ax.plot(ft, clearance["footprint"], drawstyle="steps-post")
        ax.set_ylim(-0.2, max(4, clearance["footprint"].max() + 1))
        ax.set_ylabel(f"knots in rotor footprint\n(r < {1e3 * (p.arm + p.prop_radius):.1f} mm, |dz| < 30 mm)")
    for a in axs[-1]:
        a.set_xlabel("t [s]")
    for a in axs.ravel():
        a.grid(alpha=0.3)
        for (tm, lab) in marks:
            a.axvline(tm, color="tab:red", lw=0.6, ls=":")
    for (tm, lab) in marks:
        axs[0, 0].text(tm, axs[0, 0].get_ylim()[1], lab, fontsize=8, color="tab:red", va="bottom", ha="center")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_sweep(p, rows, path):
    d = np.array([r["spacing_m"] for r in rows])
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axs[0]
    ax.plot(d, [r["sag_of_net_m"] for r in rows], "o-", label="net centre below its corners")
    ax.plot(d, [r["sag_below_drones_m"] for r in rows], "s-", label="net centre below drones")
    ax.set_ylabel("sag [m]"); ax.legend()
    ax = axs[1]
    ax.plot(d, [r["tilt_deg"] for r in rows], "o-", label="simulated tilt")
    ax.plot(d, [r["tilt_pred_deg"] for r in rows], "k--", label="force balance from string angle")
    ax.plot(d, [r["string_angle_deg"] for r in rows], "^-", color="tab:green", label="string angle below horizontal")
    ax.set_ylabel("angle [deg]"); ax.legend()
    ax = axs[2]
    ax.plot(d, [r["tension_N"] / G * 1e3 for r in rows], "o-", label="string tension (sim)")
    ax.plot(d, [r["tension_pred_N"] / G * 1e3 for r in rows], "k--", label="string tension (predicted)")
    ax.set_ylabel("tension [gf]")
    ax2 = ax.twinx()
    ax2.plot(d, [r["motor_frac_max"] for r in rows], "x-", color="tab:red", label="max motor thrust / limit")
    ax2.plot(d, [r["pos_err_mean_mm"] / 100 for r in rows], "d-", color="tab:purple", label="steady position error / 100 mm")
    ax2.set_ylim(0, 1.05)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    for a in axs:
        a.set_xlabel("actual drone spacing D [m]"); a.grid(alpha=0.3)
    fig.suptitle(f"Quasi-static spacing sweep: {p.L:.1f} m net, {p.m_net * 1e3:.0f} g, {p.string * 100:.0f} cm corner strings")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_gif(p, log, path, fps=10):
    from matplotlib.animation import FuncAnimation, PillowWriter
    stride = max(1, int(round(1.0 / fps / np.diff(log["frames_t"]).mean())))
    idx = range(0, len(log["frames_t"]), stride)
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(projection="3d")

    frames = list(idx)
    print(f"  rendering animation: {len(frames)} frames (this is the slow part)", flush=True)

    def draw(i):
        if frames.index(i) % 50 == 0:
            print(f"  animation frame {frames.index(i):3d} / {len(frames)}", flush=True)
        ax.cla()
        draw_net(ax, p, log["frames"][i])
        ax.set_xlim(-0.7, 0.7); ax.set_ylim(-0.7, 0.7); ax.set_zlim(0, 1.4)
        ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=20, azim=-60)
        ax.set_title(f"t = {log['frames_t'][i]:.1f} s")

    FuncAnimation(fig, draw, frames=frames).save(path, writer=PillowWriter(fps=fps), dpi=70)
    plt.close(fig)


# ---------------------------------------------------------------- scenarios
def load_ff(p: Params, path, spacing=None):
    """Net pull vs spacing, identified by a previous sweep (scaled to this net's weight).

    Outside the table's spacing range np.interp holds the edge value (no extrapolation).
    Only the hold spacing is checked; the default takeoff spacing (1.20 m) is inside the
    table (the old 0.80 m takeoff was below it and used the 0.90 m edge value).
    Rows are sorted by spacing here, so the table may list them in any order."""
    with open(path) as f:
        js = json.load(f)
    tp = js["params"]
    for key, default, unit, sc in (("hook_z", -0.010, "mm", 1e3), ("yaw_deg", 0.0, "deg", 1.0)):
        if abs(tp.get(key, default) - getattr(p, key)) > 1e-9:
            print(f"note: the feedforward table {path} was made with {key} = {tp.get(key, default) * sc:g} {unit}, "
                  f"this run uses {getattr(p, key) * sc:g} {unit}", flush=True)
    rows = sorted((r for r in js["rows"] if r["motor_frac_max"] < 0.9), key=lambda r: r["spacing_m"])
    d = np.array([r["spacing_m"] for r in rows])
    if spacing is not None and not d.min() <= spacing <= d.max():
        print(f"warning: hold spacing {spacing:.3f} m is outside the feedforward table range "
              f"{d.min():.3f}-{d.max():.3f} m ({path}); the feedforward uses the edge value "
              f"there (no extrapolation), so the PID integrator has to absorb the difference", flush=True)
    m_tab = js["params"]["m_net"] + (js.get("payload") or {}).get("mass_kg", 0.0)
    if (js.get("payload") or getattr(p, "payload_mass", 0.0)) and abs(m_tab - hanging_mass(p)) > 1e-9:
        print(f"note: the feedforward table {path} was made for {1e3 * m_tab:g} g hanging, this run carries "
              f"{1e3 * hanging_mass(p):g} g (pull scaled; its lift height is not)", flush=True)
    fh = np.array([r["pull_horizontal_gf"] for r in rows]) * G / 1e3 * hanging_mass(p) / m_tab
    lift = np.array([r["sag_below_drones_m"] for r in rows])
    return (lambda spacing: float(np.interp(spacing, d, fh)),
            lambda spacing: float(np.interp(spacing, d, lift)))


def landing_stats(p: Params, log, t_contract0, t_desc0, t_desc1, t_off):
    t, pos, ref = log["t"], log["pos"], log["ref"]
    s = (t >= t_contract0) & (t < t_off)
    herr = np.linalg.norm((pos[s] - ref[s])[:, :, :2], axis=2)
    zerr = np.abs((pos[s] - ref[s])[:, :, 2])
    vz = np.gradient(pos[:, :, 2], t, axis=0)
    desc = (t >= t_desc0) & (t < t_off)
    n = p.n
    nodes, hooks = log["frames"][-1][:n * n], log["frames"][-1][n * n:]
    clearance = [float(np.linalg.norm(nodes[:, :2] - h[:2], axis=1).min()) for h in hooks]
    final = pos[-1]
    return {
        "horizontal_err_max_mm": float(1e3 * herr.max()),
        "height_err_max_mm": float(1e3 * zerr.max()),
        "tilt_max_deg_during_descent": float(log["tilt"][desc].max()),
        "tension_max_gf_during_descent": float(log["tension"][desc].max() / G * 1e3),
        "descent_speed_max_m_s": float(-vz[desc].min()),
        "touchdown_speed_m_s": float(-vz[t >= t_desc1].min()),
        "final_tilt_max_deg": float(log["tilt"][-1].max()),
        "final_height_mm": (1e3 * final[:, 2]).round(1).tolist(),
        "final_speed_max_mm_s": float(1e3 * np.abs(np.gradient(pos, t, axis=0)[-1]).max()),
        "final_spacing_m": float(np.mean([np.linalg.norm(final[a, :2] - final[(a + 1) % 4, :2]) for a in range(4)])),
        "final_offset_from_ref_mm": float(1e3 * np.linalg.norm((final - ref[-1])[:, :2], axis=1).max()),
        "net_max_height_mm": float(1e3 * nodes[:, 2].max()),
        "net_corner_span_m": float(log["corner_span"][-1]),
        "drone_to_nearest_knot_mm": (1e3 * np.array(clearance)).round(1).tolist(),
    }


def quat_to_mat(q):
    """Rotation matrices (body to world) from MuJoCo quaternions (w, x, y, z), any leading shape."""
    w, x, y, z = np.moveaxis(q / np.linalg.norm(q, axis=-1, keepdims=True), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def rotor_layout(p: Params):
    """Rotor disc centres in the body frame (4, 3) for the selected drone model, at height prop_z."""
    if is_official(p):
        xy = np.array(ROTORS_OFFICIAL, float) * p.rotor_xy
    else:
        xy = np.array(CORNERS, float) * p.arm / np.sqrt(2)
    return np.column_stack([xy, np.full(4, p.prop_z)])


def disc_distance(q, c, nrm, r):
    """Distance from points to flat discs of radius r, per frame.

    q (F, M, 3) points, c (F, K, 3) disc centres, nrm (F, K, 3) unit normals -> (F, K, M).
    With v = q - c: height h = v.n, in-plane radius rho = sqrt(|v|^2 - h^2), distance
    hypot(max(rho - r, 0), h). Dot products by matmul (no (F, K, M, 3) temporaries)."""
    qt = np.swapaxes(q, 1, 2)
    h = nrm @ qt - np.sum(c * nrm, axis=-1)[..., None]
    v2 = np.sum(q * q, axis=-1)[:, None, :] - 2.0 * (c @ qt) + np.sum(c * c, axis=-1)[..., None]
    rho = np.sqrt(np.maximum(v2 - h * h, 0.0))
    return np.hypot(np.maximum(rho - r, 0.0), h)


def net_links(n):
    """Knot index pairs (flattened i * n + j) and names of the net's links."""
    a, b, names = [], [], []
    for i in range(n):
        for j in range(n):
            if i < n - 1:
                a.append(i * n + j); b.append((i + 1) * n + j); names.append(f"a{i}_{j}")
            if j < n - 1:
                a.append(i * n + j); b.append(i * n + j + 1); names.append(f"b{i}_{j}")
    return np.array(a), np.array(b), names


# ---------------------------------------------------------------- propeller guards (clearance check only)
# The real Crazyflie 2.1 Brushless prop guards, from Crazyflow's mesh
#   assets/crazyflow_cf21B/assets/cf21B/cf21B_prop-guards.stl (learnsyslab/drone-models, commit 6c72a95,
#   see assets/crazyflow_cf21B/SOURCE.txt; binary STL in mm, model origin = our body origin, rotors at
#   +-35.36 mm on the diagonals as ours), reduced once by guard_points_from_stl() below to a point set
#   that covers the mesh surface: every surface point lies within GUARD_COVER of one of the points.
#   It is embedded (_GUARD_B64, at the end of this file), so the script does not need the assets folder;
#   check_guard_model() rebuilds it from the STL when the folder is there and compares.
# Geometry (from the mesh): z -15.43..+15.43 mm, 40.1..84.4 mm from the body axis. Per rotor: a ring
#   30..37 mm from the rotor axis, z +5..+15.4, on the rotor's outer arc only (about 160 deg facing away
#   from the body centre), struts to it, a motor mount, and a foot under the motor that is the lowest
#   point of the airframe (z -15.43 mm, 49..51.5 mm from the body axis).
GUARD_STL = os.path.join(HERE, "assets", "crazyflow_cf21B", "assets", "cf21B", "cf21B_prop-guards.stl")
GUARD_FEET_Z = -0.01543         # m, lowest airframe point (guard feet) relative to the body origin
GUARD_COVER = 1.6e-3            # m, cover radius of the embedded point set (1.5 mm greedy cover + 0.05 mm grid, measured 1.56 mm)
_GUARD_UNIT = 5e-5              # m, quantisation of the embedded coordinates


def read_binary_stl(path, max_tri=500_000):
    """Triangles (n, 3, 3) of a binary STL, read as plain data (header, count, 50-byte records).
    Refuses files whose size does not match the triangle count, ASCII STL and non-finite values."""
    with open(path, "rb") as f:
        raw = f.read()
    if len(raw) < 84:
        raise ValueError(f"{path}: too short for a binary STL")
    n = int(np.frombuffer(raw, "<u4", 1, 80)[0])
    if n > max_tri or len(raw) != 84 + 50 * n:
        raise ValueError(f"{path}: not a binary STL of the declared size ({n} triangles, {len(raw)} bytes)")
    rec = np.frombuffer(raw, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n, 84)
    V = rec["v"].astype(float)
    if not np.isfinite(V).all():
        raise ValueError(f"{path}: non-finite vertex")
    return V


def guard_points_from_stl(path=GUARD_STL, cover=1.5e-3, scale=1e-3):
    """How the embedded guard model was made: sample every triangle on a barycentric grid at most
    cover/4 apart, then keep points greedily (in z, y, x order) so that no two kept points are closer
    than `cover`; every sample is then within `cover` of a kept point. Returns (M, 3) in metres."""
    V = read_binary_stl(path) * scale
    step = cover / 4
    e = np.max(np.linalg.norm(V[:, [1, 2, 0]] - V, axis=2), axis=1)
    ks = np.maximum(1, np.ceil(e / step).astype(int))
    dense = []
    for k in np.unique(ks):
        T = V[ks == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        m = i + j <= k
        a, b = i[m] / k, j[m] / k
        dense.append((T[:, None, 0] + a[None, :, None] * (T[:, None, 1] - T[:, None, 0])
                      + b[None, :, None] * (T[:, None, 2] - T[:, None, 0])).reshape(-1, 3))
    P = np.unique(np.vstack(dense).round(7), axis=0)
    P = P[np.lexsort((P[:, 0], P[:, 1], P[:, 2]))]
    cells, kept, r2 = {}, [], cover * cover
    for q in P:
        c = np.floor(q / cover).astype(int)
        near = (pt for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
                for pt in cells.get((c[0] + dx, c[1] + dy, c[2] + dz), ()))
        if all((q - pt) @ (q - pt) >= r2 for pt in near):
            kept.append(q)
            cells.setdefault(tuple(c), []).append(q)
    return np.array(kept)


def encode_guard(points):
    """Points (m) -> base64 text: int16 columns in _GUARD_UNIT, sorted by z, z delta-coded, zlib."""
    import base64
    import zlib
    q = np.rint(points / _GUARD_UNIT).astype(np.int64)
    q = q[np.lexsort((q[:, 0], q[:, 1], q[:, 2]))].T.copy()
    q[2] = np.diff(np.concatenate([[0], q[2]]))
    return base64.b64encode(zlib.compress(q.astype("<i2").tobytes(), 9)).decode()


_guard_cache = {}


def guard_points():
    """The embedded guard point set, (M, 3) in metres, body frame."""
    if "pts" not in _guard_cache:
        import base64
        import zlib
        q = np.frombuffer(zlib.decompress(base64.b64decode("".join(_GUARD_B64))), "<i2").astype(np.int64)
        q = q.reshape(3, -1)
        q[2] = np.cumsum(q[2])
        _guard_cache["pts"] = q.T * _GUARD_UNIT
    return _guard_cache["pts"]


def check_guard_model(path=GUARD_STL):
    """Compare the embedded guard points with the STL (needs the assets folder; takes ~20 s):
        python -c "import net_sim; net_sim.check_guard_model()"   (run in this folder)"""
    from scipy.spatial import cKDTree
    emb = guard_points()
    V = read_binary_stl(path) * 1e-3
    P = V.reshape(-1, 3)
    lo, hi = P.min(0), P.max(0)
    print(f"STL {path}: {len(V)} triangles, bbox [mm] x {lo[0] * 1e3:+.2f}..{hi[0] * 1e3:+.2f} "
          f"y {lo[1] * 1e3:+.2f}..{hi[1] * 1e3:+.2f} z {lo[2] * 1e3:+.2f}..{hi[2] * 1e3:+.2f}, "
          f"radial {np.hypot(P[:, 0], P[:, 1]).min() * 1e3:.2f}..{np.hypot(P[:, 0], P[:, 1]).max() * 1e3:.2f}")
    lo, hi = emb.min(0), emb.max(0)
    print(f"embedded: {len(emb)} points, bbox [mm] x {lo[0] * 1e3:+.2f}..{hi[0] * 1e3:+.2f} "
          f"y {lo[1] * 1e3:+.2f}..{hi[1] * 1e3:+.2f} z {lo[2] * 1e3:+.2f}..{hi[2] * 1e3:+.2f}, "
          f"radial {np.hypot(emb[:, 0], emb[:, 1]).min() * 1e3:.2f}..{np.hypot(emb[:, 0], emb[:, 1]).max() * 1e3:.2f}")
    new = guard_points_from_stl(path)
    d = cKDTree(new).query(emb)[0]
    print(f"rebuilt from the STL: {len(new)} points; embedded vs rebuilt: max {d.max() * 1e3:.3f} mm "
          f"(quantisation {_GUARD_UNIT * 1e3 / 2:.3f} mm), same encoding: {encode_guard(new) == ''.join(_GUARD_B64)}")
    # cover: fine samples of every triangle (at most 0.15 mm apart) -> nearest embedded point
    tree, worst = cKDTree(emb), 0.0
    e = np.max(np.linalg.norm(V[:, [1, 2, 0]] - V, axis=2), axis=1)
    ks = np.maximum(1, np.ceil(e / 1.5e-4).astype(int))
    for k in np.unique(ks):
        T = V[ks == k]
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        m = i + j <= k
        a, b = i[m] / k, j[m] / k
        step = max(1, 200_000 // len(a))
        for c0 in range(0, len(T), step):
            Tc = T[c0:c0 + step]
            S = (Tc[:, None, 0] + a[None, :, None] * (Tc[:, None, 1] - Tc[:, None, 0])
                 + b[None, :, None] * (Tc[:, None, 2] - Tc[:, None, 0])).reshape(-1, 3)
            worst = max(worst, tree.query(S)[0].max())
    print(f"cover: every surface sample is within {worst * 1e3:.3f} mm of an embedded point "
          f"(GUARD_COVER = {GUARD_COVER * 1e3:.2f} mm)")
    return worst


def guard_part(p, g):
    """Name the part of the guard at body point g (m): nearest rotor and foot / mount / strut / ring."""
    xy = rotor_layout(p)[:, :2]
    r = int(np.argmin(np.linalg.norm(xy - g[:2], axis=1)))
    rho = float(np.linalg.norm(xy[r] - g[:2]))
    part = "foot" if g[2] < -0.006 else "mount" if rho < 0.013 else "strut" if rho < 0.030 else "ring"
    return r, part


def guard_clearance(p, nodes, hooks, Pd, R, taut, chunk=20, link_step=0.005, string_step=0.002, near=0.10):
    """Lower bound on the distance from each drone's propeller guards to the net, per frame.

    Elements as in clearance_analysis (knots, links, taut strings), sampled finer (links every
    link_step, strings every string_step). Each sample is moved into the drone's body frame; within
    `near` of the guard's bounding sphere its distance to the nearest embedded guard point is taken,
    minus GUARD_COVER (so the value is a lower bound, at most ~2 * GUARD_COVER below the true
    distance); farther samples get the bounding-sphere distance (a lower bound, > `near`).
    Returns dicts out (F, 4) [m], idx (F, 4) element index, gpt (F, 4, 3) guard point (body frame, m),
    keyed knot / link / own / other."""
    from scipy.spatial import cKDTree
    G_ = guard_points()
    tree = cKDTree(G_)
    r_bound = float(np.linalg.norm(G_, axis=1).max()) + GUARD_COVER
    n, N = p.n, p.n * p.n
    F = len(nodes)
    la, lb, _ = net_links(n)
    m_link = max(2, int(np.ceil(p.L / (n - 1) / link_step)))
    s_link = np.arange(1, m_link) / m_link
    m_str = max(1, int(np.ceil((p.string if p.string_each is None else max(p.string_each)) / string_step)))
    s_str = np.arange(m_str) / m_str
    ci = np.array([(n - 1 if sx > 0 else 0) * n + (n - 1 if sy > 0 else 0) for sx, sy in CORNERS])
    nl, ns_ = len(la) * len(s_link), 4 * m_str
    out = {key: np.full((F, 4), np.inf) for key in ("knot", "link", "own", "other")}
    idx = {key: np.zeros((F, 4), int) for key in out}
    gpt = {key: np.full((F, 4, 3), np.nan) for key in out}
    eye = np.eye(4, dtype=bool)
    for f0 in range(0, F, chunk):
        sl = slice(f0, min(F, f0 + chunk))
        nd, hk = nodes[sl], hooks[sl]
        f_n = len(nd)
        pa, pb = nd[:, la], nd[:, lb]
        lpts = (pa[:, :, None] + (pb - pa)[:, :, None] * s_link[None, None, :, None]).reshape(f_n, -1, 3)
        cn = nd[:, ci]
        spts = (hk[:, :, None] + (cn - hk)[:, :, None] * s_str[None, None, :, None]).reshape(f_n, -1, 3)
        Q = np.concatenate([nd, lpts, spts], axis=1)                      # (f, M, 3) world
        # body frame of every drone: q_b = R^T (q - P)
        qb = np.einsum("fmj,fkji->fkmi", Q, R[sl]) - np.einsum("fkj,fkji->fki", Pd[sl], R[sl])[:, :, None]
        rad = np.linalg.norm(qb, axis=-1)                                 # (f, drone, M)
        dist = rad - r_bound
        gi = np.full(dist.shape, -1)
        m = rad < r_bound + near
        if m.any():
            d_, i_ = tree.query(qb[m])
            dist[m] = np.maximum(d_ - GUARD_COVER, 0.0)
            gi[m] = i_
        dk, gk = dist[..., :N], gi[..., :N]
        dl = dist[..., N:N + nl].reshape(f_n, 4, len(la), len(s_link))
        gl = gi[..., N:N + nl].reshape(f_n, 4, len(la), len(s_link))
        ds = dist[..., N + nl:].reshape(f_n, 4, 4, m_str)
        gs = gi[..., N + nl:].reshape(f_n, 4, 4, m_str)
        ds = np.where(taut[sl][:, None, :, None], ds, np.inf)             # slack strings not measured

        def put(key, d2, g2, mask=None):
            """d2 (f, drone, element, samples): min over samples then elements."""
            if mask is not None:
                d2 = np.where(mask[None, :, :, None], d2, np.inf)
            flat = d2.reshape(f_n, 4, -1)
            j = flat.argmin(axis=2)
            out[key][sl] = np.take_along_axis(flat, j[..., None], 2)[..., 0]
            idx[key][sl] = j // d2.shape[3]
            g = np.take_along_axis(g2.reshape(f_n, 4, -1), j[..., None], 2)[..., 0]
            gpt[key][sl] = np.where((g >= 0)[..., None], G_[np.maximum(g, 0)], np.nan)

        put("knot", dk[..., None], gk[..., None])
        put("link", dl, gl)
        put("own", ds, gs, mask=eye)
        put("other", ds, gs, mask=~eye)
    return out, idx, gpt


PHASE_NAMES = {"on_ground": "on ground", "takeoff": "takeoff", "spread": "spread",
               "hold_gust": "hold and gust", "relax_descent": "relax and descent", "landed": "landed",
               "pre_travel": "hold before trip", "accelerate": "accelerate", "cruise": "cruise",
               "decelerate": "decelerate", "settle": "settle after stop"}


def clearance_analysis(p: Params, log, phases, sample=0.01, chunk=50):
    """Minimum distance between each drone's four rotor discs and the net, at every logged frame.

    Rotor discs: flat, radius prop_radius, centred at the rotor position in the body frame at
    height prop_z above the centre of mass (so they tilt with the drone). Measured against
    (a) every knot, (b) every link (straight segment, sampled about every `sample` m, end
    knots excluded), (c) every corner string while it is taut (straight, hook to corner knot,
    sampled the same way). A string counts as taut when its tendon limit force is > 0 or its
    end-to-end length is within 0.5 mm of full length; slack strings are not measured (their
    shape is not simulated) and the time each drone's own string is slack while the drone is
    airborne is reported instead. Also counted: knots within arm + prop_radius horizontally and
    30 mm vertically of the drone centre (the rotor footprint).
    Propeller guards (guard_clearance): the same elements against the embedded guard mesh points,
    reported per phase under "guard" with their own warning threshold (guard_warn).
    phases: list of (key, t0, t1); the last phase includes t1.
    Returns (summary dict for the JSON, per-frame series for plotting / the log)."""
    n, N = p.n, p.n * p.n
    ft = log["frames_t"]
    F = len(ft)
    nodes, hooks = log["frames"][:, :N], log["frames"][:, N:]
    Pd = log["frames_pos"]
    R = quat_to_mat(log["frames_quat"])                                  # (F, drone, 3, 3)
    C = Pd[:, :, None, :] + np.einsum("fkij,rj->fkri", R, rotor_layout(p))  # (F, drone, rotor, 3)
    nrm = R[:, :, None, :, 2]                                             # (F, drone, 1, 3)
    la, lb, lnames = net_links(n)
    m_link = max(1, int(np.ceil(p.L / (n - 1) / sample)))
    s_link = np.arange(1, m_link) / m_link                                # interior samples
    m_str = max(1, int(np.ceil((p.string if p.string_each is None else max(p.string_each)) / sample)))
    s_str = np.arange(m_str) / m_str                                      # hook ... just before the corner knot
    ci = np.array([(n - 1 if sx > 0 else 0) * n + (n - 1 if sy > 0 else 0) for sx, sy in CORNERS])
    # string taut flag per frame, using the 100 Hz tension log at the same instants
    li = np.clip(np.rint(np.interp(ft, log["t"], np.arange(len(log["t"])))).astype(int), 0, len(log["t"]) - 1)
    str_len = np.linalg.norm(nodes[:, ci] - hooks, axis=-1)               # (F, string)
    taut = (log["tension"][li] > 0) | (str_len >= (p.string if p.string_each is None
                                                   else np.asarray(p.string_each, float)) - 5e-4)
    r = p.prop_radius
    out = {key: np.full((F, 4), np.inf) for key in ("knot", "link", "own", "other")}
    idx = {key: np.zeros((F, 4), int) for key in out}                    # element index
    rot = {key: np.zeros((F, 4), int) for key in out}                    # rotor index
    for f0 in range(0, F, chunk):
        sl = slice(f0, min(F, f0 + chunk))
        nd = nodes[sl]
        f_n = len(nd)
        c = C[sl].reshape(f_n, 16, 3)                                    # all 16 discs (drone, rotor)
        nr = np.repeat(nrm[sl], 4, axis=2).reshape(f_n, 16, 3)
        shp = (f_n, 4, 4)
        # knots
        d = disc_distance(nd, c, nr, r).reshape(*shp, -1)                # (f, drone, rotor, knot)
        dk = d.min(axis=2)
        idx["knot"][sl] = dk.argmin(axis=2)
        out["knot"][sl] = dk.min(axis=2)
        rot["knot"][sl] = np.take_along_axis(d.argmin(axis=2), idx["knot"][sl][..., None], 2)[..., 0]
        # links
        pa, pb = nd[:, la], nd[:, lb]
        pts = (pa[:, :, None] + (pb - pa)[:, :, None] * s_link[None, None, :, None]).reshape(len(nd), -1, 3)
        d = disc_distance(pts, c, nr, r).reshape(*shp, len(la), len(s_link)).min(axis=4)
        dk = d.min(axis=2)
        idx["link"][sl] = dk.argmin(axis=2)
        out["link"][sl] = dk.min(axis=2)
        rot["link"][sl] = np.take_along_axis(d.argmin(axis=2), idx["link"][sl][..., None], 2)[..., 0]
        # strings (all four), masked where slack
        hk, cn = hooks[sl], nd[:, ci]
        pts = (hk[:, :, None] + (cn - hk)[:, :, None] * s_str[None, None, :, None]).reshape(len(nd), -1, 3)
        d = disc_distance(pts, c, nr, r).reshape(*shp, 4, len(s_str)).min(axis=4)
        d = np.where(taut[sl][:, None, None, :], d, np.inf)              # (f, drone, rotor, string)
        dk = d.min(axis=2)                                               # (f, drone, string)
        rk = d.argmin(axis=2)
        eye = np.eye(4, dtype=bool)
        out["own"][sl] = np.where(eye[None], dk, np.inf).min(axis=2)
        idx["own"][sl] = np.arange(4)[None]
        rot["own"][sl] = np.einsum("fkk->fk", rk)
        other = np.where(eye[None], np.inf, dk)
        idx["other"][sl] = other.argmin(axis=2)
        out["other"][sl] = other.min(axis=2)
        rot["other"][sl] = np.take_along_axis(rk, idx["other"][sl][..., None], 2)[..., 0]
    # rotor footprint: knots near the drone centre
    rel = nodes[:, None, :, :] - Pd[:, :, None, :]                       # (F, drone, knot, 3)
    footprint = ((np.hypot(rel[..., 0], rel[..., 1]) < p.arm + p.prop_radius)
                 & (np.abs(rel[..., 2]) < 0.030)).sum(axis=2)
    net = np.minimum(out["knot"], out["link"])
    allc = np.minimum(net, np.minimum(out["own"], out["other"]))
    # propeller guards (contact hazard), reported separately from the rotor discs (strike hazard)
    try:
        g_out, g_idx, g_pt = guard_clearance(p, nodes, hooks, Pd, R, taut)
        g_all = np.minimum(np.minimum(g_out["knot"], g_out["link"]), np.minimum(g_out["own"], g_out["other"]))
        g_note = None
    except ImportError as exc:                                           # scipy (cKDTree) missing
        g_out, g_all, g_note = None, np.full((F, 4), np.nan), f"guard clearance not available: {exc}"
        print(f"warning: {g_note}", flush=True)

    def describe(key, f, k):
        i = idx[key][f, k]
        if key == "knot":
            what = f"knot n{i // n}_{i % n}"
        elif key == "link":
            what = f"link {lnames[i]}"
        elif key == "own":
            what = f"own string str{k}"
        else:
            what = f"other string str{i}"
        return {"value_mm": float(1e3 * out[key][f, k]), "t_s": float(ft[f]), "drone": int(k),
                "rotor": int(rot[key][f, k]), "element": key, "what": what}

    def describe_g(key, f, k):
        i = g_idx[key][f, k]
        what = {"knot": f"knot n{i // n}_{i % n}", "link": f"link {lnames[i] if key == 'link' else ''}",
                "own": f"own string str{k}", "other": f"other string str{i}"}[key]
        res = {"value_mm": float(1e3 * g_out[key][f, k]), "t_s": float(ft[f]), "drone": int(k),
               "element": key, "what": what}
        g = g_pt[key][f, k]
        if np.isfinite(g).all():
            r_, part = guard_part(p, g)
            res.update(guard_rotor=r_, guard_part=part, guard_point_body_mm=(1e3 * g).round(1).tolist())
        else:
            res.update(guard_rotor=None, guard_part="far (bounding-sphere lower bound)", guard_point_body_mm=None)
        return res

    # own-string slack time while airborne (100 Hz log)
    t, dt_log = log["t"], float(np.median(np.diff(log["t"])))
    airborne = log["pos"][:, :, 2] > 0.020                               # rest height of the CoM is ~0.010 m
    slack = log["tension"] <= 0.0

    def slack_stats(mask_t):
        res = []
        for k in range(4):
            m = mask_t & airborne[:, k] & slack[:, k]
            edges = np.flatnonzero(np.diff(np.concatenate([[0], m.astype(int), [0]])))
            spans = [[round(float(t[a]), 2), round(float(t[b - 1] + dt_log), 2)] for a, b in zip(edges[::2], edges[1::2])]
            res.append({"slack_airborne_s": round(float(m.sum() * dt_log), 2),
                        "airborne_s": round(float((mask_t & airborne[:, k]).sum() * dt_log), 2),
                        "intervals_s": spans[:6] + ([["...", len(spans) - 6]] if len(spans) > 6 else [])})
        return res

    summary = {"method": {"prop_radius_m": p.prop_radius, "prop_z_m": p.prop_z, "prop_z_note": "assumed, not published",
                          "rotor_body_xy_m": rotor_layout(p)[:, :2].round(5).tolist(),
                          "link_sample_m": float(p.L / (n - 1) / m_link), "string_sample_m": float(p.string / m_str),
                          "warn_below_mm": 1e3 * p.clearance_warn, "frame_dt_s": float(np.median(np.diff(ft))),
                          "footprint_radius_mm": 1e3 * (p.arm + p.prop_radius), "footprint_dz_mm": 30.0,
                          "taut": "tendon limit force > 0 or length >= string - 0.5 mm; slack strings not measured",
                          "guard": g_note or {
                              "source": "Crazyflow cf21B_prop-guards.stl, reduced to an embedded point set (see GUARD_STL)",
                              "points": int(len(guard_points())), "cover_mm": 1e3 * GUARD_COVER,
                              "value": "lower bound: distance to the nearest guard point minus the cover radius",
                              "link_sample_m": 0.005, "string_sample_m": 0.002,
                              "warn_below_mm": 1e3 * p.guard_warn, "hazard": "contact / snag (the guard is not a blade)"}},
               "phases": [], "own_string_slack_while_airborne": slack_stats(np.ones(len(t), bool))}
    for pi, (key, t0, t1) in enumerate(phases):
        last = pi == len(phases) - 1
        fm = (ft >= t0) & ((ft <= t1) if last else (ft < t1))
        lm = (t >= t0) & ((t <= t1) if last else (t < t1))
        ph = {"phase": key, "t_s": [t0, t1]}
        if fm.any():
            fi = np.flatnonzero(fm)
            sub = allc[fi]
            f, k = np.unravel_index(np.argmin(sub), sub.shape)
            f = fi[f]
            best = min(out, key=lambda e: out[e][f, k])
            ph["min"] = describe(best, f, k)
            ph["by_element_mm"] = {}
            for e in out:
                se = out[e][fi]
                if np.isfinite(se).any():
                    fe, ke = np.unravel_index(np.argmin(se), se.shape)
                    ph["by_element_mm"][e] = describe(e, fi[fe], ke)
                else:
                    ph["by_element_mm"][e] = None                       # e.g. strings slack all phase
            fp = footprint[fi]
            ff_, kf = np.unravel_index(np.argmax(fp), fp.shape)
            ph["footprint_knots_max"] = int(fp.max())
            ph["footprint_knots_max_at"] = {"t_s": float(ft[fi[ff_]]), "drone": int(kf)}
            ph["footprint_frames_with_knots_s"] = float((fp.max(axis=1) > 0).sum() * summary["method"]["frame_dt_s"])
            ph["warning"] = bool(ph["min"]["value_mm"] < 1e3 * p.clearance_warn)
            if g_out is not None:
                sub = g_all[fi]
                fg, kg = np.unravel_index(np.argmin(sub), sub.shape)
                fg = fi[fg]
                best = min(g_out, key=lambda e: g_out[e][fg, kg])
                gph = {"min": describe_g(best, fg, kg), "by_element_mm": {}}
                for e in g_out:
                    se = g_out[e][fi]
                    if np.isfinite(se).any():
                        fe, ke = np.unravel_index(np.argmin(se), se.shape)
                        gph["by_element_mm"][e] = describe_g(e, fi[fe], ke)
                    else:
                        gph["by_element_mm"][e] = None
                gph["warning"] = bool(gph["min"]["value_mm"] < 1e3 * p.guard_warn)
                ph["guard"] = gph
        ph["own_string_slack_while_airborne"] = [s["slack_airborne_s"] for s in slack_stats(lm)]
        summary["phases"].append(ph)
    series = {"t": ft, "all": allc, "net": net, "footprint": footprint,
              "knot": out["knot"], "link": out["link"], "own": out["own"], "other": out["other"], "taut": taut,
              "guard": g_all}
    return summary, series


def payload_clearance(p, log):
    """Payload sphere (on the centre knot) to every rotor disc and to the guard point set, per frame:
    (disc, guard) arrays (F, drone) in m, sphere radius and guard cover subtracted."""
    from scipy.spatial import cKDTree
    n, N = p.n, p.n * p.n
    q = log["frames"][:, (n // 2) * n + n // 2][:, None, :]                       # (F, 1, 3)
    Pd = log["frames_pos"]
    R = quat_to_mat(log["frames_quat"])
    F = len(q)
    C = (Pd[:, :, None, :] + np.einsum("fkij,rj->fkri", R, rotor_layout(p))).reshape(F, 16, 3)
    nr = np.repeat(R[:, :, None, :, 2], 4, axis=2).reshape(F, 16, 3)
    disc = disc_distance(q, C, nr, p.prop_radius).reshape(F, 4, 4).min(axis=2) - p.payload_d / 2
    tree = cKDTree(guard_points())
    qb = np.einsum("fj,fkji->fki", q[:, 0], R) - np.einsum("fkj,fkji->fki", Pd, R)  # body frame, (F, drone, 3)
    gd, _ = tree.query(qb.reshape(-1, 3))
    guard = gd.reshape(F, 4) - GUARD_COVER - p.payload_d / 2
    return disc, guard


def print_clearance(p, cl):
    """Small table of the per-phase rotor-to-net clearance, then warning lines."""
    m = cl["method"]
    print(f"rotor-disc clearance to the net (prop_z {1e3 * m['prop_z_m']:.0f} mm assumed, disc r {1e3 * m['prop_radius_m']:.1f} mm, "
          f"warning below {m['warn_below_mm']:.0f} mm):")
    fmt = lambda e: "   -  " if e is None else f"{e['value_mm']:6.1f}"
    print(f"  {'phase':<18} {'t [s]':>11} {'min[mm]':>8} {'at t':>6}  {'element (drone/rotor)':<28} "
          f"{'knot':>6} {'link':>6} {'own':>6} {'other':>6} {'fp_knots':>8} {'slack_air[s]':>14}")
    for ph in cl["phases"]:
        t0, t1 = ph["t_s"]
        slack = "/".join(f"{s:.1f}" for s in ph["own_string_slack_while_airborne"])
        if "min" not in ph:
            print(f"  {PHASE_NAMES[ph['phase']]:<18} {t0:5.1f}-{t1:5.1f}   (no frames)")
            continue
        mn, be = ph["min"], ph["by_element_mm"]
        print(f"  {PHASE_NAMES[ph['phase']]:<18} {t0:5.1f}-{t1:5.1f} {mn['value_mm']:8.1f} {mn['t_s']:6.2f}  "
              f"{mn['what'] + ' (' + str(mn['drone']) + '/' + str(mn['rotor']) + ')':<28} "
              f"{fmt(be['knot'])} {fmt(be['link'])} {fmt(be['own'])} {fmt(be['other'])} {ph['footprint_knots_max']:8d} {slack:>14}")
    g = m.get("guard")
    if isinstance(g, str):
        print(f"  {g}")
    else:
        print(f"propeller-guard clearance to the net (contact, not a blade: Crazyflow guard mesh, {g['points']} points, "
              f"lower bound, warning below {g['warn_below_mm']:.0f} mm):")
        print(f"  {'phase':<18} {'t [s]':>11} {'min[mm]':>8} {'at t':>6}  {'element (drone)':<24} {'guard part (rotor)':<19} "
              f"{'knot':>6} {'link':>6} {'own':>6} {'other':>6}")
        for ph in cl["phases"]:
            if "guard" not in ph:
                continue
            t0, t1 = ph["t_s"]
            mn, be = ph["guard"]["min"], ph["guard"]["by_element_mm"]
            part = f"{mn['guard_part']} ({mn['guard_rotor']})" if mn["guard_rotor"] is not None else "far"
            print(f"  {PHASE_NAMES[ph['phase']]:<18} {t0:5.1f}-{t1:5.1f} {mn['value_mm']:8.1f} {mn['t_s']:6.2f}  "
                  f"{mn['what'] + ' (' + str(mn['drone']) + ')':<24} {part:<19} "
                  f"{fmt(be['knot'])} {fmt(be['link'])} {fmt(be['own'])} {fmt(be['other'])}")
    for ph in cl["phases"]:
        if ph.get("warning"):
            mn = ph["min"]
            print(f"WARNING: {PHASE_NAMES[ph['phase']]}: rotor-to-net clearance {mn['value_mm']:.1f} mm < "
                  f"{m['warn_below_mm']:.0f} mm at t = {mn['t_s']:.2f} s ({mn['what']}, drone {mn['drone']}, rotor {mn['rotor']})"
                  + (f"; up to {ph['footprint_knots_max']} knots inside the rotor footprint" if ph["footprint_knots_max"] else ""))
        if ph.get("guard", {}).get("warning"):
            mn = ph["guard"]["min"]
            print(f"WARNING: {PHASE_NAMES[ph['phase']]}: guard-to-net clearance {mn['value_mm']:.1f} mm < "
                  f"{g['warn_below_mm']:.0f} mm at t = {mn['t_s']:.2f} s ({mn['what']}, drone {mn['drone']}, "
                  f"guard {mn['guard_part']} at rotor {mn['guard_rotor']})")
    sl = cl["own_string_slack_while_airborne"]
    print("  own string slack while airborne (whole flight, not measured for clearance): "
          + ", ".join(f"drone {k} {s['slack_airborne_s']:.2f} s" for k, s in enumerate(sl)))


def plot_landed(p, log, path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    frame = log["frames"][-1]
    draw_net(ax, p, frame, proj=(0, 1))
    for h in frame[p.n * p.n:]:
        ax.add_patch(plt.Circle(h[:2], p.arm + 0.0275, fill=False, color="tab:red", lw=1))   # rotor disc envelope
    ax.set_aspect("equal"); ax.grid(alpha=0.3)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def attitude_stats(log, t0, t1, f_cut=3.0):
    """Tilt fluctuation in a window (100 Hz log): std and the rms of its content above f_cut
    (linear trend removed, FFT), worst drone."""
    s = (log["t"] >= t0) & (log["t"] < t1)
    t, tilt = log["t"][s], log["tilt"][s]
    dt = float(np.median(np.diff(t)))
    tt = t - t[0]
    x = tilt - np.array([np.polyval(np.polyfit(tt, tilt[:, k], 1), tt) for k in range(tilt.shape[1])]).T
    X = np.fft.rfft(x, axis=0)
    fr = np.fft.rfftfreq(len(t), dt)
    X_hi = np.where((fr > f_cut)[:, None], X, 0.0)
    rms_hi = np.sqrt((np.fft.irfft(X_hi, n=len(t), axis=0) ** 2).mean(axis=0))
    pk = fr[1:][np.argmax(np.abs(X[1:]), axis=0)]
    return {"window_s": [t0, t1], "tilt_std_deg": float(tilt.std(axis=0).max()),
            "tilt_rms_above_3hz_deg": float(rms_hi.max()), "f_cut_hz": f_cut,
            "tilt_std_deg_per_drone": tilt.std(axis=0).round(4).tolist(),
            "dominant_freq_hz_per_drone": pk.round(2).tolist()}


def hook_check(p: Params):
    """Hook height against the guard feet: a hook below them reaches the ground first."""
    above = p.hook_z - GUARD_FEET_Z
    need = above < 0
    return {"hook_z_m": p.hook_z, "guard_feet_z_m": GUARD_FEET_Z, "hook_above_feet_mm": round(1e3 * above, 2),
            "needs_hardware_change": bool(need),
            "note": ("hook below the guard feet: it would touch the ground before the feet; the real drone needs "
                     "longer landing legs (the simulation rests on its 10 mm collision box and does not model this)")
                    if need else ("hook within 1 mm of the guard-feet plane: it sits practically on the ground at rest"
                                  if above < 1e-3 else "hook above the guard feet")}


def run_hold(p: Params, spacing, out, gif, ff=None, tag="hold"):
    t_climb, t_spread0, t_spread1, t_gust = 4.5, 6.5, 12.5, 22.0
    # landing: relax the spacing a little, descend with the net still spread, cut the motors
    t_contract0, t_contract1, t_desc0, t_desc1 = 28.0, 31.0, 31.5, 37.5
    t_off, t_end = t_desc1 + 0.3, 40.0
    d_land = min(p.d_land, spacing)
    t_arm = 0.5
    # takeoff: climb at the takeoff spacing (default 1.20 m, drones outside the net corners), then spread
    plan = Plan(z_ramps=[(t_arm, t_climb, 0.012, p.z_hover), (t_desc0, t_desc1, p.z_hover, p.z_touch)],
                d_ramps=[(t_spread0, t_spread1, p.d_takeoff, spacing), (t_contract0, t_contract1, spacing, d_land)],
                t_end=t_end, t_arm=t_arm, t_off=t_off)
    gust = (0, t_gust, t_gust + 0.5, np.array([0.04, 0.02, 0.0]))     # |F| = 0.045 N, ~12 % of drone weight
    sim = Sim(p)
    t0 = time.time()
    log = sim.run(plan, gusts=[gust], ff=ff)
    wall = time.time() - t0
    hold = window_stats(p, log, t_gust - 4.0, t_gust)
    just_after_spread = window_stats(p, log, t_spread1, t_spread1 + 2.0)
    after = window_stats(p, log, t_contract0 - 4.0, t_contract0)
    landing = landing_stats(p, log, t_contract0, t_desc0, t_desc1, t_off)
    phases = [("on_ground", 0.0, t_arm), ("takeoff", t_arm, t_spread0), ("spread", t_spread0, t_spread1),
              ("hold_gust", t_spread1, t_contract0), ("relax_descent", t_contract0, t_off), ("landed", t_off, t_end)]
    t1c = time.time()
    clearance, cl_series = clearance_analysis(p, log, phases)
    wall_clear = time.time() - t1c
    for key in ("all", "net", "footprint", "taut", "guard"):
        log[f"clearance_{key}"] = cl_series[key]
    sto = (log["t"] >= t_arm) & (log["t"] < t_spread0)
    takeoff = {"window_s": [t_arm, t_spread0],
               "pos_err_max_mm": float(1e3 * np.linalg.norm(log["pos"][sto] - log["ref"][sto], axis=2).max()),
               "tilt_max_deg": float(log["tilt"][sto].max()),
               "motor_frac_max": float(log["motor"][sto].max() / p.f_max),
               "tension_max_gf": float(log["tension"][sto].max() / G * 1e3),
               "own_string_slack_while_airborne_s": next(ph for ph in clearance["phases"]
                                                         if ph["phase"] == "takeoff")["own_string_slack_while_airborne"]}
    s = (log["t"] >= t_gust) & (log["t"] < t_contract0)
    err = np.linalg.norm(log["pos"][s] - log["ref"][s], axis=2).max(axis=1)
    tg = log["t"][s]
    late = np.where(err > 0.010)[0]
    summary = {
        "mode": "hold", "feedforward": ff is not None, "spacing_m": spacing, "takeoff_spacing_m": p.d_takeoff,
        "sim_time_s": t_end, "wall_time_s": round(wall, 1), "clearance_wall_time_s": round(wall_clear, 2),
        "takeoff": takeoff, "first_2s_after_spread": just_after_spread, "hold": hold, "after_gust": after,
        "gust": {"drone": 0, "force_N": gust[3].tolist(), "duration_s": 0.5,
                 "peak_pos_err_mm": float(1e3 * err.max()),
                 "back_within_10mm_after_s": float(tg[late[-1]] - t_gust) if len(late) else 0.0},
        "landing": landing,
        "attitude_hold": attitude_stats(log, t_gust - 4.0, t_gust),
        "hook": hook_check(p),
        "clearance": clearance,
        "params": asdict(p),
    }
    if p.net_drag:
        summary["aero"] = aero_info(p)
    if payload_info(p):
        summary["payload"] = payload_info(p)
        pc_d, pc_g = payload_clearance(p, log)
        summary["payload_clearance"] = []
        for key, a_, b_ in phases:
            fw = (log["frames_t"] >= a_) & (log["frames_t"] <= b_)
            summary["payload_clearance"].append({"phase": key, "disc_min_mm": round(float(1e3 * pc_d[fw].min()), 1),
                                                 "guard_min_mm": round(float(1e3 * pc_g[fw].min()), 1)})
        z = log["frames"][:, (p.n // 2) * p.n + p.n // 2, 2]
        up = np.flatnonzero(z > p.payload_d / 2 + 0.002)
        summary["payload_liftoff_s"] = float(log["frames_t"][up[0]]) if len(up) else None
    print(f"  simulation done in {wall:.1f} s (clearance check {wall_clear:.2f} s), writing figures to {out}", flush=True)
    np.savez_compressed(os.path.join(out, f"{tag}_log.npz"), **log)
    with open(os.path.join(out, f"{tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    plot_timeseries(p, log, os.path.join(out, f"{tag}_timeseries.png"),
                    marks=[(t_arm, "takeoff"), (t_spread0, "spread"), (t_spread1, "hold"), (t_gust, "gust"),
                           (t_contract0, "relax"), (t_desc0, "descend"), (t_off, "off")], clearance=cl_series)
    plot_snapshot(p, log, t_gust - 1.0, os.path.join(out, f"{tag}_snapshot.png"),
                  f"Hold at D = {spacing:.2f} m: sag {hold['sag_of_net_m'] * 100:.1f} cm, "
                  f"drone tilt {hold['tilt_deg']:.1f} deg")
    plot_landed(p, log, os.path.join(out, f"{tag}_landed.png"),
                f"After landing (top view): spacing {landing['final_spacing_m']:.2f} m")
    if gif:
        save_gif(p, log, os.path.join(out, f"{tag}.gif"))
    return summary


def run_sweep(p: Params, spacings, out, clearance=False):
    """Quasi-static sweep. Visit order: spacings below the takeoff spacing first, going down,
    then the rest going up from the takeoff spacing, so the high-load end is still approached
    in small steps (a takeoff below the table, e.g. 0.80 m, gives the plain ascending sweep).
    A step longer than 0.1 m gets a proportionally longer ramp. Rows are returned (and
    written) in ascending spacing order whatever the visit order (np.interp needs that)."""
    ramp0, dwell = 2.5, 4.5
    t = 6.5
    spacings = sorted(spacings)
    order = [s for s in spacings if s < p.d_takeoff][::-1] + [s for s in spacings if s >= p.d_takeoff]
    d_ramps, windows, prev = [], [], p.d_takeoff
    for dd in order:
        step = abs(dd - prev)
        ramp = ramp0 if step <= 0.1 + 1e-9 else ramp0 * step / 0.1
        d_ramps.append((t, t + ramp, prev, dd))
        windows.append((t + ramp + dwell - 2.0, t + ramp + dwell))
        t += ramp + dwell
        prev = dd
    plan = Plan(z_ramps=[(0.5, 4.5, 0.012, p.z_hover)], d_ramps=d_ramps, t_end=t)
    sim = Sim(p)
    t0 = time.time()
    log = sim.run(plan, noise=False)
    wall = time.time() - t0
    print(f"  simulation done in {wall:.1f} s, writing figures to {out}", flush=True)
    rows = [window_stats(p, log, a, b) for a, b in windows]
    if clearance:                      # rotor-disc and guard clearance to the net in each 2 s window
        cl, _ = clearance_analysis(p, log, [(f"D{dd:g}", a, b) for dd, (a, b) in zip(order, windows)])
        for r, ph in zip(rows, cl["phases"]):
            r["disc_min_mm"] = round(ph["min"]["value_mm"], 2)
            r["disc_min_what"] = ph["min"]["what"]
            r["guard_min_mm"] = round(ph["guard"]["min"]["value_mm"], 2)
            r["guard_min_what"] = ph["guard"]["min"]["what"]
    rows = sorted(rows, key=lambda r: r["spacing_ref_m"])
    if np.any(np.diff([r["spacing_m"] for r in rows]) <= 0):
        print("warning: the measured spacings of the sweep rows are not strictly increasing", flush=True)
    np.savez_compressed(os.path.join(out, "sweep_log.npz"), **log)
    with open(os.path.join(out, "sweep_summary.json"), "w") as f:
        js = {"mode": "sweep", "wall_time_s": round(wall, 1), "visit_order_m": order, "rows": rows,
              "params": asdict(p)}
        if payload_info(p):
            js["payload"] = payload_info(p)
        json.dump(js, f, indent=2)
    plot_sweep(p, rows, os.path.join(out, "sweep.png"))
    plot_timeseries(p, log, os.path.join(out, "sweep_timeseries.png"))
    return rows, wall


# ---------------------------------------------------------------- travel: the formation moves while carrying the net
TRAVEL_CRIT = {"control_err_m": 0.30, "control_tilt_deg": 60.0,            # under control, nothing hits the floor
               "motor_frac": 0.90, "motor_time_s": 0.10,                   # no motor above 90 % for more than 0.1 s
               "err_cruise_m": 0.050, "err_ramp_m": 0.100,                 # position error (3-D, worst drone)
               "disc_m": 0.030, "guard_m": 0.010,                          # clearances, takeoff to the end
               "net_low_m": 0.30,                                          # lowest knot above the floor
               "settle_band_m": 0.010, "settle_s": 5.0}                    # back within 10 mm within 5 s
CRIT_ORDER = ["control", "motor90", "pos_err", "clearance", "net_low", "settle"]
TRAVEL_HEADINGS = {"side": 0.0, "diagonal": 45.0}    # side: along +x (drones 0, 3 lead); diagonal: drone 0 leads
TRAVEL_DEFAULT = dict(speed=1.0, accel=1.0, heading="side", heading_deg=None, t_cruise=8.0, t_settle=8.0,
                      spacing=1.25, net_drag=True, keep_linear=False, thread_d=None, thread_d_scale=1.0,
                      net_mass=None, acc_ff=1.0, ff_path=None, variant="base", p_over=None)


def _runs(mask):
    """(start, end) index arrays (end exclusive) of the True runs of a 1-D bool array."""
    e = np.flatnonzero(np.diff(np.concatenate([[0], np.asarray(mask, np.int8), [0]])))
    return e[::2], e[1::2]


def travel_params(cfg):
    p = OfficialParams()
    for k, v in (cfg.get("p_over") or {}).items():
        setattr(p, k, tuple(v) if isinstance(v, list) else v)
    if cfg["net_mass"] is not None:
        p.m_net = cfg["net_mass"]
    p.net_drag, p.net_drag_keep_linear = bool(cfg["net_drag"]), bool(cfg["keep_linear"])
    if cfg["thread_d"] is not None:
        p.thread_d = cfg["thread_d"]
    if cfg["thread_d_scale"] != 1.0:
        p.thread_d = cfg["thread_d_scale"] * thread_diameter(p)
    return p


def travel_plan(p, v, a_max, heading_deg, t_cruise=8.0, t_settle=8.0, spacing=1.25):
    """Takeoff and spread exactly as in the hold scenario, settle until 15 s, then the trip."""
    t_arm, t_climb, t_spread0, t_spread1, t_go = 0.5, 4.5, 6.5, 12.5, 15.0
    plan = TravelPlan(z_ramps=[(t_arm, t_climb, 0.012, p.z_hover)],
                      d_ramps=[(t_spread0, t_spread1, p.d_takeoff, spacing)],
                      t_go=t_go, v_cruise=v, a_max=a_max, heading_deg=heading_deg,
                      t_cruise=t_cruise, t_settle=t_settle, t_arm=t_arm)
    return plan, (t_arm, t_spread0, t_spread1)


def travel_metrics(p, log, plan, cl, cl_series, t_arm, t_spread1, spacing):
    """Per-phase metrics, steady-cruise force balance, settling and the pass / fail criteria."""
    C = TRAVEL_CRIT
    t = log["t"]
    dt = float(np.median(np.diff(t)))
    ph = plan.phases()
    u = np.array([*plan.u, 0.0])
    cr = np.array([-plan.u[1], plan.u[0], 0.0])                  # cross-track: u turned +90 deg
    pos, err = log["pos"], log["pos"] - log["ref"]
    e_al, e_cr, e_z = err @ u, err @ cr, err[..., 2]
    e3 = np.linalg.norm(err, axis=2)
    side = np.stack([np.linalg.norm(pos[:, a, :2] - pos[:, (a + 1) % 4, :2], axis=1) for a in range(4)], 1)
    diag = np.stack([np.linalg.norm(pos[:, a, :2] - pos[:, a + 2, :2], axis=1) for a in range(2)], 1)
    sdev = side - log["spacing_ref"][:, None]
    ddev = diag - np.sqrt(2) * log["spacing_ref"][:, None]
    mf = log["motor"] / p.f_max
    mraw = log["motor_raw"] / p.f_max
    fc = pos.mean(axis=1)                                         # formation centre (actual drones)
    off = {}
    for key, xc in (("centre_knot", log["net_c"]), ("net_mass_centre", log["net_com"])):
        r = xc - fc
        off[key] = np.stack([r @ u, r @ cr, r[:, 2]], 1)          # along (+ ahead), cross, vertical
    tgf = log["tension"] / G * 1e3
    r_ = lambda x, k=1: np.round(np.asarray(x, float), k).tolist()
    phase_cl = {q["phase"]: q for q in cl["phases"]}

    def longest(mask2):                                           # (T, drone) -> per-drone longest run [s]
        out = []
        for k in range(mask2.shape[1]):
            a, b = _runs(mask2[:, k])
            out.append(float((b - a).max() * dt) if len(a) else 0.0)
        return out

    def block(name, t0, t1, last=False):
        s = (t >= t0) & ((t <= t1) if last else (t < t1))
        if not s.any():
            return None
        b = {"t_s": [round(t0, 3), round(t1, 3)], "err3d_max_mm": r_(1e3 * e3[s].max(0))}
        for key, e in (("along", e_al), ("cross", e_cr), ("vert", e_z)):
            es = e[s]
            i = np.abs(es).argmax(0)
            b[f"{key}_peak_mm"] = r_(1e3 * es[i, np.arange(4)])           # signed value at the peak
            b[f"{key}_rms_mm"] = r_(1e3 * np.sqrt((es ** 2).mean(0)))
        b["spacing_dev_max_mm"] = round(float(1e3 * np.abs(sdev[s]).max()), 1)
        b["spacing_dev_per_side_mm"] = r_(1e3 * np.abs(sdev[s]).max(0))     # sides 0-1, 1-2, 2-3, 3-0
        b["diagonal_dev_max_mm"] = round(float(1e3 * np.abs(ddev[s]).max()), 1)
        b["tilt_max_deg"] = r_(log["tilt"][s].max(0), 2)
        b["tilt_mean_deg"] = r_(log["tilt"][s].mean(0), 2)
        b["motor_max"] = r_(mf[s].max(axis=(0, 2)), 3)
        b["motor_mean"] = r_(mf[s].mean(axis=(0, 2)), 3)
        hi, sat = (mf[s] > C["motor_frac"]).any(axis=2), (mraw[s] >= 1.0).any(axis=2)
        b["motor_above90_s"] = r_(hi.sum(0) * dt, 2)
        b["motor_above90_longest_s"] = r_(longest(hi), 2)
        b["motor_saturated_s"] = r_(sat.sum(0) * dt, 2)
        b["tension_gf"] = {"mean": r_(tgf[s].mean(0), 2), "max": r_(tgf[s].max(0), 2), "min": r_(tgf[s].min(0), 2)}
        b["sag_net_m"] = r_([log["sag_net"][s].mean(), log["sag_net"][s].min(), log["sag_net"][s].max()], 4)
        b["sag_below_drones_m"] = r_([log["sag_drone"][s].mean(), log["sag_drone"][s].min(), log["sag_drone"][s].max()], 4)
        for key, o in off.items():
            b[f"{key}_offset_mm"] = {ax: r_(1e3 * np.array([o[s, i].mean(), o[s, i].min(), o[s, i].max()]))
                                     for i, ax in enumerate(("along", "cross", "vert"))}
        b["net_lowest_m"] = round(float(log["net_zmin"][s].min()), 4)
        q = phase_cl.get(name, {})
        b["clearance_disc_mm"] = round(q["min"]["value_mm"], 1) if "min" in q else None
        b["clearance_disc_what"] = q["min"]["what"] if "min" in q else None
        b["clearance_guard_mm"] = round(q["guard"]["min"]["value_mm"], 1) if "guard" in q else None
        return b

    phases = {"pre_travel": block("pre_travel", t_spread1, plan.t_go)}
    for i, (key, (a, b)) in enumerate(ph.items()):
        phases[key] = block(key, a, b, last=key == "settle")

    # steady cruise (last 3 s of the cruise) and force balance
    t0, t1 = max(ph["cruise"][0], ph["cruise"][1] - 3.0), ph["cruise"][1]
    s = (t >= t0) & (t < t1)
    steady = None
    if s.sum() > 10:
        V = plan.v
        W = p.m_drone * G
        aero = aero_info(p)
        c_tot = aero["total_drag_at_1_m_s_N"] if p.net_drag else 0.0
        lin_tot = aero["linear_stand_in_N_s_m_per_knot"] * p.n * p.n
        qd, ld = log["net_qdrag"][s].mean(0), log["net_ldrag"][s].mean(0)
        bd = log["body_drag"][s].mean(0)                              # (drone, 3)
        Ts = (log["tension"][s][:, :, None] * log["str_dir"][s]).mean(0)    # string force on each drone
        Tm = log["thrust_vec"][s].mean(0)
        Tp = np.array([0.0, 0.0, W]) - bd - Ts                        # zero acceleration: T + W + D + F_s = 0
        ang = lambda v: np.degrees(np.arctan2(np.hypot(v[..., 0], v[..., 1]), v[..., 2]))
        net_res = -Ts.sum(0) + np.array([0.0, 0.0, -p.m_net * G]) + qd + ld
        steady = {
            "window_s": [round(t0, 2), round(t1, 2)],
            "speed_m_s": r_(log["vel"][s].mean(0) @ u, 4),
            "knot_speed_mean_m_s": round(float(log["knot_vel_mean"][s].mean(0) @ u), 4),
            "along_err_mean_mm": r_(1e3 * e_al[s].mean(0)), "cross_err_mean_mm": r_(1e3 * e_cr[s].mean(0)),
            "vert_err_mean_mm": r_(1e3 * e_z[s].mean(0)),
            "tilt_mean_deg": r_(log["tilt"][s].mean(0), 2), "motor_mean": r_(mf[s].mean(axis=(0, 2)), 3),
            "motor_max": r_(mf[s].max(axis=(0, 2)), 3),
            "tension_mean_gf": r_(tgf[s].mean(0), 2),
            "net_offset_mm": {ax: round(float(1e3 * off["centre_knot"][s, i].mean()), 1)
                              for i, ax in enumerate(("along", "cross", "vert"))},
            "net_com_offset_mm": {ax: round(float(1e3 * off["net_mass_centre"][s, i].mean()), 1)
                                  for i, ax in enumerate(("along", "cross", "vert"))},
            "sag_net_m": round(float(log["sag_net"][s].mean()), 4),
            "net_lowest_m": round(float(log["net_zmin"][s].min()), 4),
            "net_drag_N": {"quadratic_sim_along": round(float(-qd @ u), 5),
                           "quadratic_hand": round(float(c_tot * V * V), 5),
                           "linear_stand_in_sim_along": round(float(-ld @ u), 5),
                           "linear_stand_in_hand": round(float(lin_tot * V), 5),
                           "quadratic_sim_vertical": round(float(qd[2]), 5)},
            "body_drag_N": {"sim_along": r_(-bd @ u, 5), "hand": round(float(-p.drag[0] * V), 5)},
            "force_balance": {
                "thrust_N": r_(np.linalg.norm(Tm, axis=1), 4), "thrust_pred_N": r_(np.linalg.norm(Tp, axis=1), 4),
                "thrust_tilt_deg": r_(ang(Tm), 2), "thrust_tilt_pred_deg": r_(ang(Tp), 2),
                "thrust_dir_err_deg": r_(np.degrees(np.arccos(np.clip(np.sum(Tm * Tp, 1) / np.linalg.norm(Tm, axis=1)
                                                                       / np.linalg.norm(Tp, axis=1), -1, 1))), 3),
                "string_force_along_N": r_(Ts @ u, 4), "string_force_vert_N": r_(Ts[:, 2], 4),
                "weight_N": round(W, 4), "body_drag_along_N": r_(bd @ u, 5),
                "net_residual_N": r_(net_res, 5),
                "net_strings_total_along_N": round(float(-Ts.sum(0) @ u), 5),
                "net_strings_total_up_N": round(float(-Ts.sum(0)[2]), 5),
                "net_weight_N": round(p.m_net * G, 5)},
        }

    # settling after the stop
    t_stop = ph["settle"][0]
    ss = t >= t_stop
    emax = e3.max(1)
    settle_s = None
    if ss.any():
        bad = np.flatnonzero(ss & (emax > C["settle_band_m"]))
        if not len(bad):
            settle_s = 0.0
        elif bad[-1] < len(t) - 1:
            settle_s = round(float(t[bad[-1]] + dt - t_stop), 3)
    swing_s = None
    if ss.sum() > 100:
        oa = off["centre_knot"][:, 0]
        final = oa[t >= t[-1] - 1.0].mean()
        bad = np.flatnonzero(ss & (np.abs(oa - final) > 0.010))
        swing_s = 0.0 if not len(bad) else (round(float(t[bad[-1]] + dt - t_stop), 3) if bad[-1] < len(t) - 1 else None)

    # criteria
    trav = t >= plan.t_go
    aborted = bool(np.isfinite(log.get("aborted_at", np.nan)))
    first_t = lambda mk: (float(t[np.flatnonzero(mk)[0]]) if mk.any() else None)
    crit = {}
    ctrl = trav & ((emax > C["control_err_m"]) | (log["tilt"].max(1) > C["control_tilt_deg"]) | (log["ncon"] > 0)
                   | ~np.isfinite(e3).all(1))
    crit["control"] = {"ok": not ctrl.any() and not aborted, "t_first": first_t(ctrl),
                       "err3d_max_mm": round(float(1e3 * emax[trav].max()), 1),
                       "tilt_max_deg": round(float(log["tilt"][trav].max()), 2),
                       "floor_contacts_max": int(log["ncon"][trav].max()), "aborted": aborted,
                       "drone_z_min_m": round(float(pos[trav][:, :, 2].min()), 4)}
    hi = trav & (mf > C["motor_frac"]).any(axis=(1, 2))
    a, b = _runs(hi)
    lg = float((b - a).max() * dt) if len(a) else 0.0
    n_lim = int(np.ceil(C["motor_time_s"] / dt - 1e-9))
    long_runs = [ai for ai, bi in zip(a, b) if bi - ai > n_lim]
    sat = trav & (mraw >= 1.0).any(axis=(1, 2))
    sa, sb = _runs(sat)
    crit["motor90"] = {"ok": lg <= C["motor_time_s"] + 1e-9, "longest_s": round(lg, 3),
                       "total_s": round(float(hi.sum() * dt), 3),
                       "t_first": float(t[long_runs[0] + n_lim]) if long_runs else None,
                       "motor_max": round(float(mf[trav].max()), 4),
                       "saturated_total_s": round(float(sat.sum() * dt), 3),
                       "saturated_longest_s": round(float((sb - sa).max() * dt) if len(sa) else 0.0, 3),
                       "saturated_t_first": first_t(sat)}
    cru = (t >= ph["cruise"][0]) & (t < ph["cruise"][1])
    ramp = ((t >= ph["accelerate"][0]) & (t < ph["accelerate"][1])) | ((t >= ph["decelerate"][0]) & (t < ph["decelerate"][1]))
    pe = (cru & (emax > C["err_cruise_m"])) | (ramp & (emax > C["err_ramp_m"]))
    crit["pos_err"] = {"ok": not pe.any(), "t_first": first_t(pe),
                       "cruise_max_mm": round(float(1e3 * emax[cru].max()), 1) if cru.any() else None,
                       "ramp_max_mm": round(float(1e3 * emax[ramp].max()), 1) if ramp.any() else None,
                       # cruise without its first second (the transient carried over from the acceleration)
                       "cruise_late_max_mm": round(float(1e3 * emax[cru & (t >= ph["cruise"][0] + 1.0)].max()), 1)
                       if (cru & (t >= ph["cruise"][0] + 1.0)).any() else None}
    ft = cl_series["t"]
    fm = ft >= t_arm
    dmin, gmin = cl_series["all"].min(1), np.nanmin(cl_series["guard"], axis=1)
    bad = fm & ((dmin < C["disc_m"]) | (gmin < C["guard_m"]))
    ftr = ft >= plan.t_go
    crit["clearance"] = {"ok": not bad.any(), "t_first": float(ft[np.flatnonzero(bad)[0]]) if bad.any() else None,
                         "disc_min_mm": round(float(1e3 * dmin[fm].min()), 1),
                         "guard_min_mm": round(float(1e3 * gmin[fm].min()), 1),
                         "disc_min_travel_mm": round(float(1e3 * dmin[ftr].min()), 1) if ftr.any() else None,
                         "guard_min_travel_mm": round(float(1e3 * gmin[ftr].min()), 1) if ftr.any() else None}
    nl = trav & (log["net_zmin"] <= C["net_low_m"])
    crit["net_low"] = {"ok": not nl.any(), "t_first": first_t(nl),
                       "net_lowest_m": round(float(log["net_zmin"][trav].min()), 4)}
    crit["settle"] = {"ok": settle_s is not None and settle_s <= C["settle_s"], "settle_s": settle_s,
                      "t_first": None if (settle_s is not None and settle_s <= C["settle_s"])
                      else float(min(t_stop + C["settle_s"], t[-1]))}
    failed = sorted((k for k in CRIT_ORDER if not crit[k]["ok"]),
                    key=lambda k: (crit[k]["t_first"] if crit[k]["t_first"] is not None else np.inf, CRIT_ORDER.index(k)))
    return {"pass": not failed, "failed": failed, "first_fail": failed[0] if failed else None,
            "criteria": crit, "phases": phases, "steady": steady, "settle_s": settle_s,
            "net_swing_settle_s": swing_s,
            "trip": {"T_accel_s": round(plan.T_a, 4), "t_go_s": plan.t_go, "t_stop_s": round(plan.t_d1, 4),
                     "distance_total_m": round(float(plan.v * (plan.T_a + plan.t_cruise)), 3),
                     "distance_accel_plus_brake_m": round(float(plan.v * plan.T_a), 3),
                     "distance_trapezoid_equiv_m": round(float(plan.v ** 2 / plan.a_max), 3)}}


def run_travel(cfg, out=None, tag=None, verbose=True):
    """One trip. cfg: TRAVEL_DEFAULT keys. Returns the metrics dict; with out, also writes
    <tag>_summary.json, <tag>_log.npz and <tag>_timeseries.png there."""
    cfg = {**TRAVEL_DEFAULT, **cfg}
    p = travel_params(cfg)
    hd = cfg["heading_deg"] if cfg["heading_deg"] is not None else TRAVEL_HEADINGS[cfg["heading"]]
    plan, (t_arm, t_spread0, t_spread1) = travel_plan(p, cfg["speed"], cfg["accel"], hd, cfg["t_cruise"],
                                                      cfg["t_settle"], cfg["spacing"])
    ff = load_ff(p, cfg["ff_path"], spacing=cfg["spacing"]) if cfg["ff_path"] else None
    sim = Sim(p)
    t0 = time.time()
    log = sim.run(plan, ff=ff, acc_ff=cfg["acc_ff"], extra_log=True, verbose=verbose, abort_err=1.0)
    wall = time.time() - t0
    phases = ([("on_ground", 0.0, t_arm), ("takeoff", t_arm, t_spread0), ("spread", t_spread0, t_spread1),
               ("pre_travel", t_spread1, plan.t_go)] + [(k, a, b) for k, (a, b) in plan.phases().items()])
    cl, cl_series = clearance_analysis(p, log, phases)
    met = travel_metrics(p, log, plan, cl, cl_series, t_arm, t_spread1, cfg["spacing"])
    met.update(cfg={k: v for k, v in cfg.items() if k != "p_over"} | {"p_over": cfg["p_over"]},
               heading_deg=hd, wall_s=round(wall, 1), aero=aero_info(p))
    if out:
        tag = tag or f"travel_{cfg['heading']}_v{cfg['speed']:g}_a{cfg['accel']:g}"
        for key in ("all", "net", "footprint", "taut", "guard"):
            log[f"clearance_{key}"] = cl_series[key]
        np.savez_compressed(os.path.join(out, f"{tag}_log.npz"), **log)
        with open(os.path.join(out, f"{tag}_summary.json"), "w") as f:
            json.dump({"mode": "travel", **met, "criteria_limits": TRAVEL_CRIT, "clearance": cl,
                       "params": asdict(p)}, f, indent=2)
        plot_travel_timeseries(p, log, plan, cl_series, met, os.path.join(out, f"{tag}_timeseries.png"))
        met["files"] = [f"{tag}_summary.json", f"{tag}_log.npz", f"{tag}_timeseries.png"]
    return met


def print_travel(met):
    c, cr = met["cfg"], met["criteria"]
    st = met["steady"] or {}
    print(f"travel {c['heading']} ({met['heading_deg']:g} deg) v={c['speed']:g} m/s a_max={c['accel']:g} m/s^2 "
          f"[{c['variant']}]: {'PASS' if met['pass'] else 'FAIL ' + ', '.join(met['failed'])}  "
          f"(ramp {met['trip']['T_accel_s']:.2f} s, distance {met['trip']['distance_total_m']:.1f} m, wall {met['wall_s']} s)")
    print(f"  control: err max {cr['control']['err3d_max_mm']:.1f} mm, tilt max {cr['control']['tilt_max_deg']:.1f} deg, "
          f"floor contacts {cr['control']['floor_contacts_max']}")
    print(f"  pos err: cruise {cr['pos_err']['cruise_max_mm']} mm (<= 50), accel/brake {cr['pos_err']['ramp_max_mm']} mm (<= 100)")
    print(f"  motors: max {cr['motor90']['motor_max']:.3f}, >90 % longest {cr['motor90']['longest_s']:.2f} s, "
          f"saturated {cr['motor90']['saturated_total_s']:.2f} s")
    print(f"  clearance: disc {cr['clearance']['disc_min_mm']} mm (travel {cr['clearance']['disc_min_travel_mm']}), "
          f"guard {cr['clearance']['guard_min_mm']} mm (travel {cr['clearance']['guard_min_travel_mm']})")
    print(f"  net lowest {cr['net_low']['net_lowest_m']:.3f} m, settle {met['settle_s']} s, "
          f"net swing settles in {met['net_swing_settle_s']} s")
    for name, b in met["phases"].items():
        if b is None:
            continue
        print(f"  {name:>11}: err3d {b['err3d_max_mm']} mm, along {b['along_peak_mm']} cross {b['cross_peak_mm']} mm, "
              f"tilt max {b['tilt_max_deg']} mean {b['tilt_mean_deg']}, motor max {b['motor_max']}, "
              f"T {b['tension_gf']['mean']} gf, net lag {b['centre_knot_offset_mm']['along']} mm, "
              f"low {b['net_lowest_m']} m, spacing dev {b['spacing_dev_max_mm']} mm, "
              f"disc {b['clearance_disc_mm']} guard {b['clearance_guard_mm']} mm")
    if st:
        fb = st["force_balance"]
        print(f"  steady cruise {st['window_s']}: speed {st['speed_m_s']} m/s, tilt {st['tilt_mean_deg']} deg, "
              f"motor mean {st['motor_mean']}, T {st['tension_mean_gf']} gf, net offset {st['net_offset_mm']} mm, "
              f"sag {st['sag_net_m']} m")
        print(f"    net drag sim/hand: quadratic {st['net_drag_N']['quadratic_sim_along']:.4f}/{st['net_drag_N']['quadratic_hand']:.4f} N, "
              f"linear {st['net_drag_N']['linear_stand_in_sim_along']:.4f}/{st['net_drag_N']['linear_stand_in_hand']:.4f} N; "
              f"body drag sim {st['body_drag_N']['sim_along']} hand {st['body_drag_N']['hand']} N")
        print(f"    thrust sim/pred {fb['thrust_N']} / {fb['thrust_pred_N']} N, tilt sim/pred {fb['thrust_tilt_deg']} / "
              f"{fb['thrust_tilt_pred_deg']} deg, dir err {fb['thrust_dir_err_deg']} deg, net residual {fb['net_residual_N']} N")


def plot_travel_timeseries(p, log, plan, cl_series, met, path):
    t = log["t"]
    u = np.array([*plan.u, 0.0])
    cr = np.array([-plan.u[1], plan.u[0], 0.0])
    s = t >= plan.t_go - 2.0
    ts = t[s]
    err = (log["pos"] - log["ref"])[s]
    fc = log["pos"][s].mean(axis=1)
    fig, axs = plt.subplots(5, 2, figsize=(14, 17), sharex=True)
    ax = axs[0, 0]
    ax.plot(ts, fc @ u, label="formation centre, along track")
    ax.plot(ts, log["ref"][s].mean(axis=1) @ u, "k--", lw=1, label="reference")
    ax.set_ylabel("distance [m]"); ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx()
    ax2.plot(ts, log["vel"][s].mean(axis=1) @ u, color="tab:green", lw=1)
    ax2.plot(ts, log["ref_vel"][s].mean(axis=1) @ u, color="tab:green", lw=0.8, ls="--")
    ax2.set_ylabel("speed [m/s] (green)", color="tab:green")
    ax = axs[0, 1]
    for k in range(4):
        ax.plot(ts, 1e3 * err[:, k] @ u, color=f"C{k}", lw=1, label=f"drone {k} along")
        ax.plot(ts, 1e3 * err[:, k] @ cr, color=f"C{k}", lw=0.7, ls="--")
        ax.plot(ts, 1e3 * err[:, k, 2], color=f"C{k}", lw=0.7, ls=":")
    for y in (50, -50, 100, -100):
        ax.axhline(y, color="k", lw=0.6, ls=":")
    ax.set_ylabel("position error [mm]\n(solid along + ahead, dashed cross, dotted vertical)"); ax.legend(fontsize=7, ncol=2)
    ax = axs[1, 0]
    ax.plot(ts, log["tilt"][s]); ax.set_ylabel("tilt [deg]")
    ax.legend([f"drone {k}" for k in range(4)], fontsize=7)
    ax = axs[1, 1]
    ax.plot(ts, log["motor"][s].max(axis=2) / p.f_max)
    ax.axhline(TRAVEL_CRIT["motor_frac"], color="k", lw=0.8, ls=":")
    ax.set_ylabel("highest motor thrust / limit"); ax.set_ylim(0, 1.05)
    ax = axs[2, 0]
    ax.plot(ts, log["tension"][s] / G * 1e3); ax.set_ylabel("string tension [gf]")
    ax = axs[2, 1]
    r = log["net_c"][s] - fc
    ax.plot(ts, 1e3 * (r @ u), label="along (+ ahead)")
    ax.plot(ts, 1e3 * (r @ cr), label="cross")
    ax.plot(ts, 1e3 * (r[:, 2] + 0.39), label="vertical + 390 mm")
    ax.set_ylabel("net centre knot - formation centre [mm]"); ax.legend(fontsize=8)
    ax = axs[3, 0]
    ax.plot(ts, log["net_zmin"][s], label="lowest knot")
    ax.plot(ts, log["pos"][s][:, :, 2].min(axis=1), label="lowest drone")
    ax.axhline(TRAVEL_CRIT["net_low_m"], color="k", lw=0.8, ls=":")
    ax.set_ylabel("height [m]"); ax.legend(fontsize=8)
    ax = axs[3, 1]
    side = np.stack([np.linalg.norm(log["pos"][s][:, a, :2] - log["pos"][s][:, (a + 1) % 4, :2], axis=1)
                     for a in range(4)], 1)
    ax.plot(ts, 1e3 * (side - log["spacing_ref"][s][:, None]))
    ax.legend(["side 0-1", "side 1-2", "side 2-3", "side 3-0"], fontsize=7)
    ax.set_ylabel("spacing - 1.25 m [mm]")
    ax = axs[4, 0]
    fs = cl_series["t"] >= plan.t_go - 2.0
    for k in range(4):
        ax.plot(cl_series["t"][fs], 1e3 * cl_series["all"][fs, k], color=f"C{k}", lw=1)
        ax.plot(cl_series["t"][fs], 1e3 * cl_series["guard"][fs, k], color=f"C{k}", lw=0.7, ls=":")
    ax.axhline(1e3 * TRAVEL_CRIT["disc_m"], color="k", lw=0.8, ls="--")
    ax.axhline(1e3 * TRAVEL_CRIT["guard_m"], color="tab:gray", lw=0.8, ls="-.")
    ax.set_ylim(0, 80); ax.set_ylabel("clearance [mm]\n(solid rotor disc, dotted guard)")
    ax = axs[4, 1]
    ax.plot(ts, -(log["net_qdrag"][s] + log["net_ldrag"][s]) @ u, label="net drag (total)")
    ax.plot(ts, -log["body_drag"][s] @ u, lw=0.8)
    ax.set_ylabel("drag along track [N]"); ax.legend(["net drag, total"] + [f"body drag drone {k}" for k in range(4)], fontsize=7)
    marks = [(plan.t_go, "go"), (plan.t_a1, "cruise"), (plan.t_d0, "brake"), (plan.t_d1, "stop")]
    for a in axs.ravel():
        a.grid(alpha=0.3)
        for tm, _ in marks:
            a.axvline(tm, color="tab:red", lw=0.6, ls=":")
    for tm, lab in marks:
        axs[0, 0].text(tm, axs[0, 0].get_ylim()[1], lab, fontsize=8, color="tab:red", va="bottom", ha="center")
    for a in axs[-1]:
        a.set_xlabel("t [s]")
    c = met["cfg"]
    fig.suptitle(f"Travel {c['heading']} ({met['heading_deg']:g} deg), cruise {c['speed']:g} m/s, a_max {c['accel']:g} m/s^2, "
                 f"[{c['variant']}]: {'PASS' if met['pass'] else 'FAIL: ' + ', '.join(met['failed'])}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _travel_job(cfg):
    try:
        return run_travel(cfg, verbose=False)
    except Exception as exc:                       # keep the sweep going; the run counts as a failure
        return {"cfg": cfg, "pass": False, "failed": ["error"], "first_fail": "error", "error": repr(exc)}


def _series_key(cfg):
    return (cfg["variant"], cfg["heading"], float(cfg["accel"]))


def travel_limit_search(pool, series, results, coarse, step=0.1, v_cap=10.0):
    """Speed-limit search for every series (cfg without speed): coarse grid, then extend upwards while
    everything passes, then fill the gap between the last pass and the first failure in `step`
    increments. results[key][speed] = metrics."""
    todo = [(c, v) for c in series for v in coarse if v not in results.setdefault(_series_key(c), {})]
    while todo:
        cfgs = [{**c, "speed": v} for c, v in todo]
        print(f"  running {len(cfgs)} trips ...", flush=True)
        t0 = time.time()
        for i, r in enumerate(pool.imap_unordered(_travel_job, cfgs)):
            results[_series_key(r["cfg"])][round(float(r["cfg"]["speed"]), 3)] = r
            if (i + 1) % 20 == 0 or i + 1 == len(cfgs):
                print(f"    {i + 1}/{len(cfgs)} done ({time.time() - t0:.0f} s)", flush=True)
        todo = []
        for c in series:
            res = results[_series_key(c)]
            vs = sorted(res)
            fails = [v for v in vs if not res[v]["pass"]]
            if not fails:
                if vs[-1] < v_cap:
                    todo.append((c, round(min(v_cap, vs[-1] * 1.5), 1)))
                continue
            vf = fails[0]
            vp = max([v for v in vs if v < vf], default=0.0)
            for v in np.arange(vp + step, vf - step / 2, step):
                v = round(float(v), 3)
                if v > 0 and v not in res:
                    todo.append((c, v))


def series_summary(res):
    """Limits of one series: last speed of the passing run before the first failure, and so on."""
    vs = sorted(res)
    fails = [v for v in vs if not res[v]["pass"]]
    vf = fails[0] if fails else None
    vp = max([v for v in vs if vf is None or v < vf], default=None)
    onset = {}
    for k in CRIT_ORDER + ["saturation"]:
        bad = [v for v in vs if ("criteria" in res[v]) and
               ((res[v]["criteria"]["motor90"]["saturated_total_s"] > 0) if k == "saturation"
                else not res[v]["criteria"][k]["ok"])]
        onset[k] = bad[0] if bad else None
    # alternative reading of the cruise rule: the cruise error counted from 1 s after the end of the ramp
    # (only on the speeds that happen to be tested; the search is driven by the strict rule)
    def alt_ok(r):
        if "criteria" not in r:
            return False
        c = r["criteria"]
        pe = c["pos_err"]
        return (all(c[k]["ok"] for k in CRIT_ORDER if k != "pos_err")
                and (pe["ramp_max_mm"] or 0) <= 1e3 * TRAVEL_CRIT["err_ramp_m"]
                and (pe["cruise_late_max_mm"] or 0) <= 1e3 * TRAVEL_CRIT["err_cruise_m"])
    alt_f = [v for v in vs if not alt_ok(res[v])]
    alt_vp = max([v for v in vs if not alt_f or v < alt_f[0]], default=None)
    return {"v_max_m_s": vp, "v_first_fail_m_s": vf,
            "alt_late_cruise_rule": {"v_max_m_s": alt_vp, "v_first_fail_m_s": alt_f[0] if alt_f else None},
            "first_fail": res[vf]["first_fail"] if vf is not None else None,
            "failed_at_first_fail": res[vf]["failed"] if vf is not None else [],
            "passes_above_first_fail": [v for v in vs if vf is not None and v > vf and res[v]["pass"]],
            "first_failing_speed_per_criterion": onset, "speeds_tested": vs}


def plot_travel_sweep(results, path, accels, variant="base"):
    heads = list(TRAVEL_HEADINGS)
    fig, axs = plt.subplots(6, 2, figsize=(15, 23), sharex=True)
    marker = {"control": "X", "motor90": "^", "pos_err": "s", "clearance": "D", "net_low": "v", "settle": "P", "error": "*"}
    for j, h in enumerate(heads):
        for ia, a in enumerate(accels):
            res = results.get((variant, h, float(a)))
            if not res:
                continue
            vs = np.array(sorted(res))
            ok = [v for v in vs if "criteria" in res[v]]
            col = f"C{ia}"
            g = lambda f: np.array([f(res[v]) for v in ok], float)
            ax = axs[0, j]
            for v in vs:
                r = res[v]
                if r["pass"]:
                    ax.plot(v, ia, "o", color="tab:green", ms=6)
                else:
                    ax.plot(v, ia, marker.get(r["first_fail"], "x"), color="tab:red", ms=7)
            sm = series_summary(res)
            if sm["v_max_m_s"]:
                ax.annotate(f"{sm['v_max_m_s']:g}", (sm["v_max_m_s"], ia), xytext=(0, 7), textcoords="offset points",
                            fontsize=8, ha="center", color="tab:green")
            axs[1, j].plot(ok, g(lambda r: r["criteria"]["pos_err"]["cruise_max_mm"] or np.nan), "-o", ms=3, color=col,
                           label=f"a {a:g} cruise")
            axs[1, j].plot(ok, g(lambda r: r["criteria"]["pos_err"]["ramp_max_mm"] or np.nan), "--", color=col, lw=1)
            axs[2, j].plot(ok, g(lambda r: r["criteria"]["motor90"]["motor_max"]), "-o", ms=3, color=col, label=f"a {a:g}")
            axs[2, j].plot(ok, g(lambda r: min(1.0, 0.5 + r["criteria"]["motor90"]["longest_s"])), ":", color=col, lw=1)
            axs[3, j].plot(ok, g(lambda r: r["criteria"]["clearance"]["disc_min_travel_mm"] or np.nan), "-o", ms=3, color=col,
                           label=f"a {a:g}")
            axs[3, j].plot(ok, g(lambda r: r["criteria"]["clearance"]["guard_min_travel_mm"] or np.nan), "--", color=col, lw=1)
            axs[4, j].plot(ok, g(lambda r: r["criteria"]["net_low"]["net_lowest_m"]), "-o", ms=3, color=col, label=f"a {a:g}")
            axs[5, j].plot(ok, g(lambda r: r["settle_s"] if r["settle_s"] is not None else 9.0), "-o", ms=3, color=col,
                           label=f"a {a:g}")
        ax = axs[0, j]
        ax.set_yticks(range(len(accels))); ax.set_yticklabels([f"{a:g}" for a in accels])
        ax.set_ylabel("a_max [m/s^2]")
        ax.set_title(f"heading: {h} ({TRAVEL_HEADINGS[h]:g} deg)  green o pass, red: first failing criterion")
        for k, mk in marker.items():
            ax.plot([], [], mk, color="tab:red", label=k)
        ax.legend(fontsize=7, ncol=4, loc="lower left")
        axs[1, j].axhline(50, color="k", lw=0.8, ls=":"); axs[1, j].axhline(100, color="k", lw=0.8, ls="--")
        axs[1, j].set_ylabel("peak position error [mm]\n(solid cruise <= 50, dashed accel/brake <= 100)")
        axs[1, j].set_ylim(0, 300)
        axs[2, j].axhline(0.9, color="k", lw=0.8, ls=":")
        axs[2, j].set_ylabel("peak motor thrust / limit (solid)\n0.5 + longest time > 90 % [s] (dotted)")
        axs[2, j].set_ylim(0.5, 1.02)
        axs[3, j].axhline(30, color="k", lw=0.8, ls=":"); axs[3, j].axhline(10, color="k", lw=0.8, ls="--")
        axs[3, j].set_ylabel("min clearance in the trip [mm]\n(solid rotor disc >= 30, dashed guard >= 10)")
        axs[3, j].set_ylim(0, 60)
        axs[4, j].axhline(0.3, color="k", lw=0.8, ls=":")
        axs[4, j].set_ylabel("lowest net point [m] (>= 0.3)")
        axs[5, j].axhline(5, color="k", lw=0.8, ls=":")
        axs[5, j].set_ylabel("time to settle within 10 mm [s]\n(9 = not settled in 8 s)")
        axs[5, j].set_xlabel("cruise speed [m/s]")
        for i in range(1, 6):
            axs[i, j].legend(fontsize=7, ncol=2)
        for i in range(6):
            axs[i, j].grid(alpha=0.3)
    fig.suptitle("Formation travel with the net: pass / fail and limiting metrics against cruise speed "
                 "(official model, net-pull + acceleration feedforward, quadratic net drag)")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(path, dpi=100)
    plt.close(fig)


def plot_travel_sensitivity(rows, path):
    heads = list(TRAVEL_HEADINGS)
    names = [r["variant"] for r in rows if r["heading"] == heads[0]]
    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.38
    for j, h in enumerate(heads):
        vals = [next((r["v_max_m_s"] or 0.0) for r in rows if r["variant"] == n and r["heading"] == h) for n in names]
        bars = ax.bar(np.arange(len(names)) + (j - 0.5) * w, vals, w, label=f"{h} ({TRAVEL_HEADINGS[h]:g} deg)")
        for b_, n in zip(bars, names):
            r = next(r for r in rows if r["variant"] == n and r["heading"] == h)
            ax.text(b_.get_x() + b_.get_width() / 2, b_.get_height(), f"{r['v_max_m_s']}\n{r['first_fail']}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20, fontsize=8)
    ax.set_ylabel("max cruise speed [m/s] (0.1 m/s resolution)")
    ax.set_title("Sensitivity of the speed limit (a_max 1 m/s^2); label: limit / first failing criterion above it")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_travel_sweep(base, out, jobs, accels=(0.5, 1.0, 2.0, 3.0), probe_accels=(4.0, 6.0, 8.0),
                     coarse=(0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0), variant_names=None):
    """The full travel test: speed limits for every heading and acceleration limit, an acceleration probe,
    and the sensitivity variants at a_max = 1 m/s^2; then the selected runs again with their logs."""
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    p0 = travel_params({**TRAVEL_DEFAULT, **base})
    d0 = thread_diameter(p0)
    variants = {
        "base": {},
        "thread_d_x0.5": {"thread_d_scale": 0.5},
        "thread_d_x2": {"thread_d_scale": 2.0},
        "net_drag_off": {"net_drag": False},
        "quadratic+linear": {"keep_linear": True},
        "acc_ff_off": {"acc_ff": 0.0},
        "acc_ff_incl_net": {"acc_ff": (p0.m_drone + p0.m_net / 4) / p0.m_drone},
        "net_10g": {"net_mass": 0.010},
        "net_10g_same_d": {"net_mass": 0.010, "thread_d": d0},
    }
    if variant_names is not None:
        variants = {k: v for k, v in variants.items() if k in variant_names}
    results = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(jobs) as pool:
        series = [{**TRAVEL_DEFAULT, **base, "heading": h, "accel": a, "variant": "base"}
                  for h in TRAVEL_HEADINGS for a in list(accels) + list(probe_accels)]
        series += [{**TRAVEL_DEFAULT, **base, **v, "heading": h, "accel": 1.0, "variant": name}
                   for name, v in variants.items() if name != "base" for h in TRAVEL_HEADINGS]
        print(f"travel sweep: {len(series)} series, {jobs} processes", flush=True)
        travel_limit_search(pool, series, results, list(coarse))
        limits = []
        for (var, h, a), res in sorted(results.items()):
            limits.append({"variant": var, "heading": h, "accel": a, **series_summary(res)})
        # selected runs with logs: fastest passing (over all accelerations of the base set) and the first
        # failure at that acceleration, per heading
        picks = []
        for h in TRAVEL_HEADINGS:
            cand = [r for r in limits if r["variant"] == "base" and r["heading"] == h and r["v_max_m_s"]]
            best = max(cand, key=lambda r: (r["v_max_m_s"], r["accel"]))
            picks.append(("fastest_pass", h, best["accel"], best["v_max_m_s"]))
            if best["v_first_fail_m_s"] is not None:
                picks.append(("first_fail", h, best["accel"], best["v_first_fail_m_s"]))
        saved = pool.starmap(run_travel, [({**TRAVEL_DEFAULT, **base, "heading": h, "accel": a, "speed": v,
                                            "variant": "base"}, out,
                                           f"travel_{h}_{kind}_v{v:g}_a{a:g}", False) for kind, h, a, v in picks])
    wall = time.time() - t0
    for (kind, h, a, v), m in zip(picks, saved):
        ref = results[("base", h, float(a))][round(float(v), 3)]
        if ref["pass"] != m["pass"] or ref["failed"] != m["failed"]:
            print(f"warning: re-run of {h} v={v} a={a} differs from the sweep run", flush=True)
    runs = [r for res in results.values() for r in res.values()]
    with open(os.path.join(out, "travel_sweep.json"), "w") as f:
        json.dump({"mode": "travel_sweep", "wall_time_s": round(wall, 1), "criteria_limits": TRAVEL_CRIT,
                   "criteria_order": CRIT_ORDER, "variants": variants, "thread_d_default_m": d0,
                   "base_cfg": {**TRAVEL_DEFAULT, **base}, "limits": limits,
                   "selected_runs": [{"kind": k, "heading": h, "accel": a, "speed": v, "files": m.get("files")}
                                     for (k, h, a, v), m in zip(picks, saved)],
                   "runs": runs}, f, indent=1)
    plot_travel_sweep(results, os.path.join(out, "travel_sweep.png"), list(accels) + list(probe_accels))
    plot_travel_sensitivity([r for r in limits if r["accel"] == 1.0], os.path.join(out, "travel_sensitivity.png"))
    print(f"travel sweep: {len(runs)} trips in {wall:.0f} s wall time")
    print(f"{'variant':>18} {'heading':>9} {'a_max':>5} {'v_max':>6} {'v_fail':>6}  first fail (all failing)   onset per criterion")
    for r in limits:
        print(f"{r['variant']:>18} {r['heading']:>9} {r['accel']:5.1f} {str(r['v_max_m_s']):>6} {str(r['v_first_fail_m_s']):>6}  "
              f"{str(r['first_fail']):>9} {r['failed_at_first_fail']}  {r['first_failing_speed_per_criterion']}")
    return limits


# ---------------------------------------------------------------- transfer: shortest A -> B move, shape only at B
# The formation leaves its settled hold at A and must be settled in shape at B, a distance L away; during the
# move only the safety list applies (TRANSFER_CRIT), not the tracking tolerance or the 1.25 m pattern.
TRANSFER_CRIT = {"tilt_deg": 60.0, "alt_err_m": 0.30, "motor_frac": 0.90, "motor_time_s": 0.10,
                 "disc_m": 0.030, "guard_m": 0.010, "net_low_m": 0.30, "separation_m": 0.50,
                 "slack_s": 0.05,                    # total time any corner string is slack in flight (not relied on)
                 "settle_pos_m": 0.010, "settle_sag_m": 0.005, "settle_net_off_m": 0.010, "settle_window_s": 2.0,
                 # sag and net offset are judged on a trailing 0.5 s mean: with the linear knot damping replaced by
                 # the quadratic drag, the net centre keeps a nearly undamped ~3 Hz bounce of a few mm (even at rest,
                 # excited by the drones' control noise); the instantaneous version is reported as well
                 "settle_avg_s": 0.5}
TRANSFER_SAFETY = ["control", "motor90", "clearance", "net_low", "separation", "slack"]
TRANSFER_DEFAULT = dict(L=10.0, heading="side", heading_deg=None, speed=2.0, a_up=1.0, a_down=None,
                        D_c=None, c_T=None, s_off=0.0, s_T=None, dz=0.0, t_settle=12.0, spacing=1.25,
                        acc_ff=1.0, ff_path=None, strategy="rigid", p_over=None, tag=None, keep_linear=False)


class TransferPlan(Plan):
    """Hover plan (climb, spread) + a straight move of length L along heading_deg from t_go.

    Speed: min-jerk ramp 0 -> v over T_u = 1.875 v / a_up, cruise, min-jerk ramp v -> 0 over
    T_b = 1.875 v / a_down (asymmetric allowed). If L is too short to reach v, v is lowered to the peak that
    exactly fits (no cruise), keeping both accelerations. Spacing and height ramps (contraction, climb and
    their reversal) come in through d_ramps / z_ramps. Attributes match TravelPlan (t_go, t_a1, t_d0, t_d1,
    u, v, phases) so the travel analysis helpers apply."""

    def __init__(self, z_ramps, d_ramps, t_go, L, heading_deg, v, a_up, a_down, t_settle, t_arm=0.5):
        k = 1.875
        v_fit = np.sqrt(2 * L / (k * (1 / a_up + 1 / a_down)))
        self.v_cmd, self.v = v, min(v, v_fit)
        self.L, self.a_up, self.a_down, self.heading_deg, self.t_go = L, a_up, a_down, heading_deg, t_go
        self.a_max = max(a_up, a_down)
        self.T_u, self.T_b = k * self.v / a_up, k * self.v / a_down
        self.T_a = self.T_u
        self.t_cruise = max(0.0, (L - 0.5 * self.v * (self.T_u + self.T_b)) / self.v)
        h = np.radians(heading_deg)
        self.u = np.array([np.cos(h), np.sin(h)])
        self.t_a1 = t_go + self.T_u
        self.t_d0 = self.t_a1 + self.t_cruise
        self.t_d1 = self.t_d0 + self.T_b
        self.t_settle = t_settle
        super().__init__(z_ramps, d_ramps, t_end=self.t_d1 + t_settle, t_arm=t_arm)

    def travel(self, t):
        v = self.v
        if t < self.t_a1:
            return minjerk_travel(t, self.t_go, self.T_u, v)
        s_a = 0.5 * v * self.T_u
        if t < self.t_d0:
            return s_a + v * (t - self.t_a1), v, 0.0
        s, sd, sdd = minjerk_travel(t, self.t_d0, self.T_b, v)
        tt = min(t - self.t_d0, self.T_b)
        return s_a + v * self.t_cruise + v * tt - s, v - sd, -sdd

    phases = TravelPlan.phases

    def ref(self, t):
        pos, vel, d = Plan.ref(self, t)
        s, sd, _ = self.travel(t)
        pos[:, :2] += s * self.u
        vel[:, :2] += sd * self.u
        return pos, vel, d

    def acc(self, t):
        a = np.zeros((4, 3))
        a[:, :2] = self.travel(t)[2] * self.u
        return a


def transfer_plan(p, cfg, heading_deg):
    """TransferPlan for cfg (TRANSFER_DEFAULT keys): contraction to D_c over c_T from t_go (default: the
    acceleration ramp), re-spread to the hold spacing starting s_off after the brake starts, over s_T (default:
    the brake ramp); dz: climb by dz together with the contraction and come back down with the re-spread."""
    t_arm, t_climb, t_spread0, t_spread1, t_go = 0.5, 4.5, 6.5, 12.5, 15.0
    a_down = cfg["a_down"] or cfg["a_up"]
    probe = TransferPlan([(t_arm, t_climb, 0.012, p.z_hover)], [(t_spread0, t_spread1, p.d_takeoff, cfg["spacing"])],
                         t_go, cfg["L"], heading_deg, cfg["speed"], cfg["a_up"], a_down, cfg["t_settle"], t_arm)
    z_ramps = [(t_arm, t_climb, 0.012, p.z_hover)]
    d_ramps = [(t_spread0, t_spread1, p.d_takeoff, cfg["spacing"])]
    if cfg["D_c"] is not None:
        c_T = cfg["c_T"] or max(probe.T_u, 0.5)
        s_T = cfg["s_T"] or max(probe.T_b, 0.5)
        s0 = max(probe.t_d0 + cfg["s_off"], t_go + c_T)            # never re-spread before the contraction ends
        d_ramps += [(t_go, t_go + c_T, cfg["spacing"], cfg["D_c"]), (s0, s0 + s_T, cfg["D_c"], cfg["spacing"])]
        if cfg["dz"]:
            z_ramps += [(t_go, t_go + c_T, p.z_hover, p.z_hover + cfg["dz"]),
                        (s0, s0 + s_T, p.z_hover + cfg["dz"], p.z_hover)]
        t_end_shape = s0 + s_T
    else:
        t_end_shape = probe.t_d1
    plan = TransferPlan(z_ramps, d_ramps, t_go, cfg["L"], heading_deg, cfg["speed"], cfg["a_up"], a_down,
                        max(cfg["t_settle"], t_end_shape - probe.t_d1 + cfg["t_settle"]), t_arm)
    plan.t_shape = t_end_shape                                       # reference back in shape (spacing, height)
    return plan, (t_arm, t_spread0, t_spread1)


def transfer_metrics(p, log, plan, cl, cl_series, t_arm):
    C = TRANSFER_CRIT
    t = log["t"]
    dt = float(np.median(np.diff(t)))
    u = np.array([*plan.u, 0.0])
    cr = np.array([-plan.u[1], plan.u[0], 0.0])
    pos, ref = log["pos"], log["ref"]
    trav = t >= plan.t_go
    fc = pos.mean(axis=1)
    rel = log["net_c"] - fc
    off = np.stack([rel @ u, rel @ cr, rel[:, 2]], 1)
    off_h = np.hypot(off[:, 0], off[:, 1])
    pre = (t >= plan.t_go - 2.0) & (t < plan.t_go)
    sag0 = float(log["sag_net"][pre].mean())
    off0 = off[pre].mean(0)
    dfin = np.linalg.norm(pos - ref[-1][None], axis=2).max(1)       # worst drone, distance to its final hold position
    n_w = int(round(C["settle_window_s"] / dt))
    n_a = max(1, int(round(C["settle_avg_s"] / dt)))

    def trailing_mean(x):
        c_ = np.cumsum(np.concatenate([[0.0], x]))
        idx = np.arange(len(x))
        lo = np.maximum(0, idx + 1 - n_a)
        return (c_[idx + 1] - c_[lo]) / (idx + 1 - lo)

    def first_window(ok_):
        run_len = np.zeros(len(t) + 1, int)                         # run_len[i] = consecutive ok samples from i on
        for i in range(len(t) - 1, -1, -1):
            run_len[i] = run_len[i + 1] + 1 if ok_[i] else 0
        good = np.flatnonzero(run_len[:len(t)] >= n_w)
        return int(good[0]) if len(good) else None

    sag_dev = np.abs(log["sag_net"] - sag0)
    ok_inst = trav & (dfin <= C["settle_pos_m"]) & (sag_dev <= C["settle_sag_m"]) & (off_h <= C["settle_net_off_m"])
    ok = (trav & (dfin <= C["settle_pos_m"]) & (np.abs(trailing_mean(log["sag_net"]) - sag0) <= C["settle_sag_m"])
          & (trailing_mean(off_h) <= C["settle_net_off_m"]))
    i_set = first_window(ok)
    i_inst = first_window(ok_inst)
    pre_hold = (t >= plan.t_go - 2.0) & (t < plan.t_go)
    transfer_s = round(float(t[i_set] - plan.t_go), 3) if i_set is not None else None
    t_win_end = float(t[i_set] + C["settle_window_s"]) if i_set is not None else float(t[-1])
    W = trav & (t <= t_win_end)
    first_t = lambda mk: (float(t[np.flatnonzero(mk)[0]]) if mk.any() else None)
    aborted = bool(np.isfinite(log.get("aborted_at", np.nan)))
    crit = {}
    zerr = np.abs(pos[:, :, 2] - ref[:, :, 2]).max(1)
    bad = W & ((log["tilt"].max(1) > C["tilt_deg"]) | (log["ncon"] > 0) | (zerr > C["alt_err_m"])
               | ~np.isfinite(pos).all(axis=(1, 2)))
    crit["control"] = {"ok": not bad.any() and not aborted, "t_first": first_t(bad),
                       "tilt_max_deg": round(float(log["tilt"][W].max()), 2),
                       "alt_err_max_mm": round(float(1e3 * zerr[W].max()), 1),
                       "floor_contacts_max": int(log["ncon"][W].max()), "aborted": aborted,
                       "horiz_err_max_mm": round(float(1e3 * np.linalg.norm((pos - ref)[W][:, :, :2], axis=2).max()), 1)}
    mf, mraw = log["motor"] / p.f_max, log["motor_raw"] / p.f_max
    hi = W & (mf > C["motor_frac"]).any(axis=(1, 2))
    a, b = _runs(hi)
    lg = float((b - a).max() * dt) if len(a) else 0.0
    sat = W & (mraw >= 1.0).any(axis=(1, 2))
    sa, sb = _runs(sat)
    crit["motor90"] = {"ok": lg <= C["motor_time_s"] + 1e-9, "longest_s": round(lg, 3),
                       "t_first": first_t(hi) if lg > C["motor_time_s"] else None,
                       "motor_max": round(float(mf[W].max()), 4), "above90_total_s": round(float(hi.sum() * dt), 3),
                       "saturated_total_s": round(float(sat.sum() * dt), 3),
                       "saturated_longest_s": round(float((sb - sa).max() * dt) if len(sa) else 0.0, 3)}
    ft = cl_series["t"]
    fm = (ft >= t_arm) & (ft <= t_win_end)
    dmin, gmin = cl_series["all"].min(1), np.nanmin(cl_series["guard"], axis=1)
    badc = fm & ((dmin < C["disc_m"]) | (gmin < C["guard_m"]))
    ftr = fm & (ft >= plan.t_go)
    crit["clearance"] = {"ok": not badc.any(), "t_first": float(ft[np.flatnonzero(badc)[0]]) if badc.any() else None,
                         "disc_min_mm": round(float(1e3 * dmin[fm].min()), 1),
                         "guard_min_mm": round(float(1e3 * gmin[fm].min()), 1),
                         "disc_min_move_mm": round(float(1e3 * dmin[ftr].min()), 1) if ftr.any() else None,
                         "guard_min_move_mm": round(float(1e3 * gmin[ftr].min()), 1) if ftr.any() else None}
    nl = W & (log["net_zmin"] <= C["net_low_m"])
    crit["net_low"] = {"ok": not nl.any(), "t_first": first_t(nl), "net_lowest_m": round(float(log["net_zmin"][W].min()), 4)}
    sep = np.min([np.linalg.norm(pos[:, i] - pos[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)], axis=0)
    bs = W & (sep < C["separation_m"])
    crit["separation"] = {"ok": not bs.any(), "t_first": first_t(bs), "min_m": round(float(sep[W].min()), 4)}
    slack = W[:, None] & (log["tension"] <= 0.0)
    sl_tot = slack.sum(0) * dt
    sl_long = []
    for k in range(4):
        a_, b_ = _runs(slack[:, k])
        sl_long.append(float((b_ - a_).max() * dt) if len(a_) else 0.0)
    crit["slack"] = {"ok": float(sl_tot.max()) <= C["slack_s"] + 1e-9, "t_first": first_t(slack.any(1)),
                     "slack_total_s": np.round(sl_tot, 3).tolist(), "slack_longest_s": np.round(sl_long, 3).tolist()}
    crit["settled"] = {"ok": transfer_s is not None, "t_first": None if transfer_s is not None else float(t[-1])}
    safety_failed = sorted((k for k in TRANSFER_SAFETY if not crit[k]["ok"]),
                           key=lambda k: (crit[k]["t_first"] if crit[k]["t_first"] is not None else np.inf))
    failed = safety_failed + ([] if transfer_s is not None else ["not_settled"])
    # the earlier strict tracking criteria, evaluated on the same run (strategy 1)
    e3 = np.linalg.norm(pos - ref, axis=2).max(1)
    ph = plan.phases()
    cru = (t >= ph["cruise"][0]) & (t < ph["cruise"][1])
    ramp = ((t >= ph["accelerate"][0]) & (t < ph["accelerate"][1])) | ((t >= ph["decelerate"][0]) & (t < ph["decelerate"][1]))
    strict = {"err3d_max_mm": round(float(1e3 * e3[trav & (t < ph["settle"][0])].max()), 1),
              "cruise_max_mm": round(float(1e3 * e3[cru].max()), 1) if cru.any() else None,
              "ramp_max_mm": round(float(1e3 * e3[ramp].max()), 1) if ramp.any() else None}
    strict["ok"] = bool((strict["err3d_max_mm"] <= 300) and (strict["cruise_max_mm"] or 0) <= 50
                        and (strict["ramp_max_mm"] or 0) <= 100 and not safety_failed and transfer_s is not None)
    # what the net does
    mv = trav & (t < plan.t_d1)
    arr = t >= plan.t_d1
    side = np.stack([np.linalg.norm(pos[:, a_, :2] - pos[:, (a_ + 1) % 4, :2], axis=1) for a_ in range(4)], 1)
    vf = log["vel"].mean(axis=1) @ u
    dist_set = None if i_set is None else float(dfin[i_set:].max())
    net = {"sag0_m": round(sag0, 4),
           "net_lowest_m": round(float(log["net_zmin"][W].min()), 4),
           "trail_max_mm": round(float(-1e3 * (off[mv, 0] - off0[0]).min()), 1) if mv.any() else None,
           "ahead_max_mm": round(float(1e3 * (off[W, 0] - off0[0]).max()), 1),
           "cross_max_mm": round(float(1e3 * np.abs(off[W, 1]).max()), 1),
           "vert_range_mm": [round(float(1e3 * (off[W, 2] - off0[2]).min()), 1), round(float(1e3 * (off[W, 2] - off0[2]).max()), 1)],
           "sag_range_m": [round(float(log["sag_net"][W].min()), 4), round(float(log["sag_net"][W].max()), 4)],
           "arrival_swing_along_mm": round(float(1e3 * np.abs(off[arr, 0] - off0[0]).max()), 1) if arr.any() else None,
           "arrival_sag_dev_mm": round(float(1e3 * np.abs(log["sag_net"][arr] - sag0).max()), 1) if arr.any() else None,
           "tension_gf_min": np.round(log["tension"][W].min(0) / G * 1e3, 2).tolist(),
           "tension_gf_max": np.round(log["tension"][W].max(0) / G * 1e3, 2).tolist(),
           "spacing_min_m": round(float(side[W].min()), 4), "spacing_max_m": round(float(side[W].max()), 4)}
    t_move = plan.t_d1 - plan.t_go
    return {"pass": not failed, "failed": failed, "first_fail": failed[0] if failed else None,
            "transfer_s": transfer_s,
            "transfer_s_instantaneous": round(float(t[i_inst] - plan.t_go), 3) if i_inst is not None else None,
            "sag_noise_at_rest_mm": {"std": round(float(1e3 * log["sag_net"][pre_hold].std()), 2),
                                     "peak_dev": round(float(1e3 * sag_dev[pre_hold].max()), 2)},
            "sag_noise_end_mm": {"std": round(float(1e3 * log["sag_net"][t >= t[-1] - 2.0].std()), 2),
                                 "peak_dev": round(float(1e3 * sag_dev[t >= t[-1] - 2.0].max()), 2)},
            "criteria": crit, "strict": strict, "net": net,
            "peak_speed_ref_m_s": round(float(plan.v), 3),
            "peak_speed_actual_m_s": round(float(vf[trav].max()), 3),
            "time_breakdown_s": {"accelerate": round(plan.T_u, 3), "cruise": round(plan.t_cruise, 3),
                                 "brake": round(plan.T_b, 3),
                                 "settle_after_ref_stop": None if transfer_s is None else round(transfer_s - t_move, 3),
                                 "ref_back_in_shape_after_stop": round(float(getattr(plan, "t_shape", plan.t_d1) - plan.t_d1), 3)},
            "tilt_max_deg": np.round(log["tilt"][W].max(0), 2).tolist(),
            "motor_max": np.round(mf[W].max(axis=(0, 2)), 3).tolist(),
            "settled_max_drone_offset_mm": None if dist_set is None else round(1e3 * dist_set, 2),
            "t_settled_abs_s": None if i_set is None else float(t[i_set]),
            "plan": {"t_go": plan.t_go, "t_accel_end": plan.t_a1, "t_brake_start": plan.t_d0, "t_ref_stop": plan.t_d1,
                     "t_shape": float(getattr(plan, "t_shape", plan.t_d1)), "v": plan.v, "v_cmd": plan.v_cmd}}


def run_transfer(cfg, out=None, tag=None, verbose=True):
    cfg = {**TRANSFER_DEFAULT, **cfg}
    p = travel_params({**TRAVEL_DEFAULT, "p_over": cfg["p_over"], "keep_linear": cfg["keep_linear"]})  # quadratic drag on
    hd = cfg["heading_deg"] if cfg["heading_deg"] is not None else TRAVEL_HEADINGS[cfg["heading"]]
    plan, (t_arm, t_spread0, t_spread1) = transfer_plan(p, cfg, hd)
    ff = load_ff(p, cfg["ff_path"], spacing=cfg["spacing"]) if cfg["ff_path"] else None
    sim = Sim(p)
    t0 = time.time()
    log = sim.run(plan, ff=ff, acc_ff=cfg["acc_ff"], extra_log=True, verbose=verbose, abort_err=5.0)
    wall = time.time() - t0
    phases = ([("on_ground", 0.0, t_arm), ("takeoff", t_arm, t_spread0), ("spread", t_spread0, t_spread1),
               ("pre_travel", t_spread1, plan.t_go)] + [(k, a, b) for k, (a, b) in plan.phases().items()])
    cl, cl_series = clearance_analysis(p, log, phases)
    met = transfer_metrics(p, log, plan, cl, cl_series, t_arm)
    met.update(cfg=cfg, heading_deg=hd, wall_s=round(wall, 1))
    if out:
        tag = tag or cfg["tag"] or f"transfer_{cfg['strategy']}_{cfg['heading']}_L{cfg['L']:g}"
        # three moments for the screenshots: peak speed (mid-transit), braking, and the arrival swing
        vf = log["vel"].mean(axis=1) @ np.array([*plan.u, 0.0])
        sm = (log["t"] >= plan.t_go) & (log["t"] < plan.t_d1)
        t_peak = float(log["t"][sm][np.argmax(vf[sm])]) if sm.any() else plan.t_go
        t_brake = 0.5 * (plan.t_d0 + plan.t_d1)
        offa = np.abs((log["net_c"] - log["pos"].mean(axis=1)) @ np.array([*plan.u, 0.0]))
        sa = log["t"] >= plan.t_d1
        t_swing = float(log["t"][sa][np.argmax(offa[sa])]) if sa.any() else plan.t_d1
        met["render_shots"] = [("transfer_01_peak_speed_wide", t_peak, "wide", f"peak speed {vf[sm].max():.2f} m/s"),
                               ("transfer_02_peak_speed_side", t_peak, "side", f"peak speed, side view along +y"),
                               ("transfer_03_braking_wide", t_brake, "wide", "braking"),
                               ("transfer_04_arrival_swing_side", t_swing, "side", "largest net swing after the stop")]
        for key in ("all", "net", "footprint", "taut", "guard"):
            log[f"clearance_{key}"] = cl_series[key]
        np.savez_compressed(os.path.join(out, f"{tag}_log.npz"), **log)
        with open(os.path.join(out, f"{tag}_summary.json"), "w") as f:
            json.dump({"mode": "transfer", **met, "criteria_limits": TRANSFER_CRIT, "clearance": cl,
                       "aero": aero_info(p), "params": asdict(p)}, f, indent=2)
        plot_transfer_timeseries(p, log, plan, cl_series, met, os.path.join(out, f"{tag}_timeseries.png"))
        met["files"] = [f"{tag}_summary.json", f"{tag}_log.npz", f"{tag}_timeseries.png"]
    return met


def print_transfer(met):
    c, cr = met["cfg"], met["criteria"]
    print(f"transfer {c['strategy']} {c['heading']} L={c['L']:g} m v={met['peak_speed_ref_m_s']} (actual peak "
          f"{met['peak_speed_actual_m_s']}) a_up={c['a_up']:g} a_down={c['a_down'] or c['a_up']:g} D_c={c['D_c']} dz={c['dz']}: "
          f"{'PASS' if met['pass'] else 'FAIL ' + ', '.join(met['failed'])}  transfer {met['transfer_s']} s  "
          f"breakdown {met['time_breakdown_s']}  strict {met['strict']}")
    print(f"  control {cr['control']}\n  motors {cr['motor90']}\n  clearance {cr['clearance']}\n"
          f"  net_low {cr['net_low']['net_lowest_m']} separation {cr['separation']['min_m']} slack {cr['slack']}\n"
          f"  net {met['net']}\n  tilt max {met['tilt_max_deg']} motor max {met['motor_max']}")


def plot_transfer_timeseries(p, log, plan, cl_series, met, path):
    t = log["t"]
    u = np.array([*plan.u, 0.0])
    cr = np.array([-plan.u[1], plan.u[0], 0.0])
    s = t >= plan.t_go - 2.0
    ts = t[s]
    pos, ref = log["pos"][s], log["ref"][s]
    fc = pos.mean(axis=1)
    C = TRANSFER_CRIT
    fig, axs = plt.subplots(5, 2, figsize=(14, 17), sharex=True)
    ax = axs[0, 0]
    for k in range(4):
        ax.plot(ts, pos[:, k] @ u, color=f"C{k}", lw=1, label=f"drone {k}")
    ax.plot(ts, ref.mean(axis=1) @ u, "k--", lw=0.8, label="reference centre")
    ax.set_ylabel("along-track position [m]"); ax.legend(fontsize=7, loc="upper left")
    ax2 = ax.twinx()
    ax2.plot(ts, log["vel"][s].mean(axis=1) @ u, color="tab:green", lw=1)
    ax2.plot(ts, log["ref_vel"][s].mean(axis=1) @ u, color="tab:green", lw=0.8, ls="--")
    ax2.set_ylabel("speed [m/s] (green)", color="tab:green")
    ax = axs[0, 1]
    for k in range(4):
        e = pos[:, k] - ref[:, k]
        ax.plot(ts, 1e3 * (e @ u), color=f"C{k}", lw=1, label=f"drone {k} along")
        ax.plot(ts, 1e3 * (e @ cr), color=f"C{k}", lw=0.7, ls="--")
        ax.plot(ts, 1e3 * e[:, 2], color=f"C{k}", lw=0.7, ls=":")
    ax.set_ylabel("drone - reference [mm]\n(solid along, dashed cross, dotted vertical)"); ax.legend(fontsize=7, ncol=2)
    ax = axs[1, 0]
    ax.plot(ts, log["tilt"][s]); ax.axhline(p.max_tilt_deg, color="k", lw=0.8, ls=":")
    ax.set_ylabel("tilt [deg] (dotted: controller tilt limit)"); ax.legend([f"drone {k}" for k in range(4)], fontsize=7)
    ax = axs[1, 1]
    ax.plot(ts, log["motor"][s].max(axis=2) / p.f_max); ax.axhline(C["motor_frac"], color="k", lw=0.8, ls=":")
    ax.set_ylabel("highest motor thrust / limit"); ax.set_ylim(0, 1.05)
    ax = axs[2, 0]
    ax.plot(ts, log["tension"][s] / G * 1e3); ax.set_ylabel("string tension [gf]")
    ax = axs[2, 1]
    r = log["net_c"][s] - fc
    ax.plot(ts, 1e3 * (r @ u), label="along (+ ahead)")
    ax.plot(ts, 1e3 * (r @ cr), label="cross")
    ax.plot(ts, 1e3 * (log["sag_net"][s] - met["net"]["sag0_m"]), label="sag - hold sag")
    ax.set_ylabel("net centre knot - formation centre [mm]\nand sag change [mm]"); ax.legend(fontsize=8)
    ax = axs[3, 0]
    ax.plot(ts, log["net_zmin"][s], label="lowest knot")
    ax.plot(ts, pos[:, :, 2].min(axis=1), label="lowest drone")
    ax.axhline(C["net_low_m"], color="k", lw=0.8, ls=":")
    ax.set_ylabel("height [m]"); ax.legend(fontsize=8)
    ax = axs[3, 1]
    side = np.stack([np.linalg.norm(pos[:, a, :2] - pos[:, (a + 1) % 4, :2], axis=1) for a in range(4)], 1)
    ax.plot(ts, side); ax.plot(ts, log["spacing_ref"][s], "k--", lw=0.8)
    sep = np.min([np.linalg.norm(pos[:, i] - pos[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)], axis=0)
    ax.plot(ts, sep, color="tab:gray", lw=0.8)
    ax.axhline(C["separation_m"], color="k", lw=0.8, ls=":")
    ax.legend(["side 0-1", "side 1-2", "side 2-3", "side 3-0", "reference spacing", "closest pair"], fontsize=7)
    ax.set_ylabel("spacing [m]")
    ax = axs[4, 0]
    fs = cl_series["t"] >= plan.t_go - 2.0
    for k in range(4):
        ax.plot(cl_series["t"][fs], 1e3 * cl_series["all"][fs, k], color=f"C{k}", lw=1)
        ax.plot(cl_series["t"][fs], 1e3 * cl_series["guard"][fs, k], color=f"C{k}", lw=0.7, ls=":")
    ax.axhline(1e3 * C["disc_m"], color="k", lw=0.8, ls="--"); ax.axhline(1e3 * C["guard_m"], color="tab:gray", lw=0.8, ls="-.")
    ax.set_ylim(0, 80); ax.set_ylabel("clearance [mm]\n(solid rotor disc, dotted guard)")
    ax = axs[4, 1]
    dfin = np.linalg.norm(pos - log["ref"][-1][None], axis=2).max(1)
    offh = np.hypot(r @ u, r @ cr)
    ax.semilogy(ts, np.maximum(1e3 * dfin, 1e-2), label="worst drone to final hold position")
    ax.semilogy(ts, np.maximum(1e3 * np.abs(log["sag_net"][s] - met["net"]["sag0_m"]), 1e-2), label="|sag - hold sag|")
    ax.semilogy(ts, np.maximum(1e3 * offh, 1e-2), label="net centre horizontal offset")
    for y in (10, 5):
        ax.axhline(y, color="k", lw=0.6, ls=":")
    if met["t_settled_abs_s"] is not None:
        ax.axvspan(met["t_settled_abs_s"], met["t_settled_abs_s"] + C["settle_window_s"], color="tab:green", alpha=0.2)
    ax.set_ylim(0.1, 5e3); ax.set_ylabel("settling [mm] (green: first 2 s in shape)"); ax.legend(fontsize=7)
    marks = [(plan.t_go, "go"), (plan.t_a1, "cruise"), (plan.t_d0, "brake"), (plan.t_d1, "stop")]
    if met["t_settled_abs_s"] is not None:
        marks.append((met["t_settled_abs_s"], "settled"))
    for a in axs.ravel():
        a.grid(alpha=0.3)
        for tm, _ in marks:
            a.axvline(tm, color="tab:red", lw=0.6, ls=":")
    for tm, lab in marks:
        axs[0, 0].text(tm, axs[0, 0].get_ylim()[1], lab, fontsize=8, color="tab:red", va="bottom", ha="center")
    for a in axs[-1]:
        a.set_xlabel("t [s]")
    c = met["cfg"]
    fig.suptitle(f"Transfer {c['strategy']}, {c['heading']}, L = {c['L']:g} m: transfer time {met['transfer_s']} s, "
                 f"peak {met['peak_speed_actual_m_s']} m/s, a_up {c['a_up']:g} a_down {c['a_down'] or c['a_up']:g}, "
                 f"D_c {c['D_c']} dz {c['dz']}  [{'PASS' if met['pass'] else 'FAIL: ' + ', '.join(met['failed'])}]")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _transfer_job(cfg):
    try:
        return run_transfer(cfg, verbose=False)
    except Exception as exc:
        return {"cfg": cfg, "pass": False, "failed": ["error"], "first_fail": "error", "transfer_s": None, "error": repr(exc)}


def _tkey(cfg):
    return tuple((k, cfg.get(k)) for k in ("L", "heading", "speed", "a_up", "a_down", "D_c", "c_T", "s_off", "s_T",
                                          "dz", "strategy", "p_over_tilt", "keep_linear"))


def strict_candidates(travel_sweep_path, L, heading, accels=(0.5, 1.0, 2.0, 3.0, 4.0), fracs=(1.0, 0.9, 0.8)):
    """Strategy 1 candidates from the travel sweep: for each acceleration limit the speed that minimises the
    time to cover L without exceeding that acceleration's strict speed limit, and two lower speeds."""
    with open(travel_sweep_path) as f:
        js = json.load(f)
    out = []
    for r in js["limits"]:
        if r["variant"] != "base" or r["heading"] != heading or r["accel"] not in accels or not r["v_max_m_s"]:
            continue
        a = r["accel"]
        v_star = min(r["v_max_m_s"], np.sqrt(L * a / 1.875))
        for fr in fracs:
            out.append({"speed": round(float(v_star * fr), 3), "a_up": a, "a_down": a, "strict_vmax": r["v_max_m_s"]})
    return out


def run_transfer_sweep(base, out, jobs, Ls=(2.0, 5.0, 10.0, 30.0), travel_sweep_path=None):
    """Shortest transfer time for every L and heading: strategy 1 (strict, from the travel sweep), 2 (rigid
    pattern, free speed / acceleration / braking), 3 (contraction in transit, optional climb), then a tilt
    limit of 45 deg at L = 10 m as a sensitivity. Coarse grid, then local refinement around the best runs."""
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    travel_sweep_path = travel_sweep_path or os.path.join(HERE, "out_official", "travel", "travel_sweep.json")
    heads = list(TRAVEL_HEADINGS)
    res = {}
    t0 = time.time()

    def mk(**kw):
        c = {**TRANSFER_DEFAULT, **base, **kw}
        if c["a_down"] is None:
            c["a_down"] = c["a_up"]
        c["p_over_tilt"] = (c["p_over"] or {}).get("max_tilt_deg")
        return c

    def batch(pool, cfgs, label):
        todo = []
        for c in cfgs:
            if _tkey(c) not in res and _tkey(c) not in {_tkey(x) for x in todo}:
                todo.append(c)
        print(f"  {label}: {len(todo)} runs ...", flush=True)
        t1 = time.time()
        for i, r in enumerate(pool.imap_unordered(_transfer_job, todo)):
            res[_tkey(r["cfg"])] = r
            if (i + 1) % 40 == 0 or i + 1 == len(todo):
                print(f"    {i + 1}/{len(todo)} ({time.time() - t1:.0f} s)", flush=True)

    def best(L, h, strategy, tilt=None, n=1, strict=False):
        strategies = strategy if isinstance(strategy, tuple) else (strategy,)
        rs = [r for r in res.values() if r["cfg"]["L"] == L and r["cfg"]["heading"] == h
              and r["cfg"]["strategy"] in strategies and r["cfg"].get("p_over_tilt") == tilt
              and r.get("transfer_s") is not None and (r["strict"]["ok"] if strict else r["pass"])]
        rs.sort(key=lambda r: (r["transfer_s"], -r["peak_speed_ref_m_s"]))
        return rs[:n]

    def v_grid(L, a_u, a_d, vs):
        v_fit = np.sqrt(2 * L / (1.875 * (1 / a_u + 1 / a_d)))
        g = sorted({round(min(v, v_fit), 2) for v in vs})
        return g

    def rigid_neighbours(c, tilt_over=None):
        cs = []
        for fu in (0.67, 1.0, 1.5):
            for fd in (0.67, 1.0, 1.5):
                for dv in (-0.5, 0.0, 0.5):
                    a_u, a_d = round(c["a_up"] * fu, 2), round(c["a_down"] * fd, 2)
                    v = round(c["speed"] + dv, 2)
                    if v > 0.2:
                        for vv in v_grid(c["L"], a_u, a_d, [v]):
                            cs.append(mk(L=c["L"], heading=c["heading"], speed=vv, a_up=a_u, a_down=a_d,
                                         strategy="rigid", p_over=tilt_over))
        return cs

    def reshape_set(c, tilt_over=None):
        cs = []
        # timings of the re-spread: during the braking ramp; finished before braking starts (or as soon as the
        # contraction ends); after the stop over 2 s; from the brake start over the brake ramp + 2 s
        for D_c, dz in ((1.1, 0.0), (1.0, 0.0), (0.9, 0.3)):
            for s_off, s_T in ((0.0, None), (-2.0, 2.0), (None, 2.0), (0.0, "plus2")):
                for dv, fu in ((0.0, 1.0), (1.0, 1.0), (0.0, 1.5)):
                    a_u = round(c["a_up"] * fu, 2)
                    for v in v_grid(c["L"], a_u, c["a_down"], [round(c["speed"] + dv, 2)]):
                        T_b = 1.875 * v / c["a_down"]
                        so = T_b if s_off is None else s_off                 # None: re-spread after the stop
                        sT = (T_b + 2.0) if s_T == "plus2" else s_T
                        cs.append(mk(L=c["L"], heading=c["heading"], speed=v, a_up=a_u, a_down=c["a_down"], D_c=D_c,
                                     dz=dz, s_off=round(so, 3), s_T=sT, strategy="reshape", p_over=tilt_over))
        return cs

    with mp.get_context("spawn").Pool(jobs) as pool:
        # stage 1: strategy 1 candidates and strategy 2 coarse grid
        cfgs = []
        for L in Ls:
            for h in heads:
                for sc in strict_candidates(travel_sweep_path, L, h):
                    cfgs.append(mk(L=L, heading=h, speed=sc["speed"], a_up=sc["a_up"], a_down=sc["a_down"], strategy="strict"))
                for a in (1.0, 2.0, 3.0, 4.0, 6.0):
                    for v in v_grid(L, a, a, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0]):
                        cfgs.append(mk(L=L, heading=h, speed=v, a_up=a, a_down=a, strategy="rigid"))
        batch(pool, cfgs, "stage 1 (strict candidates, rigid coarse)")
        # stage 2: rigid refinement around the best two, reshaping around the best rigid run
        cfgs = []
        for L in Ls:
            for h in heads:
                for b in best(L, h, ("rigid", "strict"), n=1):
                    cfgs += rigid_neighbours(b["cfg"])
        batch(pool, cfgs, "stage 2a (rigid refinement)")
        cfgs = []
        for L in Ls:
            for h in heads:
                b = best(L, h, ("rigid", "strict"))
                if b:
                    cfgs += reshape_set(b[0]["cfg"])
        batch(pool, cfgs, "stage 2b (reshaping around the best rigid run)")
        # stage 3: refine around the best reshaped run and the best rigid run once more
        cfgs = []
        for L in Ls:
            for h in heads:
                for b in best(L, h, "reshape", n=1):
                    c = b["cfg"]
                    for dD in (-0.05, 0.0, 0.05):
                        for fu, fd in ((1.0, 1.0), (1.25, 1.0), (1.0, 1.25)):
                            for dv in (0.0, 0.5):
                                a_u, a_d = round(c["a_up"] * fu, 2), round(c["a_down"] * fd, 2)
                                D_c = round(min(1.2, max(0.9, c["D_c"] + dD)), 3)
                                for v in v_grid(L, a_u, a_d, [round(c["speed"] + dv, 2)]):
                                    cfgs.append(mk(L=L, heading=h, speed=v, a_up=a_u, a_down=a_d, D_c=D_c, dz=c["dz"],
                                                   s_off=c["s_off"], s_T=c["s_T"], strategy="reshape"))
                for b in best(L, h, ("rigid", "strict"), n=1):
                    cfgs += rigid_neighbours(b["cfg"])
        batch(pool, cfgs, "stage 3 (refinement)")
        # stage 4: tilt limit 45 deg at L = 10 m (sensitivity): rigid and reshaped neighbourhoods of the best runs
        cfgs = []
        tilt = {"max_tilt_deg": 45.0}
        for h in heads:
            for strat in ("rigid", "reshape"):
                for b in best(10.0, h, strat, n=1):
                    c = dict(b["cfg"])
                    c["p_over"] = tilt
                    cfgs.append(mk(**{k: c[k] for k in TRANSFER_DEFAULT if k in c}))
                    if strat == "rigid":
                        cfgs += rigid_neighbours(c, tilt)
                    else:
                        cfgs += reshape_set(c, tilt)[:30]
        batch(pool, cfgs, "stage 4 (tilt limit 45 deg, L = 10 m)")
        # selected runs with logs: best overall strategy at L = 5 and 30 m, per heading
        picks = []
        for L in (5.0, 30.0):
            for h in heads:
                cand = best(L, h, "rigid") + best(L, h, "reshape")
                if cand:
                    b = min(cand, key=lambda r: r["transfer_s"])
                    picks.append(b["cfg"])
        saved = pool.starmap(run_transfer, [(c, out, f"transfer_best_{c['heading']}_L{c['L']:g}_{c['strategy']}", False)
                                            for c in picks])
    wall = time.time() - t0
    table = []
    for L in Ls:
        for h in heads:
            row = {"L_m": L, "heading": h}
            # strategy 2 may also use the strict runs (they satisfy the safety list too)
            for strat, key, strict in (("strict", "strategy1_strict", True), (("rigid", "strict"), "strategy2_rigid", False),
                                       ("reshape", "strategy3_reshape", False)):
                b = best(L, h, strat, strict=strict)
                row[key] = None if not b else {
                    "transfer_s": b[0]["transfer_s"], "peak_speed_ref_m_s": b[0]["peak_speed_ref_m_s"],
                    "peak_speed_actual_m_s": b[0]["peak_speed_actual_m_s"],
                    "params": {k: b[0]["cfg"][k] for k in ("speed", "a_up", "a_down", "D_c", "dz", "s_off", "s_T")},
                    "time_breakdown_s": b[0]["time_breakdown_s"], "net": b[0]["net"],
                    "binding": transfer_binding(res, b[0])}
            table.append(row)
    t45 = []
    for h in heads:
        for strat in ("rigid", "reshape"):
            b35, b45 = best(10.0, h, strat), best(10.0, h, strat, tilt=45.0)
            t45.append({"heading": h, "strategy": strat, "tilt35_s": b35[0]["transfer_s"] if b35 else None,
                        "tilt45_s": b45[0]["transfer_s"] if b45 else None,
                        "tilt45_params": None if not b45 else {k: b45[0]["cfg"][k] for k in ("speed", "a_up", "a_down", "D_c", "dz")},
                        "tilt45_peak_m_s": b45[0]["peak_speed_actual_m_s"] if b45 else None})
    runs = list(res.values())
    with open(os.path.join(out, "transfer_sweep.json"), "w") as f:
        json.dump({"mode": "transfer_sweep", "wall_time_s": round(wall, 1), "criteria_limits": TRANSFER_CRIT,
                   "table": table, "tilt45_L10": t45,
                   "selected_runs": [{"cfg": c, "files": m.get("files"), "transfer_s": m["transfer_s"]} for c, m in zip(picks, saved)],
                   "runs": runs}, f, indent=1)
    plot_transfer_summary(table, runs, os.path.join(out, "transfer_summary.png"))
    print(f"transfer sweep: {len(runs)} runs in {wall:.0f} s wall time")
    for row in table:
        print(f"L={row['L_m']:4.0f} {row['heading']:>8}: " + "  ".join(
            f"{k.split('_')[0]}={'-' if row[k] is None else str(row[k]['transfer_s']) + 's/' + str(row[k]['peak_speed_actual_m_s']) + 'm/s ' + str(row[k]['params']) + ' bind=' + str(row[k]['binding'])}"
            for k in ("strategy1_strict", "strategy2_rigid", "strategy3_reshape")))
    print("tilt 45 deg at L = 10 m:", t45)
    return table


def transfer_binding(res, b):
    """What stops the best run from being faster: the failures of its faster neighbours (same strategy, L,
    heading, tilt): the runs with a higher peak speed or larger accelerations that failed, by criterion."""
    c = b["cfg"]
    same = [r for r in res.values() if all(r["cfg"].get(k) == c.get(k) for k in ("L", "heading", "strategy", "p_over_tilt"))]
    faster = [r for r in same if (r["cfg"]["speed"] > c["speed"] + 1e-9 or r["cfg"]["a_up"] > c["a_up"] + 1e-9
                                   or r["cfg"]["a_down"] > c["a_down"] + 1e-9)]
    count = {}
    for r in faster:
        for k in (r["failed"] if not r["pass"] else ["passed_but_slower"]):
            count[k] = count.get(k, 0) + 1
        if c["strategy"] == "strict" and r["pass"] and not r["strict"]["ok"]:
            count["strict_tracking"] = count.get("strict_tracking", 0) + 1
    return dict(sorted(count.items(), key=lambda kv: -kv[1]))


def plot_transfer_summary(table, runs, path):
    heads = list(TRAVEL_HEADINGS)
    Ls = sorted({r["L_m"] for r in table})
    fig, axs = plt.subplots(3, 2, figsize=(14, 13))
    names = [("strategy1_strict", "1 strict (shape kept)"), ("strategy2_rigid", "2 rigid, relaxed tracking"),
             ("strategy3_reshape", "3 reshaped in transit")]
    w = 0.26
    for j, h in enumerate(heads):
        rows = [r for r in table if r["heading"] == h]
        for i, (key, lab) in enumerate(names):
            tt = [r[key]["transfer_s"] if r[key] else np.nan for r in rows]
            vv = [r[key]["peak_speed_actual_m_s"] if r[key] else np.nan for r in rows]
            x = np.arange(len(rows)) + (i - 1) * w
            bars = axs[0, j].bar(x, tt, w, label=lab, color=f"C{i}")
            for b_, v_ in zip(bars, tt):
                if np.isfinite(v_):
                    axs[0, j].text(b_.get_x() + w / 2, v_, f"{v_:.1f}", ha="center", va="bottom", fontsize=7)
            axs[1, j].bar(x, vv, w, label=lab, color=f"C{i}")
            # time breakdown (stacked) for each strategy
            parts = ("accelerate", "cruise", "brake", "settle_after_ref_stop")
            bottom = np.zeros(len(rows))
            for pi_, part in enumerate(parts):
                val = np.array([(r[key]["time_breakdown_s"][part] or 0.0) if r[key] else 0.0 for r in rows])
                axs[2, j].bar(x, val, w, bottom=bottom, color=f"C{i}", alpha=0.35 + 0.65 * (pi_ + 1) / len(parts),
                              edgecolor="k", lw=0.3)
                bottom += val
        for a in axs[:, j]:
            a.set_xticks(range(len(rows))); a.set_xticklabels([f"L = {r['L_m']:g} m" for r in rows])
            a.grid(alpha=0.3, axis="y")
        axs[0, j].set_ylabel("transfer time [s] (leave A -> settled in shape at B)")
        axs[0, j].set_title(f"heading {h} ({TRAVEL_HEADINGS[h]:g} deg)"); axs[0, j].legend(fontsize=8)
        axs[1, j].set_ylabel("peak speed reached [m/s]")
        axs[2, j].set_ylabel("time breakdown [s]: accelerate / cruise /\nbrake / settle (darker = later)")
    fig.suptitle("Transfer A -> B: shortest time with the net shape required only at B")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- missions: random net-centre targets in a 10 m cube
# Box: x, y in [0, 10] m, z in [0, 10] m (floor at z = 0), open (no walls or ceiling in the model); leaving it is
# a failure. Simulation coordinates are box coordinates minus BOX_OFFSET: the formation takes off around the
# middle of the floor exactly as in the hold scenario (sim origin = box (5, 5, 0)).
BOX_SIZE = 10.0
BOX_OFFSET = np.array([5.0, 5.0, 0.0])
DRONE_R = 0.0844                    # m, airframe (guard) extent from the body axis (mesh: 84.4 mm)
DRONE_HALF_H = 0.0155               # m, airframe half height (guards +-15.4 mm)
MISSION_CRIT = {**TRANSFER_CRIT, "knot_below_disc_m": 0.05, "leg_timeout_s": 20.0,
                "target_margin_lat_m": 0.5, "target_net_floor_m": 0.5, "target_ceiling_m": 0.3}
# Settle (arrival) tolerances at each target: largest drone distance from its final hold position, net-centre
# horizontal offset from the formation centre (trailing 0.5 s mean) and net sag change from its hold value
# (trailing 0.5 s mean), all held for 2 s. The first mission study used the transfer-mode values (10 / 10 / 5 mm,
# MISSION_TOL_STRICT); the default is now sized to the net (1 m sides, ~317 mm sag): 50 / 50 / 20 mm.
MISSION_TOL_STRICT = {"pos_m": 0.010, "net_off_m": 0.010, "sag_m": 0.005}
MISSION_TOL = {"pos_m": 0.050, "net_off_m": 0.050, "sag_m": 0.020}
# Failing criteria. net_below_rotors (interior knots >= 50 mm below the lowest rotor hub) is recorded but no
# longer fails a leg: rotor and guard contact are covered by the 30 mm disc / 10 mm guard clearance checks.
# The first study (out_official/missions) also failed on it: MISSION_SAFETY_ROTOR (cfg rotor_check=True).
MISSION_SAFETY = ["control", "motor90", "clearance", "net_low", "separation", "slack", "box"]
MISSION_SAFETY_ROTOR = MISSION_SAFETY + ["net_below_rotors"]


def hold_drop(default=0.401):
    """Drone (mean) height above the net centre in the reviewed hold (out_official/hold_ff_summary.json)."""
    try:
        with open(os.path.join(HERE, "out_official", "hold_ff_summary.json")) as f:
            return float(json.load(f)["hold"]["sag_below_drones_m"])
    except (OSError, KeyError, ValueError):
        return default


def ramp_profile(t, t0, v, Tu, Tc, Tb):
    """Distance, speed, acceleration along a leg: min-jerk speed ramp 0 -> v over Tu, cruise Tc, ramp v -> 0 over Tb."""
    if t < t0 + Tu:
        return minjerk_travel(t, t0, Tu, v)
    s_a = 0.5 * v * Tu
    if t < t0 + Tu + Tc:
        return s_a + v * (t - t0 - Tu), v, 0.0
    t1 = t0 + Tu + Tc
    s, sd, sdd = minjerk_travel(t, t1, Tb, v)
    tt = min(t - t1, Tb)
    return s_a + v * Tc + v * tt - s, v - sd, -sdd


def fit_profile(L, v, a_up, a_down):
    """Durations of the min-jerk profile; v is lowered to the peak that fits L if needed (no cruise)."""
    k = 1.875
    v = min(v, np.sqrt(2 * L / (k * (1 / a_up + 1 / a_down)))) if L > 0 else 0.0
    if v <= 0:
        return 0.0, 0.0, 0.0, 0.0
    Tu, Tb = k * v / a_up, k * v / a_down
    Tc = max(0.0, (L - 0.5 * v * (Tu + Tb)) / v)
    return v, Tu, Tc, Tb


def mission_targets(seed, n=10, h_dn=None, spacing=1.25):
    """n net-centre targets (box coordinates), uniform in the volume where the system at rest in its hold shape
    keeps 0.5 m from the side walls (drone guards included), the net centre (lowest point) >= 0.5 m above the
    floor and the drone tops >= 0.3 m below z = 10."""
    C = MISSION_CRIT
    h_dn = hold_drop() if h_dn is None else h_dn
    lat = spacing / 2 + DRONE_R + C["target_margin_lat_m"]
    lo = np.array([lat, lat, C["target_net_floor_m"]])
    hi = np.array([BOX_SIZE - lat, BOX_SIZE - lat, BOX_SIZE - C["target_ceiling_m"] - DRONE_HALF_H - h_dn])
    rng = np.random.default_rng(seed)
    return lo + rng.random((n, 3)) * (hi - lo), (lo, hi)


class MissionPlan(Plan):
    """Takeoff and spread as in the hold scenario, then straight legs between net-centre targets.

    The formation (rigid 1.25 m pattern, yaw 45 deg) moves as a whole: its reference centre goes along each leg
    vector with the min-jerk profile returned by planner(leg_vector) = dict(v, a_up, a_down). A leg departs at
    t_go0 (first leg) or as soon as the previous one is settled: all drones within 10 mm of their final hold
    positions, the net sag within 5 mm of its hold value and the net centre's horizontal offset from the
    formation centre within 10 mm, continuously for 2 s (settle_mode "avg": sag and offset on a trailing 0.5 s
    mean as in transfer mode; "inst": instantaneous; "pos": drone positions only). The leg time ends where that
    2 s window begins; the next leg leaves at its end. A leg not settled leg_timeout after its reference stops is
    recorded as not settled and the next one leaves anyway. observe() is called by Sim.run at 100 Hz.
    tol: settle tolerances (MISSION_TOL keys; the numbers above are MISSION_TOL_STRICT, the default is MISSION_TOL).
    final_strict_s > 0 with tolerances looser than MISSION_TOL_STRICT: after the last target is settled, keep
    hovering up to final_strict_s and record when the strict rule (avg mode) would be met as well."""

    def __init__(self, p, targets_sim, planner, h_dn, settle_mode="avg", t_go0=15.0, spacing=1.25,
                 leg_timeout=None, t_arm=0.5, tol=None, final_strict_s=0.0):
        self.tol = dict(MISSION_TOL if tol is None else tol)
        self.final_strict_s = final_strict_s if any(self.tol[k] > MISSION_TOL_STRICT[k] + 1e-12
                                                    for k in MISSION_TOL_STRICT) else 0.0
        self.ok_run_s = 0
        z_ramps = [(t_arm, 4.5, 0.012, p.z_hover)]
        d_ramps = [(6.5, 12.5, p.d_takeoff, spacing)]
        self.C0 = np.array([0.0, 0.0, p.z_hover])
        self.Cs = [self.C0] + [np.asarray(T, float) + np.array([0.0, 0.0, h_dn]) for T in targets_sim]
        self.pattern = np.array([[sx * spacing / 2, sy * spacing / 2, 0.0] for sx, sy in CORNERS])
        self.planner, self.settle_mode, self.t_go0 = planner, settle_mode, t_go0
        self.leg_timeout = MISSION_CRIT["leg_timeout_s"] if leg_timeout is None else leg_timeout
        self.legs, self.k, self.done = [], -1, False
        self.sag_buf, self.sag0 = [], None
        self.n_w = int(round(MISSION_CRIT["settle_window_s"] / 0.01))
        self.n_a = int(round(MISSION_CRIT["settle_avg_s"] / 0.01))
        super().__init__(z_ramps, d_ramps, t_end=t_go0 + 90.0 * len(targets_sim) + 30.0, t_arm=t_arm)

    def _depart(self, t):
        self.k += 1
        if self.k >= len(self.Cs) - 1:
            self.done = True
            return
        d = self.Cs[self.k + 1] - self.Cs[self.k]
        L = float(np.linalg.norm(d))
        prof = self.planner(d)
        v, Tu, Tc, Tb = fit_profile(L, prof["v"], prof["a_up"], prof["a_down"])
        self.legs.append({"leg": self.k, "t_dep": t, "from_sim": self.Cs[self.k].tolist(), "to_sim": self.Cs[self.k + 1].tolist(),
                          "L": L, "u": (d / L).tolist() if L > 0 else [0.0, 0.0, 0.0], "profile": prof, "v": v,
                          "Tu": Tu, "Tc": Tc, "Tb": Tb, "t_ref_stop": t + Tu + Tc + Tb,
                          "t_settled": None, "settled": None, "t_next_dep": None, "t_strict_settled": None})
        self.ok_run, self.buf_sag, self.buf_off = 0, [], []

    def _leg_state(self, t):
        if self.k < 0 or not self.legs:
            return np.zeros(3), np.zeros(3), np.zeros(3)
        lg = self.legs[-1]
        u = np.array(lg["u"])
        s, sd, sdd = ramp_profile(t, lg["t_dep"], lg["v"], lg["Tu"], lg["Tc"], lg["Tb"])
        return self.Cs[lg["leg"]] - self.C0 + s * u, sd * u, sdd * u

    def ref(self, t):
        pos, vel, d = Plan.ref(self, t)
        off, voff, _ = self._leg_state(t)
        return pos + off, vel + voff, d

    def acc(self, t):
        a = np.zeros((4, 3))
        a[:] = self._leg_state(t)[2]
        return a

    def observe(self, t, P, net_c, sag):
        if self.done:
            return
        if self.k < 0:
            if t >= self.t_go0 - 2.0:
                self.sag_buf.append(sag)
            if t >= self.t_go0 - 1e-9:
                self.sag0 = float(np.mean(self.sag_buf))
                self._depart(t)
            return
        lg = self.legs[-1]
        fin = self.Cs[lg["leg"] + 1] + self.pattern
        dfin = float(np.linalg.norm(P - fin, axis=1).max())
        fc = P.mean(axis=0)
        offh = float(np.hypot(net_c[0] - fc[0], net_c[1] - fc[1]))
        self.buf_sag = (self.buf_sag + [sag])[-self.n_a:]
        self.buf_off = (self.buf_off + [offh])[-self.n_a:]
        C = self.tol
        if self.settle_mode == "pos":
            ok = dfin <= C["pos_m"]
        elif self.settle_mode == "inst":
            ok = dfin <= C["pos_m"] and abs(sag - self.sag0) <= C["sag_m"] and offh <= C["net_off_m"]
        else:
            ok = (dfin <= C["pos_m"] and abs(np.mean(self.buf_sag) - self.sag0) <= C["sag_m"]
                  and np.mean(self.buf_off) <= C["net_off_m"])
        # the 2 s window starts at the earliest when the reference has stopped (with the looser tolerances the
        # formation can be within them during the last centimetres of the braking ramp; the 10 mm rule never was)
        ok = ok and t >= lg["t_ref_stop"] - 1e-9
        if self.final_strict_s > 0 and self.k == len(self.Cs) - 2:
            # last leg: also follow the strict rule (avg mode) until it holds for 2 s or final_strict_s runs out
            S = MISSION_TOL_STRICT
            ok_s = (dfin <= S["pos_m"] and abs(np.mean(self.buf_sag) - self.sag0) <= S["sag_m"]
                    and np.mean(self.buf_off) <= S["net_off_m"] and t >= lg["t_ref_stop"] - 1e-9)
            self.ok_run_s = self.ok_run_s + 1 if ok_s else 0
            if lg["settled"]:
                if self.ok_run_s >= self.n_w:
                    lg["t_strict_settled"] = round(t - (self.n_w - 1) * 0.01, 4)
                    self.done = True
                elif t - lg["t_settled"] > self.final_strict_s:
                    lg["t_strict_settled"] = None
                    self.done = True
                return
        self.ok_run = self.ok_run + 1 if ok else 0
        if self.ok_run >= self.n_w and self.final_strict_s > 0 and self.k == len(self.Cs) - 2:
            lg["t_settled"], lg["settled"], lg["t_next_dep"] = round(t - (self.n_w - 1) * 0.01, 4), True, t
            if self.ok_run_s >= self.n_w:          # cannot happen before the looser rule, kept for safety
                lg["t_strict_settled"] = round(t - (self.n_w - 1) * 0.01, 4)
                self.done = True
        elif self.ok_run >= self.n_w:
            lg["t_settled"], lg["settled"], lg["t_next_dep"] = round(t - (self.n_w - 1) * 0.01, 4), True, t
            self._depart(t)
        elif t - lg["t_ref_stop"] > self.leg_timeout:
            lg["settled"], lg["t_next_dep"] = False, t
            self._depart(t)


# ---------------------------------------------------------------- leg planners
# Relaxed planner ("strategy 2": rigid pattern, no tracking tolerance in transit), one rule for any leg vector.
# Horizontal numbers from the transfer search (out_official/transfer): v 3 m/s, 3 m/s^2 up and down is within
# 0.1-0.3 s of the best rigid profile found for L = 2, 5 and 10 m on both headings; braking is softened for
# horizontal runs beyond 10 m (at L = 30 m, 0.9-1.3 m/s^2 braking settled 1.4 s sooner than 3 m/s^2).
# Vertical numbers from the climb / descent characterisation (--mode mission --characterise); the descent speed
# is capped at 2.0 m/s (no rotor-wake physics in the model: vortex-ring onset is not represented).
# Each component of the leg keeps its own limits: v = min(v_h / |cos(el)|, v_v / |sin(el)|), same for the
# accelerations, times the oblique derating; `factor` scales speed and both accelerations.
# Limits under the current settle tolerances (50 / 50 / 20 mm; characterisation in out_official/missions_tol50):
# - climb: 2 m/s^2 is the quickest (5 m: 4.34 s, motor 0.79; 3 m/s^2 moves faster but then needs 1.2-1.5 s to
#   settle, 4.71 s; 4 m/s^2 breaks the 90 % motor rule). 8 m climbs at 2 m/s^2 reach 2.9 m/s (5.61 s, motor 0.82),
#   so the speed limit is 3 m/s. (First study, 10 mm rules: 1 m/s^2 / 2 m/s, because the 10-21 mm altitude
#   overshoot after faster climbs took up to 9 s to settle within 10 mm.)
# - descent: speed capped at 2.0 m/s (no rotor-wake physics in the model). At that speed 2, 3 and 4 m/s^2 give
#   the same 5 m descent time (4.45 / 4.49 / 4.39 s); 3 m/s^2 is 0.37 s quicker over 8 m but brings the net to
#   8 mm of the rotor-hub line (2 m/s^2: 31 mm); in the 100 study legs 3 m/s^2 gained 0.004-0.006 m/s. 2 m/s^2.
# - horizontal: unchanged (5 m level legs: 3 m/s^2 5.09-5.13 s; 4 m/s^2 4.86-5.07 s with the rotor-disc
#   clearance down to 30.5 mm on the diagonal; 6 m/s^2 fails the clearance).
# - oblique legs: component limits are combined independently, so a climbing oblique asks for the horizontal and
#   the climb acceleration together: every failure of the first trial factor sweep (13 legs at factor 1.0-1.2,
#   28-50 deg climbs) was the 90 % motor rule. Climbing obliques are derated by 0.8 (all 100 legs then pass up to
#   factor 1.2). Descending obliques keep their 0.75 derating: without it they pass at factor 1 but a -30 deg leg
#   reaches 43.8 deg tilt and the motor rule at 1.5, with it they pass up to 1.5 at the same leg time.
PLANNER_RELAXED = {"v_h": 3.0, "a_h": 3.0, "a_h_brake_long": 2.0, "long_h_m": 10.0,
                   "v_climb": 3.0, "a_climb": 2.0, "v_descent": 2.5, "a_descent": 2.0, "descent_cap": 2.0,
                   "descent_oblique_derate": 0.75, "climb_oblique_derate": 0.8}
# The rule of the first study (10 mm settle rules, net_below_rotors failing; out_official/missions).
# Climb: 4 m/s^2 breaks the 90 % motor rule, 3 m/s^2 peaks at 0.86, but climbs faster than ~1 m/s^2 settle
# slowly (up to 9 s): the quickest 5 m climb was 2 m/s at 1 m/s^2 (7.4 s). Descent: strings never went slack;
# the net rising toward the rotors binds at ~2.8 m/s or 4 m/s^2 (model limit ~2.5 m/s); 2 m/s at 3 m/s^2 was
# the quickest 5 m descent (5.55 s).
PLANNER_RELAXED_FIRST = {"v_h": 3.0, "a_h": 3.0, "a_h_brake_long": 2.0, "long_h_m": 10.0,
                         "v_climb": 2.0, "a_climb": 1.0, "v_descent": 2.5, "a_descent": 3.0, "descent_cap": 2.0,
                         # oblique descents: with independent component limits a -60 deg descent let the net rise to
                         # 20 mm above the rotor limit (the tilt lowers the leading rotors while the net floats up);
                         # 0.8 passed with 3 mm to spare. Climbing obliques passed up to 1.5x with no derating.
                         "descent_oblique_derate": 0.75, "climb_oblique_derate": 1.0}
# Strict planner: shape kept in transit (50 mm cruise, 100 mm ramp tolerance). For each acceleration limit a,
# the strict speed limit per axis (horizontal: travel sweep, worst heading = diagonal; vertical: characterisation);
# the planner picks the acceleration that gives the shortest leg.
# (vertical: 5 m climb / descent tests; where the 5 m leg could not reach the speed, the highest speed reached)
PLANNER_STRICT = {"a": [0.5, 1.0, 2.0, 3.0], "v_h": [3.6, 1.4, 1.0, 0.6],
                  "v_up": [1.1, 1.6, 2.0, 1.0], "v_down": [1.1, 1.6, 2.0, 1.5]}


def _axis_min(u, lim_h, lim_up, lim_down):
    """Largest along-leg value whose horizontal and vertical components respect their own limits."""
    ch, cz = float(np.hypot(u[0], u[1])), float(u[2])
    vals = []
    if ch > 1e-9:
        vals.append(lim_h / ch)
    if abs(cz) > 1e-9:
        vals.append((lim_up if cz > 0 else lim_down) / abs(cz))
    return min(vals)


def plan_leg_relaxed(d, factor=1.0, P=None):
    P = P or PLANNER_RELAXED
    L = float(np.linalg.norm(d))
    u = d / L
    Lh = float(np.hypot(d[0], d[1]))
    a_hb = P["a_h_brake_long"] if Lh > P["long_h_m"] else P["a_h"]
    v_dn = min(P["v_descent"], P["descent_cap"]) if P["descent_cap"] else P["v_descent"]
    v = _axis_min(u, P["v_h"], P["v_climb"], v_dn)
    # climbs and descents each use their own (symmetric) vertical acceleration for speeding up and braking
    a_up = _axis_min(u, P["a_h"], P["a_climb"], P["a_descent"])
    a_dn = _axis_min(u, a_hb, P["a_climb"], P["a_descent"])
    ch = np.hypot(u[0], u[1])
    der = 1.0
    if ch > 0.17 and u[2] < -0.17:
        der = P["descent_oblique_derate"]
    elif ch > 0.17 and u[2] > 0.17:
        der = P.get("climb_oblique_derate", 1.0)
    f = factor * der
    return {"v": v * f, "a_up": a_up * f, "a_down": a_dn * f, "rule": "relaxed", "factor": factor}


def plan_leg_strict(d, factor=1.0, P=None):
    P = P or PLANNER_STRICT
    L = float(np.linalg.norm(d))
    u = d / L
    best = None
    for a, vh, vu, vd in zip(P["a"], P["v_h"], P["v_up"], P["v_down"]):
        v = _axis_min(u, vh, vu, vd)
        vv, Tu, Tc, Tb = fit_profile(L, v, a, a)
        T = Tu + Tc + Tb
        if best is None or T < best[0]:
            best = (T, {"v": v * factor, "a_up": a * factor, "a_down": a * factor, "rule": "strict", "factor": factor})
    return best[1]


MISSION_DEFAULT = dict(seed=0, n_targets=10, planner="relaxed", factor=1.0, settle_mode="avg", ff_path=None,
                       targets=None, legs=None, p_over=None, keep_linear=False, label=None, planner_over=None,
                       tol=None, rotor_check=False, final_strict_s=30.0, spacing=1.25, h_dn=None, targets_from=None,
                       params_note=None)
# p_over (Params overrides, e.g. {"L": 0.5, "m_net": 0.01, "d_takeoff": 0.70}), spacing (hold / pattern spacing),
# h_dn (drone height above the net centre in hold; None: hold_drop()), ff_path (net-pull table), targets_from (a
# study JSON whose "targets" are reused for the same seed) describe another net / formation.
# The first study's setting, reproduced exactly: tol=MISSION_TOL_STRICT, rotor_check=True (final_strict_s is then
# inactive, the strict rule being the settle rule itself).
MISSION_FIRST_STUDY = dict(tol=MISSION_TOL_STRICT, rotor_check=True, planner_over=PLANNER_RELAXED_FIRST)


def mission_planner(cfg):
    if cfg.get("legs") is not None:                      # explicit per-leg profiles (characterisation tests)
        it = iter(cfg["legs"])
        return lambda d: dict(next(it))
    if cfg["planner"] == "strict":
        return lambda d: plan_leg_strict(d, cfg["factor"])
    P = {**PLANNER_RELAXED, **(cfg.get("planner_over") or {})}
    return lambda d: plan_leg_relaxed(d, cfg["factor"], P)


def mission_leg_metrics(p, log, plan, cl_series, t_arm, safety=None):
    """Safety and timing per leg (window: departure to the next departure, or the end of the run)."""
    C = MISSION_CRIT
    safety = MISSION_SAFETY if safety is None else safety
    t = log["t"]
    dt = float(np.median(np.diff(t)))
    pos, ref = log["pos"], log["ref"]
    mf, mraw = log["motor"] / p.f_max, log["motor_raw"] / p.f_max
    sep = np.min([np.linalg.norm(pos[:, i] - pos[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)], axis=0)
    zerr = np.abs(pos[:, :, 2] - ref[:, :, 2]).max(1)
    e3 = np.linalg.norm(pos - ref, axis=2).max(1)
    # distance to the box faces (sim coordinates), drones with their airframe extent, knots as points
    lo, hi = -BOX_OFFSET, np.array([BOX_SIZE, BOX_SIZE, BOX_SIZE]) - BOX_OFFSET
    knots_min = log["net_zmin"]
    db = np.minimum.reduce([(pos[:, :, 0] - DRONE_R - lo[0]).min(1), (hi[0] - pos[:, :, 0] - DRONE_R).min(1),
                            (pos[:, :, 1] - DRONE_R - lo[1]).min(1), (hi[1] - pos[:, :, 1] - DRONE_R).min(1),
                            (hi[2] - pos[:, :, 2] - DRONE_HALF_H).min(1), (pos[:, :, 2] - DRONE_HALF_H).min(1)])
    fr = log["frames"][:, :p.n * p.n]
    fb = np.minimum.reduce([fr[:, :, 0].min(1) - lo[0], hi[0] - fr[:, :, 0].max(1), fr[:, :, 1].min(1) - lo[1],
                            hi[1] - fr[:, :, 1].max(1), fr[:, :, 2].min(1), hi[2] - fr[:, :, 2].max(1)])
    # net below the rotors: every knot except the four corner knots (they hang from the hooks ~83 mm below the
    # centre of mass by construction; their distance to discs and guards is the clearance check) stays 50 mm
    # below the lowest rotor hub of the formation. Literal variant (any knot vs the lowest edge of any tilted
    # disc) reported as a diagnostic only: at hover it has 26 mm of margin, and a 30 deg tilt of one drone
    # alone uses it up through geometry, without the net moving.
    kmargin = log["disc_czmin"] - C["knot_below_disc_m"] - log["knot_zmax_inner"]
    kmargin_lit = log["disc_zmin"] - C["knot_below_disc_m"] - log["knot_zmax"]
    ft = cl_series["t"]
    dmin, gmin = cl_series["all"].min(1), np.nanmin(cl_series["guard"], axis=1)
    sag = log["sag_net"]
    fc = pos.mean(axis=1)
    rel = log["net_c"] - fc
    offh = np.hypot(rel[:, 0], rel[:, 1])
    vmag = np.linalg.norm(log["vel"].mean(axis=1), axis=1)
    # net shape in the formation frame (origin at the drones' mean position, translation removed): every knot's
    # offset from its place in the steady hold before the first departure (20 Hz frames); the centre knot (the
    # payload, if any) at 100 Hz for its swing after each stop
    nn = p.n
    ftt = log["frames_t"]
    relk = log["frames"][:, :nn * nn] - log["frames_pos"].mean(axis=1)[:, None, :]
    w0 = (ftt >= plan.t_go0 - 1.0) & (ftt < plan.t_go0)
    rest = relk[w0].mean(axis=0) if w0.any() else relk[0]
    kdev = np.linalg.norm(relk - rest, axis=2)
    s_rms, s_max = np.sqrt((kdev ** 2).mean(axis=1)), kdev.max(axis=1)
    cc = (nn // 2) * nn + nn // 2
    cdev = np.linalg.norm(rel - rest[cc], axis=1)
    out = []
    for lg in plan.legs:
        t0 = lg["t_dep"]
        t1 = lg["t_next_dep"] if lg["t_next_dep"] is not None else float(t[-1])
        W = (t >= t0) & (t <= t1)
        fW = (ft >= t0) & (ft <= t1)
        if not W.any():
            continue
        r = {k: lg[k] for k in ("leg", "t_dep", "L", "v", "Tu", "Tc", "Tb", "t_ref_stop", "t_settled", "settled")}
        # formation (drone-mean) centre at the start and end of the leg, and the net-centre target
        r["from_box"] = (np.array(lg["from_sim"]) + BOX_OFFSET).round(3).tolist()
        r["to_box"] = (np.array(lg["to_sim"]) + BOX_OFFSET).round(3).tolist()
        r["target_net_box"] = (np.array(lg["to_sim"]) + BOX_OFFSET - np.array([0.0, 0.0, MISSION_HDN[0]])).round(3).tolist()
        u = np.array(lg["u"])
        r["elevation_deg"] = round(float(np.degrees(np.arcsin(np.clip(u[2], -1, 1)))), 2)
        r["heading_deg"] = round(float(np.degrees(np.arctan2(u[1], u[0]))), 2)
        r["profile"] = lg["profile"]
        r["leg_time_s"] = round(lg["t_settled"] - t0, 3) if lg["settled"] else None
        r["move_time_s"] = round(lg["t_ref_stop"] - t0, 3)
        r["settle_time_s"] = round(lg["t_settled"] - lg["t_ref_stop"], 3) if lg["settled"] else None
        r["avg_speed_m_s"] = round(lg["L"] / r["leg_time_s"], 3) if r["leg_time_s"] else None
        r["peak_speed_m_s"] = round(float(vmag[W].max()), 3)
        crit = {}
        bad = W & ((log["tilt"].max(1) > C["tilt_deg"]) | (log["ncon"] > 0) | (zerr > C["alt_err_m"]))
        crit["control"] = {"ok": not bad.any(), "tilt_max_deg": round(float(log["tilt"][W].max()), 2),
                           "alt_err_max_mm": round(float(1e3 * zerr[W].max()), 1),
                           "contacts": int(log["ncon"][W].max())}
        hi90 = W & (mf > C["motor_frac"]).any(axis=(1, 2))
        a_, b_ = _runs(hi90)
        lgst = float((b_ - a_).max() * dt) if len(a_) else 0.0
        sat = W & (mraw >= 1.0).any(axis=(1, 2))
        crit["motor90"] = {"ok": lgst <= C["motor_time_s"] + 1e-9, "longest_s": round(lgst, 3),
                           "motor_max": round(float(mf[W].max()), 4), "saturated_s": round(float(sat.sum() * dt), 3)}
        crit["clearance"] = {"ok": bool(fW.any() and dmin[fW].min() >= C["disc_m"] and gmin[fW].min() >= C["guard_m"]),
                             "disc_min_mm": round(float(1e3 * dmin[fW].min()), 1) if fW.any() else None,
                             "guard_min_mm": round(float(1e3 * gmin[fW].min()), 1) if fW.any() else None}
        crit["net_low"] = {"ok": bool(log["net_zmin"][W].min() > C["net_low_m"]),
                           "net_lowest_m": round(float(log["net_zmin"][W].min()), 4)}
        crit["separation"] = {"ok": bool(sep[W].min() >= C["separation_m"]), "min_m": round(float(sep[W].min()), 4)}
        slack = W[:, None] & (log["tension"] <= 0.0)
        crit["slack"] = {"ok": float(slack.sum(0).max() * dt) <= C["slack_s"] + 1e-9,
                         "slack_max_s": round(float(slack.sum(0).max() * dt), 3)}
        dbox = min(float(db[W].min()), float(fb[fW].min()) if fW.any() else np.inf)
        crit["box"] = {"ok": dbox >= 0.0, "min_dist_m": round(dbox, 4)}
        crit["net_below_rotors"] = {"ok": bool(kmargin[W].min() >= 0.0), "margin_min_m": round(float(kmargin[W].min()), 4),
                                    "literal_margin_min_m": round(float(kmargin_lit[W].min()), 4)}
        r["criteria"] = crit
        r["failed"] = [k for k in safety if not crit[k]["ok"]] + ([] if lg["settled"] else ["not_settled"])
        # state at departure (what this leg inherits from the previous settle): drone error from the leg's start
        # positions, drone speed, net offset / sag change (instantaneous and 0.5 s mean, as judged), net swing
        # (largest 3-D excursion of the net centre, relative to the formation centre, from its mean over the
        # last 1 s) and the net centre's speed relative to the formation (over the last 0.1 s)
        i0 = int(np.searchsorted(t, t0 - 1e-9))
        i0 = min(i0, len(t) - 1)
        st = np.array(lg["from_sim"]) + plan.pattern
        na, n1 = max(1, int(round(0.5 / dt))), max(1, int(round(1.0 / dt)))
        a0, b0 = max(0, i0 - na + 1), i0 + 1
        c0 = max(0, i0 - n1 + 1)
        relw = rel[c0:b0]
        nv = max(1, int(round(0.1 / dt)))
        relv = (rel[i0] - rel[max(0, i0 - nv)]) / (nv * dt)
        r["departure"] = {"pos_err_mm": round(float(1e3 * np.linalg.norm(pos[i0] - st, axis=1).max()), 2),
                          "drone_speed_mm_s": round(float(1e3 * np.linalg.norm(log["vel"][i0], axis=1).max()), 1),
                          "net_off_mm": round(float(1e3 * offh[i0]), 2),
                          "net_off_avg_mm": round(float(1e3 * offh[a0:b0].mean()), 2),
                          "sag_dev_mm": round(float(1e3 * (sag[i0] - plan.sag0)), 2),
                          "sag_dev_avg_mm": round(float(1e3 * (sag[a0:b0].mean() - plan.sag0)), 2),
                          "net_swing_mm": round(float(1e3 * np.linalg.norm(relw - relw.mean(0), axis=1).max()), 2),
                          "net_rel_speed_mm_s": round(float(1e3 * np.linalg.norm(relv)), 1)}
        if lg.get("t_strict_settled") is not None or (lg is plan.legs[-1] and plan.final_strict_s > 0 and lg["settled"]):
            ts = lg.get("t_strict_settled")
            r["final_strict"] = {"t_strict_settled": ts,
                                 "extra_after_settled_s": None if ts is None else round(ts - lg["t_settled"], 3),
                                 "after_ref_stop_s": None if ts is None else round(float(ts - lg["t_ref_stop"]), 3),
                                 "waited_up_to_s": plan.final_strict_s}
        r["pass"] = not r["failed"]
        # the strict tracking tolerances on the same leg (reference phases)
        ramp = W & (((t >= t0) & (t < t0 + lg["Tu"])) | ((t >= t0 + lg["Tu"] + lg["Tc"]) & (t < lg["t_ref_stop"])))
        cru = W & (t >= t0 + lg["Tu"]) & (t < t0 + lg["Tu"] + lg["Tc"])
        r["strict"] = {"ramp_max_mm": round(float(1e3 * e3[ramp].max()), 1) if ramp.any() else None,
                       "cruise_max_mm": round(float(1e3 * e3[cru].max()), 1) if cru.any() else None}
        r["strict"]["ok"] = bool((r["strict"]["ramp_max_mm"] or 0) <= 100 and (r["strict"]["cruise_max_mm"] or 0) <= 50)
        # shape and swing
        mvf = (ftt >= t0) & (ftt < lg["t_ref_stop"])
        lgf = (ftt >= t0) & (ftt <= t1)
        at = lambda tt: int(np.argmin(np.abs(ftt - tt)))
        t1d = float(t[-1]) if lg is plan.legs[-1] else t1          # last target: the swing until the log ends
        Wd = (t >= lg["t_ref_stop"]) & (t <= t1d)
        dec = {}
        for thr in (0.050, 0.025, 0.010):
            over = np.flatnonzero(Wd & (cdev > thr))
            if not len(over):
                dec[f"{int(thr * 1e3)}mm"] = 0.0
            elif over[-1] >= np.flatnonzero(Wd)[-1]:
                dec[f"{int(thr * 1e3)}mm"] = None                     # not yet within thr when the window ends
            else:
                dec[f"{int(thr * 1e3)}mm"] = round(float(t[over[-1]] + dt - lg["t_ref_stop"]), 3)
        Wm = W & (t < lg["t_ref_stop"])
        Wa = W & (t >= lg["t_ref_stop"])
        r["shape"] = {"rms_move_peak_mm": round(float(1e3 * s_rms[mvf].max()), 1) if mvf.any() else None,
                      "max_move_peak_mm": round(float(1e3 * s_max[mvf].max()), 1) if mvf.any() else None,
                      "rms_leg_peak_mm": round(float(1e3 * s_rms[lgf].max()), 1) if lgf.any() else None,
                      "rms_at_settled_mm": round(float(1e3 * s_rms[at(lg["t_settled"])]), 2) if lg["settled"] else None,
                      "max_at_settled_mm": round(float(1e3 * s_max[at(lg["t_settled"])]), 2) if lg["settled"] else None,
                      "rms_at_next_dep_mm": round(float(1e3 * s_rms[at(t1)]), 2),
                      "centre_dev_at_stop_mm": round(float(1e3 * cdev[np.argmin(np.abs(t - lg["t_ref_stop"]))]), 1),
                      "centre_swing_within_s": dec,
                      "drone_err_move_max_mm": round(float(1e3 * e3[Wm].max()), 1) if Wm.any() else None,
                      "tilt_move_max_deg": round(float(log["tilt"][Wm].max()), 2) if Wm.any() else None,
                      "drone_err_after_stop_max_mm": round(float(1e3 * e3[Wa].max()), 1) if Wa.any() else None,
                      "tilt_after_stop_max_deg": round(float(log["tilt"][Wa].max()), 2) if Wa.any() else None,
                      "tilt_after_stop_std_deg": round(float(log["tilt"][Wa].std()), 3) if Wa.any() else None}
        r["net"] = {"sag_range_m": [round(float(sag[W].min()), 4), round(float(sag[W].max()), 4)],
                    "offset_h_max_mm": round(float(1e3 * offh[W].max()), 1),
                    "vert_offset_range_mm": [round(float(1e3 * (rel[W, 2] + MISSION_HDN[0]).min()), 1),
                                             round(float(1e3 * (rel[W, 2] + MISSION_HDN[0]).max()), 1)],
                    "tension_gf_min": round(float(log["tension"][W].min() / G * 1e3), 2),
                    "tension_gf_max": round(float(log["tension"][W].max() / G * 1e3), 2)}
        out.append(r)
    return out


MISSION_HDN = [0.401]                # drone height above the net centre in hold (set by run_mission)


def run_mission(cfg, out=None, tag=None, verbose=True):
    """One mission (cfg: MISSION_DEFAULT keys). targets: box coordinates (N, 3) or None for mission_targets(seed)."""
    cfg = {**MISSION_DEFAULT, **cfg}
    p = travel_params({**TRAVEL_DEFAULT, "p_over": cfg["p_over"], "keep_linear": cfg["keep_linear"]})
    h_dn = hold_drop() if cfg["h_dn"] is None else float(cfg["h_dn"])
    MISSION_HDN[0] = h_dn
    if cfg["targets"] is None and cfg["targets_from"]:
        with open(cfg["targets_from"]) as f:
            T = np.asarray(json.load(f)["targets"][str(cfg["seed"])], float)[:cfg["n_targets"]]
    elif cfg["targets"] is None:
        T, _ = mission_targets(cfg["seed"], cfg["n_targets"], h_dn)
    else:
        T = np.asarray(cfg["targets"], float)
    tol = dict(MISSION_TOL if cfg["tol"] is None else cfg["tol"])
    cfg["tol"] = tol
    plan = MissionPlan(p, T - BOX_OFFSET, mission_planner(cfg), h_dn, settle_mode=cfg["settle_mode"], tol=tol,
                       final_strict_s=cfg["final_strict_s"], spacing=cfg["spacing"])
    ff_path = cfg["ff_path"] or os.path.join(HERE, "out_official", "sweep_summary.json")
    ff = load_ff(p, ff_path, spacing=cfg["spacing"])
    sim = Sim(p)
    t0 = time.time()
    log = sim.run(plan, ff=ff, acc_ff=1.0, extra_log=True, verbose=verbose, abort_err=8.0)
    wall = time.time() - t0
    phases = [("on_ground", 0.0, 0.5), ("takeoff", 0.5, 6.5), ("spread", 6.5, 12.5), ("pre_travel", 12.5, plan.t_go0),
              ("cruise", plan.t_go0, float(log["t"][-1]))]
    cl, cl_series = clearance_analysis(p, log, phases)
    pl_info = None
    if payload_info(p):                         # the payload sphere joins the rotor-disc and guard checks
        pc_d, pc_g = payload_clearance(p, log)
        cl_series["all"] = np.minimum(cl_series["all"], pc_d)
        cl_series["guard"] = np.fmin(cl_series["guard"], pc_g)
        pl_info = {**payload_info(p), "disc_min_mm": round(float(1e3 * pc_d.min()), 1),
                   "guard_min_mm": round(float(1e3 * pc_g.min()), 1)}
    legs = mission_leg_metrics(p, log, plan, cl_series, 0.5,
                               safety=MISSION_SAFETY_ROTOR if cfg["rotor_check"] else MISSION_SAFETY)
    done = [r for r in legs if r["leg_time_s"] is not None]
    tot_L = float(sum(r["L"] for r in legs))
    met = {"cfg": {k: (np.asarray(v).tolist() if k == "targets" and v is not None else v) for k, v in cfg.items()},
           "targets_box": np.round(T, 4).tolist(), "h_dn_m": h_dn, "sag0_m": plan.sag0,
           "n_legs": len(plan.Cs) - 1, "n_flown": len(legs), "n_pass": sum(r["pass"] for r in legs),
           "pass": len(legs) == len(plan.Cs) - 1 and all(r["pass"] for r in legs),
           "failed_legs": [(r["leg"], r["failed"]) for r in legs if not r["pass"]],
           "total_distance_m": round(tot_L, 3),
           "total_leg_time_s": round(sum(r["leg_time_s"] for r in done), 3) if len(done) == len(legs) else None,
           "mission_time_s": round(float(plan.legs[-1]["t_settled"] - plan.legs[0]["t_dep"]), 3)
           if plan.legs and plan.legs[-1]["settled"] else None,
           "aborted": bool(np.isfinite(log.get("aborted_at", np.nan))), "wall_s": round(wall, 1),
           "settle_tol": tol, "safety_criteria": MISSION_SAFETY_ROTOR if cfg["rotor_check"] else MISSION_SAFETY,
           "final_strict": legs[-1].get("final_strict") if legs else None, "legs": legs}
    if pl_info:
        met["payload"] = pl_info
    if met["total_leg_time_s"]:
        met["avg_speed_legs_m_s"] = round(tot_L / met["total_leg_time_s"], 3)
    if met["mission_time_s"]:
        met["avg_speed_mission_m_s"] = round(tot_L / met["mission_time_s"], 3)
    if out:
        tag = tag or cfg["label"] or f"mission_seed{cfg['seed']}_{cfg['planner']}_f{cfg['factor']:g}"
        met["render_shots"] = mission_shots(legs, log)
        met["box"] = {"size_m": BOX_SIZE, "offset_m": BOX_OFFSET.tolist()}
        for key in ("all", "net", "footprint", "taut", "guard"):
            log[f"clearance_{key}"] = cl_series[key]
        np.savez_compressed(os.path.join(out, f"{tag}_log.npz"), **log)
        with open(os.path.join(out, f"{tag}_summary.json"), "w") as f:
            json.dump({"mode": "mission", **met, "criteria_limits": MISSION_CRIT, "clearance": cl,
                       "planner_relaxed": {**PLANNER_RELAXED, **(cfg.get("planner_over") or {})},
                       "planner_strict": PLANNER_STRICT, "aero": aero_info(p), "params": asdict(p)}, f, indent=1,
                      default=_json_np)
        plot_mission(p, log, met, os.path.join(out, f"{tag}_timeseries.png"), os.path.join(out, f"{tag}_path.png"), cl_series)
        met["files"] = [f"{tag}_summary.json", f"{tag}_log.npz", f"{tag}_timeseries.png", f"{tag}_path.png"]
    return met


def mission_shots(legs, log):
    """Moments for render_scene: a high target, a steep climb or descent, a fast horizontal leg, the box overview."""
    shots = []
    ok = [r for r in legs if r["t_settled"] is not None]
    if ok:
        hi = max(ok, key=lambda r: r["to_box"][2])
        shots.append(("mission_01_high_target", hi["t_settled"] + 1.0, "wide",
                      f"settled at a high target, net centre z = {hi['to_box'][2] - MISSION_HDN[0]:.2f} m, drones "
                      f"{hi['to_box'][2]:.2f} m (box coordinates)"))
    st = max(legs, key=lambda r: abs(r["L"] * np.sin(np.radians(r["elevation_deg"]))))
    shots.append(("mission_02_steep_leg", st["t_dep"] + 0.5 * (st["t_ref_stop"] - st["t_dep"]), "side",
                  f"mid-leg, elevation {st['elevation_deg']:.0f} deg, {'climb' if st['elevation_deg'] > 0 else 'descent'}"))
    fl = [r for r in legs if abs(r["elevation_deg"]) < 30] or legs
    fh = max(fl, key=lambda r: r["peak_speed_m_s"])
    t, v = log["t"], np.linalg.norm(log["vel"].mean(axis=1), axis=1)
    w = (t >= fh["t_dep"]) & (t <= fh["t_ref_stop"])
    shots.append(("mission_03_fast_leg", float(t[w][np.argmax(v[w])]), "wide",
                  f"fastest flat leg, {v[w].max():.2f} m/s, elevation {fh['elevation_deg']:.0f} deg"))
    shots.append(("mission_04_box_overview", float(t[w][np.argmax(v[w])]), "box", "the 10 x 10 x 10 m box, same moment"))
    return [list(s) for s in shots]


def plot_mission(p, log, met, path_ts, path_path, cl_series):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    t = log["t"]
    pos = log["pos"] + BOX_OFFSET
    fc = pos.mean(axis=1)
    nc = log["net_c"] + BOX_OFFSET
    T = np.array(met["targets_box"])
    legs = met["legs"]
    # path figure: 3-D and projections with the box
    fig = plt.figure(figsize=(16, 5.5))
    corners = np.array([[x, y, z] for x in (0, BOX_SIZE) for y in (0, BOX_SIZE) for z in (0, BOX_SIZE)])
    edges = [(a, b) for a in range(8) for b in range(a + 1, 8) if np.sum(corners[a] != corners[b]) == 1]
    ax = fig.add_subplot(1, 4, 1, projection="3d")
    for a, b in edges:
        ax.plot(*corners[[a, b]].T, color="k", lw=0.6)
    ax.plot(*fc.T, color="tab:blue", lw=1, label="formation centre")
    ax.plot(*nc.T, color="tab:orange", lw=0.7, label="net centre")
    ax.scatter(*T.T, color="tab:red", s=18, label="targets (net centre)")
    for i, x in enumerate(T):
        ax.text(*x, str(i + 1), fontsize=7)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_zlim(0, 10); ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z"); ax.legend(fontsize=7, loc="upper left")
    for j, (i0, i1, lab) in enumerate(((0, 1, "top view x-y"), (0, 2, "side view x-z"), (1, 2, "side view y-z"))):
        ax = fig.add_subplot(1, 4, j + 2)
        ax.add_patch(plt.Rectangle((0, 0), BOX_SIZE, BOX_SIZE, fill=False, color="k", lw=0.8))
        ax.plot(fc[:, i0], fc[:, i1], color="tab:blue", lw=1)
        ax.plot(nc[:, i0], nc[:, i1], color="tab:orange", lw=0.6)
        ax.scatter(T[:, i0], T[:, i1], color="tab:red", s=14, zorder=3)
        for i, x in enumerate(T):
            ax.annotate(str(i + 1), (x[i0], x[i1]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlim(-0.3, 10.3); ax.set_ylim(-0.3, 10.3); ax.set_aspect("equal"); ax.grid(alpha=0.3)
        ax.set_title(lab); ax.set_xlabel("xyz"[i0] + " [m]"); ax.set_ylabel("xyz"[i1] + " [m]")
    c = met["cfg"]
    fig.suptitle(f"Mission seed {c['seed']}, {c['planner']} planner, factor {c['factor']}: {met['n_pass']}/{met['n_legs']} legs pass, "
                 f"{met['total_distance_m']:.1f} m in {met['total_leg_time_s']} s of leg time "
                 f"(blue: formation centre, orange: net centre, red: targets)")
    fig.tight_layout()
    fig.savefig(path_path, dpi=110)
    plt.close(fig)
    # timeseries
    s = t >= 13.0
    fig, axs = plt.subplots(6, 1, figsize=(15, 19), sharex=True)
    ax = axs[0]
    for i, lab in enumerate("xyz"):
        ax.plot(t[s], fc[s, i], label=f"formation centre {lab}")
        ax.plot(t[s], log["ref"][s].mean(axis=1)[:, i] + BOX_OFFSET[i], "k--", lw=0.6)
    ax.set_ylabel("position (box) [m]"); ax.legend(fontsize=7, ncol=3)
    ax2 = ax.twinx()
    ax2.plot(t[s], np.linalg.norm(log["vel"][s].mean(axis=1), axis=1), color="tab:gray", lw=0.8)
    ax2.set_ylabel("speed [m/s] (gray)")
    ax = axs[1]
    ax.plot(t[s], log["tilt"][s], lw=0.8); ax.axhline(p.max_tilt_deg, color="k", ls=":", lw=0.8); ax.set_ylabel("tilt [deg]")
    ax = axs[2]
    ax.plot(t[s], log["motor"][s].max(axis=2) / p.f_max, lw=0.8); ax.axhline(0.9, color="k", ls=":", lw=0.8)
    ax.set_ylabel("highest motor / limit"); ax.set_ylim(0, 1.05)
    ax = axs[3]
    ax.plot(t[s], log["tension"][s] / G * 1e3, lw=0.8); ax.set_ylabel("string tension [gf]")
    ax = axs[4]
    ax.plot(t[s], log["sag_net"][s], label="net sag"); ax.plot(t[s], log["net_zmin"][s], label="lowest knot z")
    ax.plot(t[s], log["disc_czmin"][s] - 0.05 - log["knot_zmax_inner"][s], label="net margin below rotor hubs - 50 mm")
    ax.axhline(0.3, color="k", ls=":", lw=0.8); ax.set_ylabel("[m]"); ax.legend(fontsize=7)
    ax = axs[5]
    fs = cl_series["t"] >= 13.0
    ax.plot(cl_series["t"][fs], 1e3 * cl_series["all"][fs].min(1), label="rotor-disc clearance (worst drone)")
    ax.plot(cl_series["t"][fs], 1e3 * np.nanmin(cl_series["guard"][fs], axis=1), label="guard clearance (worst drone)")
    ax.axhline(30, color="k", ls=":", lw=0.8); ax.axhline(10, color="k", ls="-.", lw=0.8)
    ax.set_ylim(0, 80); ax.set_ylabel("clearance [mm]"); ax.legend(fontsize=7)
    for a in axs:
        a.grid(alpha=0.3)
        for r in legs:
            a.axvline(r["t_dep"], color="tab:green", lw=0.6)
            if r["t_settled"] is not None:
                a.axvspan(r["t_settled"], r["t_settled"] + 2.0, color="tab:green", alpha=0.12)
    for r in legs:
        axs[0].text(r["t_dep"], axs[0].get_ylim()[1], f"{r['leg'] + 1}", fontsize=8, color="tab:green", va="bottom")
    axs[-1].set_xlabel("t [s] (green lines: departures; shaded: 2 s settled windows)")
    fig.suptitle(f"Mission seed {c['seed']}, {c['planner']} planner, factor {c['factor']}")
    fig.tight_layout()
    fig.savefig(path_ts, dpi=100)
    plt.close(fig)


def _mission_job(cfg):
    try:
        m = run_mission(cfg, verbose=False)
        m.pop("render_shots", None)
        return m
    except Exception as exc:
        import traceback
        return {"cfg": cfg, "pass": False, "error": repr(exc), "trace": traceback.format_exc(), "legs": []}


MISSION_FACTOR_FIRST = 0.8           # first study (10 mm rules, rotor check): all 100 legs pass (1.0 fails 4)
MISSION_FACTOR = 1.2                 # current rules (missions_tol50 study): fastest factor with all 100 legs passing
                                     # (1.086 m/s; 1.3 fails 2 climbing obliques on the 90 % motor rule)


def print_mission(met):
    c = met["cfg"]
    tl = met.get("settle_tol") or {}
    print(f"mission seed {c['seed']} {c['planner']} factor {c['factor']} settle {c['settle_mode']} "
          f"(tolerances {', '.join(f'{k} {1e3 * v:g} mm' for k, v in tl.items())}; fails on {met.get('safety_criteria')}): "
          f"{met['n_pass']}/{met['n_legs']} legs pass, distance {met['total_distance_m']} m, leg time "
          f"{met['total_leg_time_s']} s, mission time {met['mission_time_s']} s, average speed "
          f"{met.get('avg_speed_legs_m_s')} m/s (legs) / {met.get('avg_speed_mission_m_s')} m/s (with 2 s windows)")
    for r in met["legs"]:
        cr = r["criteria"]
        print(f"  leg {r['leg'] + 1:2d} L {r['L']:5.2f} m el {r['elevation_deg']:+6.1f} deg: "
              f"{'pass' if r['pass'] else 'FAIL ' + ','.join(r['failed'])} time {r['leg_time_s']} (move {r['move_time_s']}, "
              f"settle {r['settle_time_s']}) v_ref {r['v']:.2f} peak {r['peak_speed_m_s']} | disc {cr['clearance']['disc_min_mm']} "
              f"guard {cr['clearance']['guard_min_mm']} motor {cr['motor90']['motor_max']} tilt {cr['control']['tilt_max_deg']} "
              f"alt {cr['control']['alt_err_max_mm']} sep {cr['separation']['min_m']} box {cr['box']['min_dist_m']} "
              f"rotor margin {cr['net_below_rotors']['margin_min_m']} low {cr['net_low']['net_lowest_m']} slack {cr['slack']['slack_max_s']}")
    if met.get("final_strict"):
        fs = met["final_strict"]
        print(f"  last target: strict 10 / 10 / 5 mm rule met {fs['extra_after_settled_s']} s after the settle "
              f"({fs['after_ref_stop_s']} s after the reference stop; waited up to {fs['waited_up_to_s']} s)")


def characterise_cfgs():
    """Single test legs (net-centre targets, box coordinates) with explicit profiles: pure climb, pure descent
    (after a gentle climb), oblique legs at +-30 and +-60 deg elevation and level legs at four headings."""
    cfgs = []
    gentle = {"v": 1.0, "a_up": 1.0, "a_down": 1.0}
    z0 = 0.80
    for v in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        for a in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
            prof = {"v": v, "a_up": a, "a_down": a}
            cfgs.append({"label": f"climb_v{v:g}_a{a:g}", "kind": "climb", "test_leg": 0,
                         "targets": [[5.0, 5.0, z0 + 5.0]], "legs": [prof]})
    for v in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        for a in (0.5, 1.0, 2.0, 3.0, 4.0, 6.0):
            prof = {"v": v, "a_up": a, "a_down": a}
            cfgs.append({"label": f"descent_v{v:g}_a{a:g}", "kind": "descent", "test_leg": 1,
                         "targets": [[5.0, 5.0, z0 + 5.0], [5.0, 5.0, z0]], "legs": [gentle, prof]})
    return cfgs


def long_vertical_cfgs(L=8.0, z0=0.80):
    """8 m climbs and descents (the longest vertical legs in the missions are ~9 m): climb speeds above what a
    5 m leg can reach, descents at the 2.0 m/s cap."""
    cfgs = []
    gentle = {"v": 1.0, "a_up": 1.0, "a_down": 1.0}
    for v, a in ((2.0, 2.0), (2.5, 2.0), (3.0, 2.0), (3.5, 2.0), (2.5, 3.0), (3.0, 3.0)):
        cfgs.append({"label": f"climb8_v{v:g}_a{a:g}", "kind": "climb_long", "test_leg": 0,
                     "targets": [[5.0, 5.0, z0 + L]], "legs": [{"v": v, "a_up": a, "a_down": a}]})
    for v, a in ((2.0, 2.0), (2.0, 3.0)):
        cfgs.append({"label": f"descent8_v{v:g}_a{a:g}", "kind": "descent_long", "test_leg": 1,
                     "targets": [[5.0, 5.0, z0 + L], [5.0, 5.0, z0]], "legs": [gentle, {"v": v, "a_up": a, "a_down": a}]})
    return cfgs


def horizontal_cfgs(L=5.0, headings=(0.0, 45.0), speeds=(2.0, 3.0, 4.0, 5.0), accels=(2.0, 3.0, 4.0, 6.0)):
    """Level legs with explicit profiles (speed x acceleration grid) along a net side and a net diagonal."""
    cfgs = []
    gentle = {"v": 1.0, "a_up": 1.0, "a_down": 1.0}
    for hd in headings:
        d = np.array([np.cos(np.radians(hd)), np.sin(np.radians(hd)), 0.0]) * L
        start = np.array([5.0, 5.0, 1.0]) - d / 2
        for v in speeds:
            for a in accels:
                cfgs.append({"label": f"horizontal_hd{hd:g}_v{v:g}_a{a:g}", "kind": "horizontal", "heading": hd,
                             "test_leg": 1, "targets": [start.tolist(), (start + d).tolist()],
                             "legs": [gentle, {"v": v, "a_up": a, "a_down": a}]})
    return cfgs


def oblique_cfgs(factors=(0.8, 1.0, 1.25, 1.5), P=None):
    """Oblique and level legs flown with the relaxed rule at several aggressiveness factors."""
    cfgs = []
    gentle = {"v": 1.0, "a_up": 1.0, "a_down": 1.0}
    for el in (30.0, 60.0, -30.0, -60.0):
        L = 5.0
        dx, dz = L * np.cos(np.radians(el)), L * np.sin(np.radians(el))
        z_start = 1.0 if el > 0 else 1.0 - dz
        start = [5.0 - dx / 2, 5.0, z_start]
        end = [5.0 + dx / 2, 5.0, z_start + dz]
        for f in factors:
            prof = plan_leg_relaxed(np.array(end) - np.array(start), f, P)
            cfgs.append({"label": f"oblique_el{el:+g}_f{f:g}", "kind": "oblique", "elevation": el, "factor": f,
                         "test_leg": 1, "targets": [start, end], "legs": [gentle, prof]})
    for hd in (0.0, 22.5, 45.0, 67.5):
        L = 5.0
        d = np.array([np.cos(np.radians(hd)), np.sin(np.radians(hd)), 0.0]) * L
        start = np.array([5.0, 5.0, 1.0]) - d / 2
        end = start + d
        for f in factors:
            prof = plan_leg_relaxed(d, f, P)
            cfgs.append({"label": f"level_hd{hd:g}_f{f:g}", "kind": "level", "heading": hd, "factor": f,
                         "test_leg": 1, "targets": [start.tolist(), end.tolist()], "legs": [gentle, prof]})
    return cfgs


def run_characterise(out, jobs, which=("vertical", "vertical_long", "horizontal", "oblique"), base=None,
                     name="characterise"):
    """Step 1 tests (base: cfg keys applied to every test, e.g. the settle tolerances; base["planner_over"] also
    sets the relaxed rule the oblique / level tests use)."""
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    base = dict(base or {})
    P = {**PLANNER_RELAXED, **(base.get("planner_over") or {})}
    cfgs = []
    if "vertical" in which:
        cfgs += characterise_cfgs()
    if "vertical_long" in which:
        cfgs += long_vertical_cfgs()
    if "horizontal" in which:
        cfgs += horizontal_cfgs()
    if "oblique" in which:
        cfgs += oblique_cfgs(P=P)
    t0 = time.time()
    rows = []
    with mp.get_context("spawn").Pool(jobs) as pool:
        for i, m in enumerate(pool.imap_unordered(_mission_job, [{**MISSION_DEFAULT, **base, **c} for c in cfgs])):
            c = m["cfg"]
            legs = m.get("legs", [])
            leg = next((r for r in legs if r["leg"] == c["test_leg"]), None)
            row = {k: c.get(k) for k in ("label", "kind", "elevation", "heading", "factor")}
            row["profile"] = c["legs"][c["test_leg"]]
            row["error"] = m.get("error")
            row["setup_ok"] = all(r["pass"] for r in legs if r["leg"] < c["test_leg"])
            if leg:
                row.update({"pass": leg["pass"], "failed": leg["failed"], "leg_time_s": leg["leg_time_s"],
                            "move_time_s": leg["move_time_s"], "settle_time_s": leg["settle_time_s"],
                            "peak_speed_m_s": leg["peak_speed_m_s"], "v_ref": leg["v"], "strict": leg["strict"],
                            "criteria": leg["criteria"], "net": leg["net"], "L": leg["L"]})
            else:
                row.update({"pass": False, "failed": ["not_flown"]})
            rows.append(row)
            if (i + 1) % 25 == 0 or i + 1 == len(cfgs):
                print(f"  {i + 1}/{len(cfgs)} ({time.time() - t0:.0f} s)", flush=True)
    rows.sort(key=lambda r: r["label"])
    with open(os.path.join(out, f"{name}.json"), "w") as f:
        json.dump({"mode": "mission_characterise", "wall_time_s": round(time.time() - t0, 1),
                   "criteria_limits": MISSION_CRIT, "settle_tol": base.get("tol") or MISSION_TOL,
                   "safety_criteria": MISSION_SAFETY_ROTOR if base.get("rotor_check") else MISSION_SAFETY,
                   "planner_relaxed": P, "rows": rows}, f, indent=1)
    plot_characterise(rows, os.path.join(out, f"{name}.png"))
    return rows


def plot_characterise(rows, path):
    grids = [("climb", None), ("descent", None)]
    hds = sorted({r["heading"] for r in rows if r["kind"] == "horizontal"})
    grids += [("horizontal", h) for h in hds]
    n_rows = (len(grids) + 1) // 2 + 1
    fig, axs = plt.subplots(n_rows, 2, figsize=(15, 5 * n_rows), squeeze=False)
    mk = {"control": "X", "motor90": "^", "clearance": "D", "net_low": "v", "separation": "s", "slack": "P",
          "box": "*", "net_below_rotors": "h", "not_settled": "x", "not_flown": "x"}
    for gi, (kind, hd) in enumerate(grids):
        ax = axs[gi // 2, gi % 2]
        rs = [r for r in rows if r["kind"] == kind and (hd is None or r["heading"] == hd)]
        if not rs:
            ax.axis("off")
            continue
        accs = sorted({r["profile"]["a_up"] for r in rs})
        for r in rs:
            y = accs.index(r["profile"]["a_up"])
            x = r["profile"]["v"]
            if r["pass"]:
                ax.plot(x, y, "o", color="tab:green" if r["strict"]["ok"] else "tab:olive", ms=8)
                ax.annotate(f"{r['leg_time_s']:.1f}", (x, y), xytext=(0, 7), textcoords="offset points", fontsize=7,
                            ha="center")
            else:
                ax.plot(x, y, mk.get(r["failed"][0], "x"), color="tab:red", ms=8)
        ax.set_yticks(range(len(accs))); ax.set_yticklabels([f"{a:g}" for a in accs])
        ax.set_xlabel("commanded peak speed [m/s]"); ax.set_ylabel("acceleration limit [m/s^2]")
        what = f"pure {kind}" if hd is None else f"level leg, heading {hd:g} deg"
        ax.set_title(f"{what}, 5 m: green pass (strict tracking also ok), olive pass (relaxed only), "
                     f"red first failure; label = leg time [s]", fontsize=9)
        for k, m_ in mk.items():
            ax.plot([], [], m_, color="tab:red", label=k)
        ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3)
        if kind == "descent":
            ax.axvline(2.0, color="k", ls=":", lw=0.8)
    for j, (kind, key) in enumerate((("oblique", "elevation"), ("level", "heading"))):
        ax = axs[-1, j]
        rs = [r for r in rows if r["kind"] == kind]
        groups = sorted({r[key] for r in rs})
        for gi, g in enumerate(groups):
            sub = sorted([r for r in rs if r[key] == g], key=lambda r: r["factor"])
            ax.plot([r["factor"] for r in sub], [r["leg_time_s"] if r["leg_time_s"] else np.nan for r in sub], "-o",
                    color=f"C{gi}", label=f"{key} {g:g} deg")
            for r in sub:
                if not r["pass"]:
                    ax.plot(r["factor"], r["leg_time_s"] or 0.5, "x", color="tab:red", ms=10)
                    ax.annotate(",".join(r["failed"]), (r["factor"], r["leg_time_s"] or 0.5), fontsize=6,
                                xytext=(3, -8), textcoords="offset points", color="tab:red")
        ax.set_xlabel("planner aggressiveness factor"); ax.set_ylabel("leg time (5 m) [s]")
        ax.set_title(f"{kind} legs with the relaxed rule (red x: fails)", fontsize=9)
        ax.legend(fontsize=7); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _json_np(o):
    """json.dump default: numpy scalars and arrays."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def _tolk(tol):
    """Hashable key of a tolerance set (mm)."""
    return tuple(round(1e3 * tol[k], 3) for k in ("pos_m", "net_off_m", "sag_m"))


def tol_from_mm(mm, sag_ratio=0.4):
    return {"pos_m": mm / 1e3, "net_off_m": mm / 1e3, "sag_m": sag_ratio * mm / 1e3}


def _ref_setup(old):
    """Factor, relaxed rule and missions of a reference study (its passing factor, relaxed planner, avg rule,
    its own tolerances; not its variants)."""
    f = old.get("factor_passing") or MISSION_FACTOR_FIRST
    P = {"climb_oblique_derate": 1.0, **old["planner_relaxed"]}
    tol = old.get("settle_tol")

    def main_run(c, planner):
        return (c["planner"] == planner and c["factor"] == (f if planner == "relaxed" else 1.0)
                and c["settle_mode"] == "avg" and c.get("label") != "ref_rule"
                and (not c.get("planner_over") or {**PLANNER_RELAXED, **c["planner_over"]} == old["planner_relaxed"])
                and (tol is None or c.get("tol") is None or _tolk(c["tol"]) == _tolk(tol)))
    rel = [m for m in old["missions"] if main_run(m["cfg"], "relaxed")]
    st = [m for m in old["missions"] if main_run(m["cfg"], "strict")]
    return f, P, rel, st


def run_mission_study(out, jobs, seeds=tuple(range(1, 11)), factors=None, n=10, base=None,
                      tol_sweep_mm=(10, 20, 30, 50, 100), settle_variants=None, compare_path="auto", strict=True,
                      ref_rule=True):
    """The 10 missions with the relaxed planner at several aggressiveness factors and the strict planner, under the
    settle tolerances and failing criteria of `base` (default: MISSION_TOL, net_below_rotors recorded only).
    The chosen factor is the fastest one (legs average speed) at which all legs pass. At that factor: a tolerance
    sweep (drone position = net offset = mm, sag = 0.4 x mm), and, when the first study's results are found
    (compare_path; default: the first study), that study's relaxed rule at its factor under this configuration
    (label "ref_rule": the effect of the configuration alone) and a leg-by-leg comparison. settle_variants (default: only with the first study's rules): the
    instantaneous and position-only settle rules."""
    import multiprocessing as mp
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    base = dict(base or {})
    tol0 = dict(base.get("tol") or MISSION_TOL)
    base["tol"] = tol0
    k0 = _tolk(tol0)
    first = k0 == _tolk(MISSION_TOL_STRICT) and bool(base.get("rotor_check"))
    if factors is None:
        factors = (0.8, 1.0, 1.2, 1.4, 1.6) if first else (0.8, 1.0, 1.1, 1.2, 1.3, 1.4)
    settle_variants = first if settle_variants is None else settle_variants
    if compare_path == "auto":
        compare_path = None if first else os.path.join(HERE, "out_official", "missions", "missions_study.json")
    old = None
    if compare_path and os.path.exists(compare_path):
        with open(compare_path) as f:
            old = json.load(f)
        ref_f, ref_P, _, _ = _ref_setup(old)
    t0 = time.time()
    res = {}

    def key(c):
        return (c["planner"], c["factor"], c["settle_mode"], _tolk(c["tol"]),
                "ref_rule" if c.get("label") == "ref_rule" else "", c["seed"])

    with mp.get_context("spawn").Pool(jobs) as pool:
        def batch(cfgs, label):
            print(f"  {label}: {len(cfgs)} missions ...", flush=True)
            for m in pool.imap_unordered(_mission_job, cfgs):
                res[key(m["cfg"])] = m
            print(f"    done ({time.time() - t0:.0f} s)", flush=True)
        mk = lambda **kw: {**MISSION_DEFAULT, **base, "n_targets": n, **kw}
        batch([mk(seed=s, planner="relaxed", factor=f) for f in factors for s in seeds] +
              ([mk(seed=s, planner="strict", factor=1.0) for s in seeds] if strict else []),
              "relaxed at every factor" + (", strict" if strict else ""))
        speeds = {}
        for f in factors:
            ms = [res[("relaxed", f, "avg", k0, "", s)] for s in seeds]
            if all(m["pass"] for m in ms):
                L = sum(m["total_distance_m"] for m in ms)
                speeds[f] = L / sum(m["total_leg_time_s"] for m in ms)
        f_ok = max(speeds, key=speeds.get) if speeds else None
        tol_keys = [k0]
        if f_ok is not None:
            extra = []
            for mm in tol_sweep_mm:
                tl = tol_from_mm(mm)
                if _tolk(tl) != k0:
                    tol_keys.append(_tolk(tl))
                    extra += [mk(seed=s, planner="relaxed", factor=f_ok, tol=tl) for s in seeds]
            if settle_variants:
                extra += [mk(seed=s, planner="relaxed", factor=f_ok, settle_mode=mode) for mode in ("inst", "pos")
                          for s in seeds]
            if old is not None and ref_rule:
                extra += [mk(seed=s, planner="relaxed", factor=ref_f, planner_over=ref_P, label="ref_rule")
                          for s in seeds]
            batch(extra, f"tolerance sweep / variants at factor {f_ok}")
            picks = [(seeds[0], "relaxed", f_ok)] + ([(seeds[0], "strict", 1.0)] if strict else [])
            pool.starmap(run_mission, [(mk(seed=s, planner=pl, factor=f), out, f"mission_seed{s}_{pl}_f{f:g}", False)
                                       for s, pl, f in picks])
    with open(os.path.join(out, "missions_study.json"), "w") as f:     # raw results first (kept if the analysis fails)
        json.dump({"mode": "mission_study", "partial": True, "missions": [m for m in res.values()]}, f,
                  default=_json_np)
    summary = mission_study_summary(res, seeds, factors, f_ok, k0, tol_keys, settle_variants,
                                    ref=(ref_f, ref_P) if old is not None else None)
    if f_ok is not None:
        summary["analysis"] = mission_study_analysis(res, seeds, f_ok, k0, old)
    with open(os.path.join(out, "missions_study.json"), "w") as f:
        json.dump({"mode": "mission_study", "wall_time_s": round(time.time() - t0, 1), "seeds": list(seeds),
                   "n_targets": n, "factors": list(factors), "factor_passing": f_ok,
                   "factor_rule": "fastest factor (legs average speed) with all legs passing",
                   "settle_tol": tol0, "safety_criteria": MISSION_SAFETY_ROTOR if base.get("rotor_check") else MISSION_SAFETY,
                   "criteria_limits": MISSION_CRIT,
                   "planner_relaxed": {**PLANNER_RELAXED, **(base.get("planner_over") or {})},
                   "planner_strict": PLANNER_STRICT, "compared_with": compare_path if old is not None else None,
                   "config": {k: base.get(k) for k in ("p_over", "spacing", "h_dn", "ff_path", "targets_from")},
                   "params_note": base.get("params_note"),
                   "summary": summary,
                   "targets": {s: res[("relaxed", factors[0], "avg", k0, "", s)]["targets_box"] for s in seeds
                               if ("relaxed", factors[0], "avg", k0, "", s) in res},
                   "missions": [m for m in res.values()]}, f, indent=1, default=_json_np)
    if f_ok is not None:
        plot_mission_scatter(res, seeds, f_ok, k0, os.path.join(out, "missions_scatter.png"), old)
        if len(tol_keys) > 1:
            plot_tolerance_curve(summary["tol_sweep"], os.path.join(out, "missions_tolerance_curve.png"))
        plot_departure_states(res, seeds, f_ok, k0, os.path.join(out, "missions_departure_states.png"))
    print(json.dumps({k: v for k, v in summary.items() if k != "analysis"}, indent=1)[:6000])
    return summary


def _agg_missions(ms):
    legs = [r for m in ms for r in m.get("legs", [])]
    n_all = sum(m.get("n_legs", 10) for m in ms)
    fails = {}
    for r in legs:
        for k in r["failed"]:
            fails[k] = fails.get(k, 0) + 1
    ok = [r for r in legs if r["leg_time_s"] is not None]
    tot_L = sum(r["L"] for r in ok)
    tot_T = sum(r["leg_time_s"] for r in ok)
    mv = sum(r["move_time_s"] for r in ok)
    st = sum(r["settle_time_s"] for r in ok)
    fs = [m["final_strict"]["extra_after_settled_s"] for m in ms if m.get("final_strict")]
    return {"legs": n_all, "legs_flown": len(legs), "legs_pass": sum(r["pass"] for r in legs),
            "missions_pass": sum(bool(m.get("pass")) for m in ms), "fail_causes": fails,
            "errors": [m.get("error") for m in ms if m.get("error")],
            "avg_speed_m_s": round(tot_L / tot_T, 3) if tot_T else None,
            "mean_leg_time_s": round(tot_T / len(ok), 3) if ok else None,
            "mean_move_s": round(mv / len(ok), 3) if ok else None,
            "mean_settle_s": round(st / len(ok), 3) if ok else None,
            "settle_share": round(st / tot_T, 4) if tot_T else None,
            "total_leg_time_s": round(tot_T, 2), "total_distance_m": round(tot_L, 2),
            "mission_time_s": [m.get("mission_time_s") for m in ms],
            "strict_tracking_ok_legs": sum(r["strict"]["ok"] for r in legs),
            "rotor_margin_negative_legs": sum(r["criteria"]["net_below_rotors"]["margin_min_m"] < 0 for r in legs),
            "rotor_margin_min_m": min((r["criteria"]["net_below_rotors"]["margin_min_m"] for r in legs), default=None),
            "shape": _agg_shape(legs),
            "final_strict_extra_s": {"n": len(fs), "not_reached": sum(x is None for x in fs),
                                     "mean": round(float(np.mean([x for x in fs if x is not None])), 3)
                                     if any(x is not None for x in fs) else None,
                                     "max": max((x for x in fs if x is not None), default=None)}}


def _agg_shape(legs):
    """Study-level shape / swing / reaction statistics (legs with the "shape" block)."""
    L = [r for r in legs if r.get("shape")]
    if not L:
        return None
    sh = lambda k: [r["shape"][k] for r in L if r["shape"].get(k) is not None]
    st = lambda x: {"mean": round(float(np.mean(x)), 2), "max": round(float(np.max(x)), 2)} if x else None
    out = {k: st(sh(k)) for k in ("rms_move_peak_mm", "max_move_peak_mm", "rms_at_settled_mm", "max_at_settled_mm",
                                  "rms_at_next_dep_mm", "centre_dev_at_stop_mm", "drone_err_move_max_mm",
                                  "tilt_move_max_deg", "drone_err_after_stop_max_mm", "tilt_after_stop_max_deg")}
    for thr in ("50mm", "25mm", "10mm"):
        x = [r["shape"]["centre_swing_within_s"][thr] for r in L]
        ok = [v for v in x if v is not None]
        out[f"centre_swing_within_{thr}_s"] = {"mean": round(float(np.mean(ok)), 3) if ok else None,
                                               "max": round(float(np.max(ok)), 3) if ok else None,
                                               "not_reached": sum(v is None for v in x)}
    out["sag_change_in_leg_mm"] = st([1e3 * (r["net"]["sag_range_m"][1] - r["net"]["sag_range_m"][0]) for r in L])
    out["net_offset_h_max_mm"] = st([r["net"]["offset_h_max_mm"] for r in L])
    out["tension_gf_min"] = round(min(r["net"]["tension_gf_min"] for r in L), 2)
    out["slack_max_s"] = max(r["criteria"]["slack"]["slack_max_s"] for r in L)
    out["rotor_margin_min_mm"] = round(1e3 * min(r["criteria"]["net_below_rotors"]["margin_min_m"] for r in L), 1)
    return out


def mission_study_summary(res, seeds, factors, f_ok, k0=None, tol_keys=(), settle_variants=True, ref=None):
    k0 = _tolk(MISSION_TOL_STRICT) if k0 is None else k0
    out = {"by_factor": [], "strict": None, "settle_rules": {}, "tol_sweep": [], "ref_rule_this_config": None}
    for f in factors:
        ms = [res[("relaxed", f, "avg", k0, "", s)] for s in seeds if ("relaxed", f, "avg", k0, "", s) in res]
        out["by_factor"].append({"factor": f, **_agg_missions(ms)})
    out["strict"] = _agg_missions([res[("strict", 1.0, "avg", k0, "", s)] for s in seeds
                                   if ("strict", 1.0, "avg", k0, "", s) in res])
    if f_ok is None:
        return out
    for tk in sorted(tol_keys):
        ms = [res[("relaxed", f_ok, "avg", tk, "", s)] for s in seeds if ("relaxed", f_ok, "avg", tk, "", s) in res]
        out["tol_sweep"].append({"tol_mm": {"pos": tk[0], "net_off": tk[1], "sag": tk[2]}, **_agg_missions(ms)})
    if settle_variants:
        for mode in ("avg", "inst", "pos"):
            ms = [res[("relaxed", f_ok, mode, k0, "", s)] for s in seeds if ("relaxed", f_ok, mode, k0, "", s) in res]
            out["settle_rules"][mode] = _agg_missions(ms)
    if ref is not None:
        ms = [res[("relaxed", ref[0], "avg", k0, "ref_rule", s)] for s in seeds
              if ("relaxed", ref[0], "avg", k0, "ref_rule", s) in res]
        if ms:
            out["ref_rule_this_config"] = {"factor": ref[0], "planner": ref[1], **_agg_missions(ms)}
    return out


DIST_BANDS = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 14)]


def _bands(legs):
    out = []
    for lo, hi in DIST_BANDS:
        b = [r for r in legs if lo <= r["L"] < hi and r["leg_time_s"] is not None]
        if not b:
            continue
        out.append({"band_m": [lo, hi], "n": len(b), "mean_leg_time_s": round(float(np.mean([r["leg_time_s"] for r in b])), 3),
                    "mean_move_s": round(float(np.mean([r["move_time_s"] for r in b])), 3),
                    "mean_settle_s": round(float(np.mean([r["settle_time_s"] for r in b])), 3),
                    "avg_speed_m_s": round(sum(r["L"] for r in b) / sum(r["leg_time_s"] for r in b), 3)})
    return out


def mission_study_analysis(res, seeds, f_ok, k0, old=None):
    """Departure states (do residual errors accumulate leg after leg?), final convergence to the strict rule,
    closest approach to every limit, tightest / slowest legs, distance bands, leg-by-leg comparison with the
    first study (same targets)."""
    ms = [res[("relaxed", f_ok, "avg", k0, "", s)] for s in seeds]
    legs = [dict(r, seed=m["cfg"]["seed"]) for m in ms for r in m["legs"]]
    A = {}
    # departure states
    keys = ("pos_err_mm", "drone_speed_mm_s", "net_off_mm", "net_off_avg_mm", "sag_dev_mm", "sag_dev_avg_mm",
            "net_swing_mm", "net_rel_speed_mm_s")
    dep = {}
    later = [r for r in legs if r["leg"] >= 1]
    for k in keys:
        x = np.array([abs(r["departure"][k]) for r in later])
        idx = np.array([r["leg"] for r in later], float)
        slope, icpt = np.polyfit(idx, x, 1)
        # slope standard error
        resid = x - (slope * idx + icpt)
        se = float(np.sqrt(resid.var(ddof=2) / ((idx - idx.mean()) ** 2).sum()))
        by_leg = [{"leg": int(i) + 1, "mean": round(float(np.mean([abs(r["departure"][k]) for r in legs if r["leg"] == i])), 2),
                   "max": round(float(np.max([abs(r["departure"][k]) for r in legs if r["leg"] == i])), 2)}
                  for i in sorted({r["leg"] for r in legs})]
        dep[k] = {"legs_2_to_10": {"mean": round(float(x.mean()), 2), "p95": round(float(np.percentile(x, 95)), 2),
                                   "max": round(float(x.max()), 2)},
                  "slope_per_leg": round(float(slope), 3), "slope_se": round(se, 3),
                  "first_half_mean": round(float(x[idx <= 5].mean()), 2), "second_half_mean": round(float(x[idx > 5].mean()), 2),
                  "by_leg_number": by_leg}
    A["departure_states"] = dep
    # transit tracking error vs leg number (does the inherited state make legs worse later in the mission?)
    trk = np.array([max(r["strict"]["ramp_max_mm"] or 0, r["strict"]["cruise_max_mm"] or 0) for r in legs])
    li = np.array([r["leg"] for r in legs], float)
    A["transit_error_vs_leg"] = {"slope_mm_per_leg": round(float(np.polyfit(li, trk, 1)[0]), 2),
                                 "by_leg_number": [{"leg": int(i) + 1, "mean_mm": round(float(trk[li == i].mean()), 1)}
                                                   for i in sorted(set(li))]}
    # final convergence to the strict rule
    fs = [dict(m["final_strict"] or {}, seed=m["cfg"]["seed"]) for m in ms]
    A["final_strict"] = fs
    # limits
    cr = lambda r: r["criteria"]
    lim = {"disc_min_mm": min(cr(r)["clearance"]["disc_min_mm"] for r in legs),
           "guard_min_mm": min(cr(r)["clearance"]["guard_min_mm"] for r in legs),
           "motor_max": max(cr(r)["motor90"]["motor_max"] for r in legs),
           "motor90_longest_s": max(cr(r)["motor90"]["longest_s"] for r in legs),
           "motor_saturated_s": max(cr(r)["motor90"]["saturated_s"] for r in legs),
           "tilt_max_deg": max(cr(r)["control"]["tilt_max_deg"] for r in legs),
           "alt_err_max_mm": max(cr(r)["control"]["alt_err_max_mm"] for r in legs),
           "net_lowest_m": min(cr(r)["net_low"]["net_lowest_m"] for r in legs),
           "box_min_dist_m": min(cr(r)["box"]["min_dist_m"] for r in legs),
           "separation_min_m": min(cr(r)["separation"]["min_m"] for r in legs),
           "slack_max_s": max(cr(r)["slack"]["slack_max_s"] for r in legs),
           "rotor_margin_min_m (recorded)": min(cr(r)["net_below_rotors"]["margin_min_m"] for r in legs),
           "rotor_margin_negative_legs": sum(cr(r)["net_below_rotors"]["margin_min_m"] < 0 for r in legs),
           "literal_rotor_margin_min_m (recorded)": min(cr(r)["net_below_rotors"]["literal_margin_min_m"] for r in legs)}
    A["limits"] = lim
    # relative margin per leg: the smallest of (value - limit) / limit over the failing criteria
    C = MISSION_CRIT

    def rel_margin(r):
        c = cr(r)
        ms_ = {"disc": (c["clearance"]["disc_min_mm"] / 1e3 - C["disc_m"]) / C["disc_m"],
               "guard": (c["clearance"]["guard_min_mm"] / 1e3 - C["guard_m"]) / C["guard_m"],
               "motor": (C["motor_frac"] - c["motor90"]["motor_max"]) / C["motor_frac"],
               "tilt": (C["tilt_deg"] - c["control"]["tilt_max_deg"]) / C["tilt_deg"],
               "alt_err": (C["alt_err_m"] - c["control"]["alt_err_max_mm"] / 1e3) / C["alt_err_m"],
               "net_low": (c["net_low"]["net_lowest_m"] - C["net_low_m"]) / C["net_low_m"],
               "separation": (c["separation"]["min_m"] - C["separation_m"]) / C["separation_m"]}
        k = min(ms_, key=ms_.get)
        return ms_[k], k
    tight = sorted(legs, key=lambda r: rel_margin(r)[0])[:5]
    brief = lambda r: {"seed": r["seed"], "leg": r["leg"] + 1, "L": round(r["L"], 2), "elevation_deg": r["elevation_deg"],
                       "leg_time_s": r["leg_time_s"], "move_time_s": r["move_time_s"], "settle_time_s": r["settle_time_s"],
                       "avg_speed_m_s": r["avg_speed_m_s"], "peak_speed_m_s": r["peak_speed_m_s"],
                       "disc_min_mm": cr(r)["clearance"]["disc_min_mm"], "guard_min_mm": cr(r)["clearance"]["guard_min_mm"],
                       "motor_max": cr(r)["motor90"]["motor_max"], "tilt_max_deg": cr(r)["control"]["tilt_max_deg"],
                       "rotor_margin_m": cr(r)["net_below_rotors"]["margin_min_m"]}
    A["tightest_legs"] = [{**brief(r), "binding": rel_margin(r)[1], "relative_margin": round(rel_margin(r)[0], 3)}
                          for r in tight]
    ok = [r for r in legs if r["leg_time_s"] is not None]
    A["slowest_legs_time"] = [brief(r) for r in sorted(ok, key=lambda r: -r["leg_time_s"])[:3]]
    A["slowest_legs_speed"] = [brief(r) for r in sorted(ok, key=lambda r: r["avg_speed_m_s"])[:3]]
    A["peak_speed_max_m_s"] = max(r["peak_speed_m_s"] for r in legs)
    A["net"] = {"offset_h_max_mm": max(r["net"]["offset_h_max_mm"] for r in legs),
                "sag_range_m": [min(r["net"]["sag_range_m"][0] for r in legs), max(r["net"]["sag_range_m"][1] for r in legs)],
                "tension_gf": [min(r["net"]["tension_gf_min"] for r in legs), max(r["net"]["tension_gf_max"] for r in legs)]}
    A["bands"] = _bands(legs)
    A["bands_strict"] = (_bands([r for s in seeds for r in res[("strict", 1.0, "avg", k0, "", s)]["legs"]])
                         if all(("strict", 1.0, "avg", k0, "", s) in res for s in seeds) else None)
    A["settle_binding"] = mission_settle_binding_note()
    A["elevation_bands"] = _elev_bands(legs)
    if old is not None:
        ref_f, ref_P, om, ost = _ref_setup(old)
        olegs = {(m["cfg"]["seed"], r["leg"]): r for m in om for r in m["legs"]}
        same_targets = all(res[("relaxed", f_ok, "avg", k0, "", s)]["targets_box"] ==
                           next(m["targets_box"] for m in om if m["cfg"]["seed"] == s) for s in seeds)
        pairs = [(r, olegs[(r["seed"], r["leg"])]) for r in ok if (r["seed"], r["leg"]) in olegs
                 and olegs[(r["seed"], r["leg"])]["leg_time_s"] is not None]
        d = lambda k: [a[k] - b[k] for a, b in pairs]
        A["vs_reference"] = {
            "reference_factor": ref_f, "same_targets": same_targets, "legs_compared": len(pairs),
            "reference": {"bands": _bands([r for r in olegs.values()]),
                          "elevation_bands": _elev_bands([r for r in olegs.values()]),
                          "strict": _agg_missions(ost) if ost else None,
                          "relaxed": _agg_missions(om),
                          "net": {"offset_h_max_mm": max(r["net"]["offset_h_max_mm"] for r in olegs.values()),
                                  "tension_gf_min": min(r["net"]["tension_gf_min"] for r in olegs.values())},
                          "mean_move_s": round(float(np.mean([b["move_time_s"] for _, b in pairs])), 3),
                          "mean_settle_s": round(float(np.mean([b["settle_time_s"] for _, b in pairs])), 3),
                          "mean_leg_time_s": round(float(np.mean([b["leg_time_s"] for _, b in pairs])), 3)},
            "delta_mean_s": {k: round(float(np.mean(d(k))), 3) for k in ("leg_time_s", "move_time_s", "settle_time_s")},
            "legs_faster": sum(x < 0 for x in d("leg_time_s")), "legs_slower": sum(x > 0 for x in d("leg_time_s")),
            "delta_leg_time_range_s": [round(min(d("leg_time_s")), 3), round(max(d("leg_time_s")), 3)]}
        fr = [res.get(("relaxed", ref_f, "avg", k0, "ref_rule", s)) for s in seeds]
        if all(fr):
            flegs = {(m["cfg"]["seed"], r["leg"]): r for m in fr for r in m["legs"]}
            p2 = [(k_, flegs[k_], olegs[k_]) for k_ in sorted(flegs) if k_ in olegs
                  and flegs[k_]["leg_time_s"] is not None and olegs[k_]["leg_time_s"] is not None]
            te = lambda r: max(r["strict"]["ramp_max_mm"] or 0, r["strict"]["cruise_max_mm"] or 0)
            A["vs_reference"]["same_planner_this_config"] = {
                "note": "reference study's relaxed rule and factor in this configuration: the configuration effect alone",
                "elevation_bands": _elev_bands(list(flegs.values())), "bands": _bands(list(flegs.values())),
                "legs_compared": len(p2), "legs_pass": sum(r["pass"] for r in flegs.values()),
                "delta_mean_s": {k: round(float(np.mean([a_[k] - b_[k] for _, a_, b_ in p2])), 3)
                                 for k in ("leg_time_s", "move_time_s", "settle_time_s")},
                "move_time_identical_legs": int(sum(abs(a_["move_time_s"] - b_["move_time_s"]) < 1e-6 for _, a_, b_ in p2)),
                "transit_error_delta_mm_mean": round(float(np.mean([te(a_) - te(b_) for _, a_, b_ in p2])), 2),
                "transit_error_delta_by_leg_mm": [
                    {"leg": i + 1, "mean": round(float(np.mean([te(a_) - te(b_) for k_, a_, b_ in p2 if k_[1] == i])), 2),
                     "max": round(float(np.max([te(a_) - te(b_) for k_, a_, b_ in p2 if k_[1] == i])), 2)}
                    for i in sorted({k_[1] for k_, _, _ in p2})]}
    return A


def _elev_bands(legs, lim=15.0):
    out = []
    for name, f in (("climb (el > +15 deg)", lambda e: e > lim), ("level (|el| <= 15 deg)", lambda e: abs(e) <= lim),
                    ("descent (el < -15 deg)", lambda e: e < -lim)):
        b = [r for r in legs if f(r["elevation_deg"]) and r["leg_time_s"] is not None]
        if b:
            out.append({"band": name, "n": len(b), "mean_L_m": round(float(np.mean([r["L"] for r in b])), 3),
                        "mean_leg_time_s": round(float(np.mean([r["leg_time_s"] for r in b])), 3),
                        "mean_move_s": round(float(np.mean([r["move_time_s"] for r in b])), 3),
                        "mean_settle_s": round(float(np.mean([r["settle_time_s"] for r in b])), 3),
                        "avg_speed_m_s": round(sum(r["L"] for r in b) / sum(r["leg_time_s"] for r in b), 3)})
    return out


def mission_settle_binding_note():
    return ("settle_time_s counts from the reference stop to the start of the 2 s window; the window cannot start "
            "before the reference stops")


def plot_mission_scatter(res, seeds, f_ok, k0, path, old=None):
    fig, axs = plt.subplots(1, 3, figsize=(17, 5.2))
    sets = [(f"relaxed (factor {f_ok:g}), current tolerances", [r for s in seeds for r in res[("relaxed", f_ok, "avg", k0, "", s)]["legs"]], "tab:blue"),
            ("strict (shape kept), current tolerances", [r for s in seeds for r in res.get(("strict", 1.0, "avg", k0, "", s), {}).get("legs", [])], "tab:gray")]
    if old is not None:
        ref_f, _, om, _ = _ref_setup(old)
        tl = old.get("settle_tol") or MISSION_TOL_STRICT
        sets.append((f"reference study: relaxed factor {ref_f:g}, {1e3 * tl['pos_m']:g} mm rules"
                     + (f", {old['params_note']}" if old.get("params_note") else ""),
                     [r for m in om for r in m["legs"]], "tab:orange"))
    for lab, legs, col in sets:
        legs = [r for r in legs if r["leg_time_s"] is not None]
        L = np.array([r["L"] for r in legs])
        T = np.array([r["leg_time_s"] for r in legs])
        el = np.array([r["elevation_deg"] for r in legs])
        axs[0].scatter(L, T, s=14, color=col, label=lab, alpha=0.75)
        axs[1].scatter(L, L / T, s=14, color=col, label=lab, alpha=0.75)
        axs[2].scatter(el, L / T, s=14, color=col, label=lab, alpha=0.75)
    axs[0].set_xlabel("leg distance [m]"); axs[0].set_ylabel("leg time (departure -> settled in shape) [s]")
    axs[1].set_xlabel("leg distance [m]"); axs[1].set_ylabel("average speed = distance / leg time [m/s]")
    axs[2].set_xlabel("leg elevation [deg] (+ climb)"); axs[2].set_ylabel("average speed [m/s]")
    for a in axs:
        a.grid(alpha=0.3); a.legend(fontsize=7)
    fig.suptitle(f"100 random legs (10 missions x 10 targets) in the 10 m box; settle tolerances "
                 f"{k0[0]:g} / {k0[1]:g} / {k0[2]:g} mm (drones / net offset / sag)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_tolerance_curve(rows, path):
    rows = sorted(rows, key=lambda r: r["tol_mm"]["pos"])
    x = np.array([r["tol_mm"]["pos"] for r in rows])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.8))
    axs[0].plot(x, [r["avg_speed_m_s"] for r in rows], "-o")
    for xi, r in zip(x, rows):
        axs[0].annotate(f"{r['avg_speed_m_s']:.3f}\n{r['legs_pass']}/{r['legs']} pass", (xi, r["avg_speed_m_s"]),
                        xytext=(0, 8), textcoords="offset points", fontsize=7, ha="center")
    axs[0].set_xscale("log"); axs[0].set_xticks(x); axs[0].set_xticklabels([f"{v:g}" for v in x])
    axs[0].set_xlabel("settle tolerance: drone position = net offset [mm] (sag: 0.4 x)")
    axs[0].set_ylabel("average speed, 100 legs [m/s]")
    axs[1].plot(x, [r["mean_move_s"] for r in rows], "-o", label="moving (reference running)")
    axs[1].plot(x, [r["mean_settle_s"] for r in rows], "-o", label="settling (reference stop -> 2 s window)")
    axs[1].plot(x, [r["mean_leg_time_s"] for r in rows], "-o", label="leg time")
    axs[1].set_xscale("log"); axs[1].set_xticks(x); axs[1].set_xticklabels([f"{v:g}" for v in x])
    axs[1].set_xlabel("settle tolerance [mm]"); axs[1].set_ylabel("mean per leg [s]")
    for a in axs:
        a.grid(alpha=0.3)
    axs[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_departure_states(res, seeds, f_ok, k0, path):
    legs = [r for s in seeds for r in res[("relaxed", f_ok, "avg", k0, "", s)]["legs"]]
    keys = [("pos_err_mm", "largest drone error from the leg's start [mm]"),
            ("net_off_avg_mm", "net-centre horizontal offset, 0.5 s mean [mm]"),
            ("sag_dev_avg_mm", "|sag change|, 0.5 s mean [mm]"),
            ("net_swing_mm", "net-centre swing over the last 1 s [mm]")]
    fig, axs = plt.subplots(1, 4, figsize=(18, 4.5))
    for ax, (k, lab) in zip(axs, keys):
        idx = np.array([r["leg"] + 1 for r in legs])
        y = np.array([abs(r["departure"][k]) for r in legs])
        ax.scatter(idx + np.random.default_rng(0).uniform(-0.15, 0.15, len(idx)), y, s=10, alpha=0.6)
        ax.plot(sorted(set(idx)), [y[idx == i].mean() for i in sorted(set(idx))], "-o", color="k", ms=4, label="mean")
        ax.set_xlabel("leg number (state at its departure)"); ax.set_title(lab, fontsize=9); ax.grid(alpha=0.3)
        ax.set_xticks(range(1, 11))
    axs[0].legend(fontsize=8)
    fig.suptitle(f"Departure states, relaxed planner factor {f_ok:g}, tolerances {k0[0]:g} / {k0[1]:g} / {k0[2]:g} mm "
                 f"(leg 1 departs from the takeoff hold)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- embedded guard model (data)
# Made by encode_guard(guard_points_from_stl()) from cf21B_prop-guards.stl (see GUARD_STL above);
# check with check_guard_model().
_GUARD_B64 = (
    "eNrtmnd8V8W2tzNrZkB6770ECDU0IRB6B+kioXdEkYCiXJogRURFOgiIIL0XQem9hd4JJUDo0ruUmVmb97sD51xUVMBz3r9u1geS"
    "/HaZmdXnmVTkNVSSt1JujqRqvJxK8WbKxGepMc+k/HyQKuJ6HV5MKfhazH2BHEUteAo+W0TJ+AYV43j8K2XmeryQIshRKDfhGTFP"
    "BrBiR2/yzpj7JTNl5VP0AzVn55RszlNpJcWXRfi2u+0SyzK8kRrg57Y8keLzfdpPqeQMuuxSy+SyIc8hzZba8Pd0waWXwbyfgvmc"
    "q89nXCs2bg0lkPsph7zjEsnKvIoWU04Z6XLLAnwA67nh/J9r84/+Gl0tPuy2UWq8O7W86NLJ0ryJIilY/kSF5H4XLIP4KJXg864x"
    "X3S7XRFZk5dSDd7lcuPfTsomZ1NmedJll0U5gk67rDKMZ1MYn3JLsdIQGeWOUog8RiXkNudrc4cL5OpcVG51kym73OGycxF5EHOe"
    "TGupI0e5My5QhvA2yiAH0S8upezGX9JA7kJjuTVN4WY0jVvSBO5IgzmO7EfsOvNtmkhpuS4fpsK8jCrxasrOJ6g1tPYAOurN/aCz"
    "EXhiMp6chjdMxpuG441lOFCepPxyC5WX69ybvIsqyBOuBCeXq10Yb3ejsP7OtIyPub5cg6ZyBzrv2nFhvgVrdKCf8dkcHgWbL+TB"
    "8IyePBp+cMxdovq4Fgj5iXPIvdDEIdeIatBWV58OuH5k3ST6imZhVXW5M3+CdeWRKWUqror3hdFOeMwstwGWC5O+3zWW+eVDF4pR"
    "SsuCXEpOda35AX3MBrbazH34G9rCw7gYFaN1dIJq0UgqJadgPqupJtY5gVvzZ7TBVeKR1I2+cVGuL33A4ZQL1mlJH1EP6KwI78ao"
    "Y/gT2ZI+wDPtsIqPZWHOy+vcQzeehvEQ7kSD3HqXl9vSCZdDToOtTrhJ0GIod+LRWMdiGkzdOYyn0/uUVs7l2vCQD7gO/LkXD4Sv"
    "T+d99Cn0auBDragC9P4F7aAqsosbKYvKRFyLF9BI+KcfGZdpj6tAc2gTV6AJFCivIKbe5GS8md9zy2GRY+4OzeUx3JCXcF9YJFiO"
    "ppW80hXEXHYS03FXA/4+j2LLX91JRFFSvufqyZz4LZ78hKtQfVeLlvIgXgFL7qE91JMmcVNaLJPI4dSbCR6/iBfL3Fwb9nqXItxx"
    "vu5GUBp+l49j7pWoD/vxnxdam+bqy6+h6VxymrvpHuHpodQD/vQ2fcb54Y/rXXN5hL6lb/l4TF5YSLvpSyrmAmkYBVLZmLg6gKf3"
    "yB4yHLkiH++RHyGKz1Az7s9dqaXMBT0toNLw6m5c0JXH75Og+/eRT8bzh3SEjlF57ufm8EewQVf63PWV610PWPcY/HCUewvZ5gJi"
    "Khe35zychlvAk2vxXZdQ3qXVyGyN6BD0PIgbIAeE483L4YPDuDqlkN8jklLINPI7mum+5Zw8WKZ16+C5G7i7y8eHkB1+davdFVmG"
    "KskrsMggRFwUbFFBfsbs8slU7nvpZ6wZdAa5aT1irr2LxhxzyNPOz2obYKcIt5EPuuI0XLagUPgMu9L8P7yUhczCzVx5zHs8kZwA"
    "u0+UvWHPQxztfM3Ppw6Ioyr80BXnnLIuOZec67mfXRY5hrMhm6aRlZjtUuhqO6xfO+azvHyazpBWodSWrdVqNW10q7kh1cdYW+g8"
    "ZjqZdtETUrIErZQV3bc0kOohBte4xNyTUnNuWCsHl5T94WPTeb4bTKVda3nORdA514Z+jqkCZWSQ/Jy/d9M5EyXn+nI16sIZOoUc"
    "0YdK0Fm7WSaSF1y03RZj96OU3x2U4S6L6uNW01IXwVnUKfhNUvariT9nv+IU5/YULlvxZcrhsvIJZPfWrjNtRwU6SQNkX0T1J+6o"
    "HOouUHVezPdhm6aIrb50nce4XvIHLsKlOB1/jjzdFBnitNsNX7qMSnDD7eSKbj2y8yX3E33FEfY896AEnIh6yeMuPbwmmjJwG0RV"
    "dhmqJsjvXCkqpbbYlS6FZIx2zSV0d+UC+ppTyWZ0Q9bgosieeTgfcs8BTk3tuRJiMB1WsIIq0gZ+CxnuFtXm7xHJs2V8VxG5vjGH"
    "ueqyFeY6RX6KyDqCcbMgt11zd1EZE8lTbivPkeu4ODJHf/qe21A4pab9fJQry1KUUuaGtz1G3s8sA+RUWO97RH6UCyVPzrOSj7h5"
    "MZbPCsvvpnu2kTqJyBpKM/HTJTcQNr1MCeUsqos4aIrYW8K5kB3K8zrknySqisvjytAq5JYkKgOfR57bgog4QyncNdkcEXnPpuVL"
    "iPWWyP0r5QD3k9vhHrvY8iCqZku5kkcitkbD7jmQK3JSOvkeLFWEslBz9wMto/QcCV8RMid8fZmr5gK4Gm/APIZxcTnShmPNm/G5"
    "3134Y6fkEI7Ln6ik7hP3le2mLtl7LgtHUwHMIAe03F/GVxfY8k53yDaVB1FpIymc0/Je9AD7cdcXiJYPkYMO2Xzqnq0iP3SXXGvK"
    "LXOr6ryXfLv5vUNVzgSdt5HD5TIuA+1dQy0s6LZSMXneGXrkBrtLciwiOzP0+YnVsO37dN114srsa/KKa+2m8JesUdFjSf+Tm7Se"
    "1tiRagIy/BfcnS5xOH6rqJLyTYqyFd0gmZyvUxLOyOVVNlph11IB1y3GKlcpFV+kKfInmcY9lBn5HB2Qid0q6cfYOrqButHI1le3"
    "ZUKVXZWDvRKj/q+jebIcMtwvdBc5baHKqibaOqquXaRaKb+LK8ABLkB1dqusP+pxaoFPv7PJuDSvtnWctb4GEnASeGda/hQaLuqO"
    "2JG2tPXfWBLd0F4p3Xe2qkvA9ygOP0Te3a7SqT4qxEaoQzKPi6I18qwN5GK8A952kYqoo/IB8nBJ7qc+tYHuUztQ+Vn5e1VSXaaP"
    "bFqXmv33vIE+565sZ4vzdmgsl91h46iEfJdiob6fj9HjY7pHJ9UvNouNVqXVVOWvNgr9R2M7US20voeHqnaKY95QlPfYjFzTejKp"
    "83vDQPR6rVULG02TVRK1Tu2k1CoLr1Nl7EMqhIqazE6zC+xl20H56/fXGpvbq35W8BNaaO+qZWq7TWTvKL9DiIvIDLaDlJ+VJHQ4"
    "xAaqajY2PyaPMqvVqgAq/GFVRQ2zSXirPUzafq4COEDWUJ4abP3e19EANVT5awrgdvaJETqv9Tvg21QUtvLvlVzDvmtzoHfz/Tsv"
    "vH6Zfd92tUH2qJpg/UjOw0dIyEgVqY7RfPXA/BDzvnj6PeX3z7lQZ9/Gah6YuHoOnvCvZbCZEbce3TIb1SxlnJaJOZ3tY6+pPtbX"
    "tP/cQGVdcp3cXqJa9itF7MX4IfEVk0rXsrHkTeW/45bydXGaQuwG5ev9nEnFVzDTDaqUfeIy62D7yL0hPax8szpnMmr/mcIxKzys"
    "UHpslPGv+1kip37onphr0Hhq6DzSZLcnVR79hvZ9PbN9ZN7QfnaJK5W+oA6YtLjngSuoD5i48o4J1n5O8PcX/mxumATWt4zf2yTS"
    "e80tl0Sm02cRr8V0Rn3L+N4SYUJ0dj5t4siU+qTxa08m1KEt5oTJr30fPEibXR5dSJfQIfqQqSeDZXFuzL2QvVqiKpZFrv3BjUZN"
    "aymt2MqfugHylLjCq0QCr7GbKGbKgl4P0cALcRGyp9dCTPJSu8vykX0DMXjcllSr7TtqMmwy2A5QXewE1dAuURXsdlXQHlNZ7FWV"
    "0N5ST4zSt00snUZfMIH6hDlstppDpoTOoyNNDr0HP2fUx0xqfcok1edNPH3ZaH3T3DNPlLKP1SMV28bVwibW1qTQ9006XMuMewL1"
    "OZNXF9RRZqd5E9oroA+bfHh7DoySHuMl1r4PpsQcjqv0iOPcdhks2dROVx/ZocrPtD/Ylmq5raYibIi6aNMpOJnM5k7JSu47r7UY"
    "JCJkezdP7BP9vJlysGvj3RNlvVSU1RsgS5LHLaC3M9wPPepa15LXom+bjF4xFDucaMzdzyTGtIfVlnBRvcx74FZ4sfVjs9lUFRvM"
    "cUrEFUVsfd0E6V0mnuxkR6nPbS+1gsrqu7TF5NSpdV5Xx3VyN9wTKqereh29YV4OfdZU0Ol0Cp1Q3zXXzEVzVn0oRosV4jF2gen4"
    "FAd42b38OpfOoqPNMXPQnBQBlI2qUEeKbzeYe2qvCrOzY9adlb8WWmfQdbwy6MCHY9Z1aY/4TH1sN5pSeos5Yx6ZBDqJzqY3mAgz"
    "1tvNHWRP/poa8Y8uCXVyo5AVj8jr9pAtr5bajshWYTapfQDNXzN3zDnjx+goGc+Lphtioh2jNiul78DG+bW05axfqxK5O7KOrMrv"
    "81eUGDuHYDlXjHCdZS23RPrdRxJV3maG1YVNrq/CxoGwcX7Yl00AfD83rB1lcutAHYyrKXHXE1XS7vSWyK5eAW+I2+e6UG5qJo6r"
    "M/C3baa0XmUe2LhqgZvNd3k+bfceqLi2uCik2qqjKtBewfxS2tUx/nBQZkGNTo+136Pc+rHaZ8oijxcUU72y2sEnT5nmorxeYW6Z"
    "WNzf9kZumwIbLfW+Rd3rLVd5yfQNk0xXFtcR+1V0fO4hmnh9vHhaw4N/Nf3FDKGwn99DmxyLnqiU1/iYSOwV9Yq6XfKuXYkMEW0K"
    "6UyIl5N4zzlzGeu7ac7j03yI4yKQo+a0yarHq/a2q2rpdojW3m2RAl1oHTmW48v2No+urvebDea6qy52mf1eb1UL0TDYStvbxkdl"
    "tlRAj1OjVS1x0NPamPwivr5K3e2PZqG557T+1V02601y1JDZXrCorWPhnjDR0q7ydpie9iZt8vqpDIjVd3RpsdB7SNcQwXfdfFNS"
    "J5QNdDP7qXpDFhaHvEa6qx2mZpnpKg7GfEeU0UPVAHjiIxeL0+mD3gMTHxE9A+MnkifNfG8sVpQCHvS2SMpNdS1xk6bDxquR9x5T"
    "C73Ku+92eDn1cTPVFBPT1XuI4+tmh3cF3WKwqGOr6+9NC90GdWG2VwDeEiaS6o9tTWSkr1Q7vc9EU33dSPdFDZjs5cGedDP2L515"
    "AA1F73nELUB3VpZWuL3cUGaN6bADqLtM6J0R3V1+b6zcIGp577hJoov3mVgmS7jD6F/uSM8mUtE2t9pmK6s5iKsRdhP8MFLlQt5J"
    "bp2KY++ahNBUeu1bbJcpqv1stcJU1YtMXf2TeUtPMz+bWaaJnmAa6TnQ6GRTD9ea6Pmmml5pKuoliPnNyJZrTHHkygJ6GzRSBG8p"
    "jigtpHfj6ioTqjeZmnqxqYw3N9SzTJheYN7Wk0xLPRtvnWAmmMWmDj6rr9fi/m14m181/Fx5ymTT/gz9quRXnEz2rCps96gP1Le2"
    "gVqMDL/LZkF2VIqwH8zodspQ10v08RbI8aKRa+GtFqW9obKHixIZPCs87iK/dRnoJNeUsXiVK0nrsPeeDEbThyaBMTXh9dB0Ntle"
    "vClOmjx6iHfQi1BDVFcbTybh8aj3nVVx+66uKeaq8ca6APT+a7w0+hfTUdfWFcUhaomaUVQ/cY3tMu+cvEMd9XgzwivkddIddOMY"
    "nWZ3W2QFN0dMl23cFTEQ3Xsir42c7oJoH5eXQxH7Obkm19BlUScGijpeTWTHMDHRTIeOSupaeEeYnmp8fcSxD1Vb28N2VmNtPWS5"
    "ELXbZkKvJtwTrCCrXGe/svPMSDPKrDDHEenLzbt6lOmkw71dYiGvge0mwwpV9EjTQU80rfV001RvNGX0VLXI+5KuuszeAtHKm+ZF"
    "eNdsBYuagfqcT3fW78C+u814vDOnTmN/USHipqgDTc5ziVF39kPjw8UW00Y3tS05mE5jV7OCdoEetJFfoXq/76bJ6m4T8vNZ+USe"
    "sAVVLTXNLlShdgu88pi6FtM/+T4wF97WQs+EHQ6ZS26bG0cDuBt95sV3mdWHaiAy3VwTrlvDny6ZtPoL9aVKho5I0z0OR783judA"
    "22vMWqxovamuf8Tb5pgksrv4AL1QdmijNPL0dvip7/nbVTm7XqXDnmgRL0VF2W6KYQ5jTGNdQtTXlXH3XBNi88o6bq00coOdYbvA"
    "QmvNNLPPVMIcg+HhTRAvad1e2cdbKMeJMLdWZPQ6yQy0iU6heqwGrRnvPpYpwDDGymDU8eqo8kPUDLUTnchp9asajRoywwwzvmXm"
    "qtE2lmog/seNcye5OCclxwdFqNcd/cxiW8LVxg5Ng5Pcp8YUhedq63CdV3+oe+CeoSAe5eg7Vm44LFpcN8G1t3VNfUidt028aHHN"
    "XaRGOqeur+eZn0yo6oW9cAfxo9fMJkbuK6v9WGwNv31LJ0QXnhXd/VUZh0eI4d5Q85Hebid54Cn6KHYmJcFD1mDnWA17nEgQg0zI"
    "WZtBmvODR84Ej64Tc92/7xpdA4drAbYQBRaVDDxxUQyhrocdy68QBXYbwP6TM5D1toDcvglS7dDrS1SvKSDULdHznsb35qykkvfd"
    "SroNnlAGu8XEMrG87eZhVxIfzHUidoP7sTdsCELnE+XUsg1IgiWLfWYwp5fpwU32gwstIAMCfA+71BP4vTJ2kolASI67xditFWDs"
    "T0EpNoMZ1I75OdJF0mG3BCR3G/7NAMcozelkOtC6SNoLhhHEQRyM3vU8rl90M6kmYriILCJ3uZ/h+5HwgJ0Yczb2SEU5u8wORh3G"
    "WWVWUIbZIFFRYNUR+O6T6vLQZglZUmI3B0YdyEvoJOZ5kra7ZtyK84PH+ow6hMeAYQTKQNmFr7puPAjcKY28i718XBCXXPCYGlh/"
    "B+7KA0Bjv+ahPBbMbiIIyBSexgtAa5pSU2oBaQt5nzpRV/oEjPZL+oamgcptBuU5h52kn+GiXFGQspO0zvl2KSfLgbSvAp14D/ds"
    "c1Ox/lFgDMVg26OuOpjwEJCEReB4delt+hjkfxl0nxYrSozxa1ET6gLmiHVhBYfw03HXBASkOr1Fn2PPkxvc8j3+hKdilks4DKS6"
    "AJ4sJKPdVfcp1rCEl8HXDM12J1wp3BMBdtxINsJZRTbs8qe6zaCO0+BdW6DxsVwWRPlL0OoVHIp1BcjOPAJUbbMbxLuwy8sB3TSj"
    "3jQdVK4t1r6MIh1ogysCHxuMsRpDO5NoLnZ8KWVZ2RUyxMXmgvw2d+FxvA0d9s5nXPoL/B6G+Q8Fzz1M5bBzDgX5qsLvQY/nIDmk"
    "HyMNYJF2YKFd+WNY4QPw717Uj74AZZ5Avo//RBtpBWjvp9wET7zJVWkqOodwULU3sbPvAPtNRh+/A4RoODhJlEuG6CkNWy3EzC9T"
    "kCws33OHcHKwkyrKuKBui8F30sqNLinIajgNojdB1avxKg5Cx12ftCwvF7hj7jgo9a8g7eE0hFaBWvveuc/lAp3uhHdMgbYXyAXI"
    "e5HulnvgCvMkrgutDsC8h2O1teHbj3Au1BFnCZPAmqqD+49AxtsHEtUKMg0j5IUm6sMqu6GbcOhqDw9GbhjAw0E8j7p1iO8jMZbr"
    "gdF+8k8HQKceuSD+kKN4GLxxh9whC+OcJj8qSQ9YZhK4wh0q6Na5005yAvhMPeimMbWiMdAhYhsMLx9iZTp0PBX6qSf7uc+xojCM"
    "OYh7g9YcA+XJJ/tCcoGQdeCB4PIpQGzLyvpyLRh1bewk4si4srQc6Ra5Bji36Q7rzgZNXI333yI/H8TjjvDysoiiERhlBchQd5eT"
    "8/EGkPB1YDyX5CVw6HsuI+ZXDh4Vic9i4ZM2Lj4n57xUhz4kPxtmkBlkJXDpN0Af8yGbrecy+CyznADp4rY4dl3A2qeDWR8HtU4s"
    "Q2Qzd9k9cUsxq010Cbu7aHcNNvYzYT4qBX8eTz1kT9SH4244dPAjb0FPER+nX9EuGzR1HbNfCDkEi2ZD1IzDOVZhWVOSImXxdRyk"
    "tyvvx8nMaRDNL2kUNPgTVXRVXHf+FrYrQVUR1ZsRYacpi9wErwhBPIznSPhiT3jFt+6RO8GVaDnWVhbUrDp/AL+9D49uIjdBesIP"
    "4oFml+e+vAkcObeMRj1b5C64Oy4TsmhJnD00xLNnKJPKhN7VryrZ8HklZKzp5JPrXLBLNfkBZCTkICSH87n1j+iTt1NrzKcR5vaE"
    "coFYjnCVOSWVhd22I4fWkhNcC3j1VFDqppyc+iJyToE1B4G5/+g2gUxWQIzOQnYoSiHUnPojtjYjeyXHKdFkZIb0OPe5g7O3HrD3"
    "VgpRIWqz3WJPubjgdj49rgF/jabzYGZSppDJYDs/38/i+6Cgm+gCtZN3IUfdWXhzO5yOVZXfQOK7MPcZiFkSvGM+/CeCU2Dss8i+"
    "llq4Za48eEhyePdaKox3pAd57uvWYgd5xv2CiEzFhaDL3oirtZATkGBqCFY9BkTa1+IFMMkEqJA5obWqktwq9w7OxRaAWGfjasj7"
    "SWQD1UDNtDNtJKhgesTgZuzUT0OD0YiJBCqB2otc24xn4pSvFIjMQuRyPyPcwj4mA/JcKGb3JXjVSrlSHoHcs91cWugjF6y7h67J"
    "a6iRl0CnM0A/OTDPhVwPMdMDpxbbwMWSyhzwvvqyuQuH1x1GBumKPISzC/mRrIZzx0qoJRUQKcuxjmKyuiwFrz0BXaWAXw1ATByh"
    "cBWOPuore8kmRY+YEhXhDCxwHhH+GGy6vzxk87sJbjmYdVFkj1E8BGdeRxE3zdBzb8cpopB1ZSROFq7Avk8Qm36X4FutCqrQA9Dp"
    "/T6rjtFDPshAxFoqeKNvnYIumGvCEzZSWvc+Tl/n4zR1FfziKM5xglBlUiNemqNWZUUf0F1+rb5Wl5EN7uKUoxOiKpcMt+EWjBq+"
    "mxJjVuKDnBh59TL59LoiWHVy2BWaUIFqhW2Lu4bj5Lcr3nqc98sDMo5L497C+y7SFax1ipyCqlMST9ySt+RoedLetYlxhhqKenGD"
    "NsqNchb4YwOcon0n69q6drJdbBPhtCc3TtN+IZ9btwVp8Pl0cmjwAAWoAPWuSwhKHUkWjDoYFPoGVVKV1Eh7xPqEOifqzCXkwnvg"
    "iJvUJiVdIbcJq08AO7cDEw6xIbY/zimKgTs/hPgVcY3f1eB/P7J32yDns+re1ifUceEvW5GXH4DSDlDZUX0uoedMjf+Hgtq2swnR"
    "Vz2iE/KEfGhjcUJQ/GKqmDoGgV+BTje3m6zmxOxr7x4sn0alwd7IJ9RR6JFnqpkqAepHcUbrqPbYXeQT6qLgnUvVUhDqSfYAYqU8"
    "3pwM5Pey3UllrMJP42L4dBpYZy9W/5CaqWbqBqSfjQPynAh0Oi7mmA3nXE+Qb4qr4qoeJBNOO/er/eDTUSDU15G1P1Yfq0o2D8ju"
    "MPCTTGDVB2irvUW9bADyrFUWBPyJy8/dVXfQP43O94l5Ynw+7a8nQH6nvkPXHwCaukvtUj6jZvpZ/ayCwKa7Wp9N58F7oiEPqLqq"
    "rsZCcoMKC/m2PeafH3JcHVenht8+APl6247Dp8fJ76MDMP/2qj0I83l1XjWyxuXALJOArPWxDvbPjDH7q/7qIiQTJ7cWu45ulsCH"
    "l6gl6oq5YmpZf46IZJBqXyOP3RlEYFlo/jRFqAiVijNin1UKz0SD4KdCvJ0DfysFLu7RG/IN6a9e8h61B5bMxI+QKZxyKlDn0I9B"
    "qePII8Yn1NntLzF+eg0s+hF2v2nZ76vOqDPKgUtr5JsC2mcqaTmu3G8y4P9EGoL5+LuLX+Ff99Q9lUwn0+n5lisETnXTnQMhY3cW"
    "s4jAju82TgJOm5N0FWMGcXadXR8Eo46DHUMmGapDsRs8ZNag48iDvWUpGQSutxXcbgfI3z5zANcO4+6nEo31XYBcBBm7am6a2zFy"
    "H2zNGt+uwsaysW1CiO+5BW0R7K6r24a2le2M3Xx/OwwWnWt/tlvBIU/bW/aRjedSuyAX4sq6xq656+Y+dePdD6gBfu5tjG6yF7LP"
    "OHTGW5EbrvBDTuCl9Qp6xb0GXlOvJ2SUN8lrLlqL7qDGY8U8sRLU+Ki4J4xIhTPNksjrLakd9YGAEuPkrCxOEVvKAZCZkAjIKcgj"
    "7HyTqZQqvyqsqkFaQrpB/AidDlkPOQyJhlyFPFSPIU+U0ApsMZ6OrxNr37OS6pTgx+khGUGCs8PSOcFLc2P/+lQKYc9aEleicN6V"
    "QCfQi0AsNe8ySzg2SNNjs8KrKuLK2Po4aPR18NbK4g6Vw84ii+6leqlRkA1mBV02N1wanLBIjJcB9ltvntBFUOm7IIiP4dlPmfVT"
    "Pp1UJQS1Tgem6bOCg2DT0eYprX5KqLOAWecH9SihN3tZ+a7MH7P22ZCyOsDOFvkolvbv2Qo/OgVLd7TdbVv+DJ3GFq7mtRFXRV38"
    "3UB2mSCGDJfSpcEOfG7qR2NScOk3bWUbZluCKiyy62wecPV5TmF2teUoSFx47UFzw3xkv7AbbT6Xy4vrx7qsqPZB7kP2P/PASDDn"
    "KPMLvO4qvEyAT2eGdx1D9qzharmv0Q3tc/tdVW6LbvwD73MRIT7GDisWuoNgsO7OkCWQDMrAagSKn1O/iTUdwHkBuaXuGs5cv0Z2"
    "PagOqoywWm6sIq7d7hUXD1QR28RG24s2I/6qQME3VkOUDgIh7oAOv6oex0UFGDXOQ+4bZfPxVPhjSXLqLkgr7Kl7qJ5qOYhwbU6u"
    "h2P0VGDUq7zKorJIBsoY3+9acZ7kR5DPqn1G7XtUHlhrvdkFe52GREPOY/V+fkiHmCqKSFmGWSf20nhFvNJeM2+M2CF8Ep2WisdQ"
    "huJyEKSr+lyNV/Fj/DINfCAHZllCLzPfILvHl9dxdhGMU4GzyIp9bVL+wetl93vjVQ0wzgPG4jR6mLpkjDno5cec8okQ3Ron5Dep"
    "tg4TO0B/k3M5vRDXn8DvZnthIpbuZqVGb6LmYrYd7CavNDhyBozbQM/zKon5JqGMx/Pw/xD1jSoFNptOJ+XpMTXnG9UIhO4eRp2h"
    "OuG8ryzWn0fcNwc8f/SbFF+H6Y3eNdPStrHzvbdFCpDCh+4YuOstV08I2VRPMY8pCP8fx8yKiWQ4C+xp+6Mzme8l0m3w+R14WRLU"
    "hJRyimqDfOrPuICuIGpafzY1cL473ZtlovHXNAtwPhsYw24nmO/MRPO9mWymgHxPA6mcaWabOWaemQ9ev8gsMUvBmZaDQq4yPo9b"
    "j1Omzcid2yF74bUnkS+vYmRryMa3yeGzuWxhO9AOthPRpyy3q+xB9By3IPFcPOTBIFcO0gLSHTIGshSyB55UhEtgd94YO0f8LRRo"
    "wxxeiW76EJ9H1xXgJUZWzOuFeNW9+t67Xhfvc6+j6CG+FN+KmeInsUHsFdHihngsYiMnZqeC2DXUpmbo1bojRsbgL4B+RI9QUBaU"
    "tSF+rHwNmQvxu6toyK+QZCqZygkpC2kMWac2qgPqiLqgLoPwPVaxdBxkmtQ4uwlE/ORBVimg/agoqovh/LMkJFSX0+UhFUFmq+Kc"
    "pIZ+C1JT19X1QF7fgYUbxUhjsFH/rPIITkCOgrn3UB3tB/YtcE5L03HG2VM9cJdMG1Tsd8Gef0RcrPEqik3oJtLojnqL+9yrI741"
    "zWw6nU+PMXew5wz1x4HeLvJAORA+XwKZqjJmUQ1j1wUtrI8xm+pmurluhTd+oDtBSnq9vVU4d6oC73kX1H4pn+Yc3gTvkDgnElAQ"
    "BakgVQEyBCdsR9UJ9E63cSIvUJuT6syQkqD+mW01ewGk632u67UTs8Qy0Yp8jlcBq26n6+CccQX68kbeVpwclwB9ag1ZCxmsimMW"
    "8+Bjy+FZW81ueFEXexKSGvUyP8cBuQ4RLei+vC/9zq0IziwWwxuXmdXwQr92HzATvdI0V40wP8BzZ0L86ytjPHS72QPtHjdnQZcf"
    "4Jw7BbyyCE7ux0OWQPaCswvXBTIQMgNZtSd8qQFmPMKMBjleh6x0y+SyDZCzk4FI78A5emKVGLx3i3pqRZ/LDkOM9OW5qNk9vBtI"
    "2fmoGnbR/Wg05lDJFnfF3XSXzislxsqxch+kjRoB6YjnxpmtXMxb520S+8VY/F3CLhPlvhYThMY99XDOshbjC9SWBNxefAJbtPbu"
    "clyvg9dQLKW3ZCp4YF74XjCqbTHYuSJ8rQlOB8aA638HbfgxPMcsiNFDLJsTvXxd1LQRkEX2qJvn/Yge24+BufDv2PDlUHhKNbzh"
    "B8S/b5E56AOOqZzwqFpYa2P9kR5t8EdCdg32WV94AygvvD0U1Lo2st49o8EuKnsPRGyZV9aSn6IjLIIa2RknEn24oWiAvxibI39R"
    "YfC+SPRXne0cWwzdUFtIFvQuc6S/M3us4iGuwrHyw6YM/hrlmgrCuN9zuN7lBcR8xX76LUD/yXf57Ds9+x7r2fc4Af/7JZ+7/qJn"
    "//U99nO/K/wTz373f37j2c/iuedfNP6/no317N7n3xPrufvE78anZ+PHeu5z8exz8bvf1XPzEs/W+vz7nr9Oz82PfjP60zfJZ7/T"
    "s1FEwPNf9Nzvv73yn/r61xrEv0f43xULzO5/Z/XX44sX/iz+5I7fPvn7K/TCO+hvRvit/v54x9/pT7zEJ69y31+tVPytTv7qs5fT"
    "r/iH63z1WYq/+f7yMxWvYCfx2hZ9Pe2/vh5f/VPxmjoQr/SM+I/pT7x01L983IiXnrN4RS2+noX+mWX+Tpv/JMuLl/j5n6389XKf"
    "+Aefva4HvU7dfB1PC3hNnxMvndf/eO2v9CRewdvolSP6rzP6y9Vc8Uo1TLxC7LyoU3nRzP9aU/QSV//Yp/3VLH5f9//KuvQKfi9e"
    "uK4/ao9+0yn/2Qz+yi70m5nJF9ryz9amf9dV/usd4iXs/bSDpxfqWL+gW6R/r/jFvkj/nj39h3q6//t6vUr1W6u8uG94cdTQn+Sf"
    "161EL95tvVoV/WfeI/6D7wr4L4z88vsD8afX6bUrxX9z3X+ev+QfKMFvvYx+t2t+3ofoTzOKeLbXfz4KxO/oxW/ztviL/Cj+Lc+P"
    "IP9wv/ib3Zp4jd79/77+/+TIf3nT89ah33mC+JMnX78Tfr1d1KtH8D+9/vq77/9OxvzPzEi8IG+Il+re/rt1RPyXr7zaGp5W6/8H"
    "TuOGqg=="
)


# ================================================================ catch mode (--mode catch)
# A launcher on the floor fires 3-5 table-tennis balls (40 mm, 2.7 g, quadratic air drag) upward at random oblique
# angles; each ball's position is measured at 100 Hz with Gaussian noise; the formation estimates every ball's
# trajectory (EKF, ballistic model with drag), picks an interception point on its descending branch and moves (rigid
# pattern, the existing feedforward) so that the ball falls between the four drones into the net.
# Ball-net contact: the net is taken as a fine mesh (meshes smaller than the ball) spanned by the knots: every grid cell
# is two triangles (2 x 8 x 8 = 128); a ball within its radius of a triangle gets a penalty force (spring k, damper c on
# the normal relative speed, viscous friction capped by mu * normal force), applied with the opposite sign to the
# triangle's three knots by barycentric weights, so the ball's weight and impact load the knots, the tendons, the
# strings and the drones. A per-ball "side" (above / below the net surface, updated only while the ball is clear of
# the net) keeps a fast ball from tunnelling. Without balls (n_balls = 0) nothing of this exists in the model.
import math

CATCH_PARK = (-30.0, 0.0, 0.5)          # sim coordinates of the parked (not yet launched) balls, reset every step
NET_CONTACT = {"k_N_m": 100.0,          # normal stiffness (static sink under one ball's weight 0.26 mm)
               "c_N_s_m": 0.10,         # normal damping on the relative speed (explicit: c dt / m_knot < 1)
               "ct_N_s_m": 0.10,        # tangential viscous friction coefficient
               "mu": 0.6,               # Coulomb cap of the tangential force
               "edge_margin_m": 0.02,   # triangle edges blended over +-20 mm (partition of unity on a flat net)
               "depth_max_m": 0.06}     # beyond this on the wrong side a triangle no longer acts (ball went round)
CATCH_LAUNCHER_BOX = (1.0, 5.0, 0.30)   # m, launcher (centre of its barrel row), box coordinates
CATCH_BARREL_M = 0.06                   # m, barrels side by side along y (a salvo leaves from separate barrels)
# formation limits for the interception moves (PLANNER_RELAXED: horizontal 3 m/s / 3 m/s^2, climb 3 m/s / 2 m/s^2,
# descent 2 m/s / 2 m/s^2); a_tot caps the combined acceleration (the oblique derating of the mission planner)
CATCH_LIMITS = {"v_h": 3.0, "a_h": 3.0, "v_up": 3.0, "v_down": 2.0, "a_v": 2.0, "a_tot": 3.0}
CATCH_LIMITS_SETS = {"relaxed": CATCH_LIMITS,
                     "fast": {"v_h": 4.0, "a_h": 4.5, "v_up": 3.0, "v_down": 2.0, "a_v": 2.5, "a_tot": 4.5}}
# launch ranges (elevation and azimuth from +x in deg, speed in m/s); set from the reachability map
# Ranges at the minimum widths (elevation 15 deg, azimuth 30 deg; hard: 25 / 50 deg). The launcher fires along +x from
# box x = 1 m; at the azimuth width of 30 deg the landing points spread 2.9-3.3 m sideways whatever the elevation, so the
# tiers differ mainly in the flight time (time to the standby aim height, Monte Carlo: easy 1.73-2.14 s, medium
# 1.50-2.09 s, hard 1.06-2.02 s) and the largest offsets (from the tier's standby point, p90 / max: easy 1.68 / 2.19 m,
# medium 1.73 / 2.41 m, hard 2.71 / 3.93 m); see catch_tier_table and the reachability map (out_official/catch/reach).
# Elevations above 80 deg are left out: the ball then comes down within ~1.5 m of the launcher, next to its own rising
# branch, where the formation cannot wait (clear_up_m).
CATCH_TIERS = {"easy": {"el_deg": (65.0, 80.0), "az_deg": (-15.0, 15.0), "speed_m_s": (14.0, 18.0)},
               "medium": {"el_deg": (60.0, 75.0), "az_deg": (-15.0, 15.0), "speed_m_s": (12.0, 18.0)},
               "hard": {"el_deg": (45.0, 70.0), "az_deg": (-25.0, 25.0), "speed_m_s": (10.0, 18.0)}}
CATCH_DEFAULT = dict(seed=1, tier="medium", el_deg=None, az_deg=None, speed_m_s=None, launch="sequence",
                     interval_s=2.0, interval_range_s=None, n_balls=(3, 5), noise_m=0.005, launches=None,
                     standby=None, limits="relaxed", loaded_accel_factor=0.7, standby_accel_factor=0.5,
                     t_early_s=0.25, t_hold_s=0.15, aim_alpha=0.3, r_in_m=None, clear_up_m=None,
                     z_net_range=(0.7, 2.6), z_net_step=0.2, z_net0=1.0, replan_hz=20.0, n_meas_min=3, ff_balls=True,
                     return_standby=True, h_dn_per_ball_m=None, h_dn_table=None, hyst=0.3, t_after_last_max_s=14.0,
                     p_over=None, spacing=1.25, h_dn=None, ff_path=None, planner_off=False, label=None,
                     ekf_q_acc=0.05, ekf_sig_v0=4.0, net_side=1.0,
                     dz_penalty=0.3, slack_cap=0.8, ref_z_net=1.0, standby_xy_z_net=None, moving=None,
                     traj=None, offcentre=None, move_spacing=None, p2p=None, count_draw=None, pile_catch=False,
                     boxfix=True)
# Height / moving-catch switches (out_official/catch_height): dz_penalty (score per m of height change) and slack_cap
# (time to spare counted per catch, s); z_net_range / z_net_step (candidate net-centre heights); z_net0 (standby net-centre
# height); standby_xy_z_net (height whose mean crossing point sets the standby point's horizontal position; None: z_net0);
# ref_z_net (fixed height of the prediction record and of the reachability axis, 1.0 m as before); moving (None: every
# interception ends at rest; dict: arrive with a velocity, see CATCH_MOVING).
# Transfer-speed switches (out_official/catch_speed), all None by default (behaviour as before): traj (jerk-limited
# straight moves, CATCH_TRAJ), offcentre (formation centre moved off the ball's crossing point, CATCH_OFFCENTRE),
# move_spacing ({"spacing": m, "v_full": m/s}: drone spacing changed with the horizontal reference speed, full change at
# v_full and above, back to the hold spacing at rest), p2p ({"d": [dx, dy, dz], "t_move": s, "t_after_s": s}: no balls;
# a single move from the standby point by d at t_move, analysed by p2p_metrics; optional "n_drop": n balls released at
# rest 0.1 m above the net centre every drop_gap_s (0.4 s) first, the move settle_s (5 s) after the last, the formation
# climbing by their sink 2 s before the move; catch_netdyn identification).
# Many-ball switches (out_official/catch_10balls), off by default: count_draw ([a, b]: the ball count is drawn from
# [a, b] as by default and then replaced by n_balls (which must be fixed, [n, n]), so a seed's first balls keep the
# elevation, azimuth, speed and launch time they have with n_balls = (a, b); only the barrel row, centred on the ball
# count, moves them sideways); pile_catch (a ball resting on balls that are in the net, a chain of ball-ball contacts,
# and moving with the net counts as caught: with many balls they pile up and the top ones never touch the net).
# boxfix (out_official/catch_boxfix), on by default since 2026-11-23 ("boxfix": false turns it off completely: then
# bit-identical to the code before the fix; a dict, e.g. {"floor_tol_m": 0.0}, turns it on with those entries changed):
# when a move's path would leave the formation box, the planner brakes to a stop point ahead; without the fix that stop
# point includes the vertical velocity (a descending formation is sent lower) and is recomputed at every replan from the
# still-moving reference (the stop keeps sliding ahead and the formation can run far out of the box). With the fix:
# horizontal stop point at the current height (not below the height floor), the braking plan is kept until it reaches
# rest, a braking path that would itself leave the box keeps the current plan instead, the box check uses 0.1 mm
# instead of 5 cm, and a new path dipping more than 0.10 m below the height floor (net centre at z_net_range[0] below
# the drones' height, or the current / last planned height if lower) keeps the current plan until the next replan.
# Every run reports the reference's largest excursion out of the box / below the floor (summary "reference").
# netdyn (out_official/catch_netdyn), None by default (behaviour as before): a dict (CATCH_NETDYN entries) turns on the
# net-dynamics planner (moves shaped with the identified net model so the net centre ends at rest on the target,
# model-predicted tilt / thrust / string-slack / box checks) and the drag and acceleration-lead feedforward;
# {"plan": false} keeps the moves as without netdyn and adds only the feedforward.
CATCH_BOXFIX = {"tol_m": 1e-4,           # box check of a planned path with 0.1 mm tolerance (was 0.05 m; exactly 0
                                         # lets rounding at a box face, where targets are clipped, reject every path)
                "floor": True,           # also check the path against the height floor ...
                "floor_tol_m": 0.10,     # ... rejecting dips of more than 0.10 m (a replan with a downward velocity
                                         # always dips a few cm; 0.0 keeps the reference above the floor but costs
                                         # catches, e.g. catch seed 5: 4/5 -> 3/5 at x1, 5/5 -> 3/5 at x5)
                "horizontal_stop": True, "latch": True,
                "keep_plan": True}       # braking path would leave the box: keep the current (checked) plan instead
# (a path that only dips below the height floor keeps the current plan until the next replan: no braking)
CATCH_MOVING = {"v_frac": 0.5,        # end velocity = v_frac x the ball's horizontal velocity at the aim height ...
                "v_max": 1.5,         # ... capped at this (m/s)
                "vz_frac": 0.0,       # optional downward end velocity: vz_frac x the ball's vertical speed, capped:
                "vz_max": 1.0,
                "k_lag": 0.0075,      # net-centre trail behind the formation at constant speed, m per (m/s)^2 (travel
                                      # mode: 100 mm at 3.6 m/s, 136 mm at 4.3 m/s)
                "coast_s": 0.15,      # keep the end velocity this long after the catch ...
                "brake_factor": 0.5,  # ... then brake to rest at this share of the acceleration limits
                "t_early_s": 0.0,     # arrival margin (at the catch time itself)
                "freeze_s": 0.2,      # no replanning of the current ball in its last 0.2 s
                # end velocities the planner compares for every interception point (it keeps the one with the most
                # time to spare): rest (arrive early and wait, as without moving), ball (v_frac x the ball's horizontal
                # velocity), approach_half / approach (along the approach direction, v_max / 2 and v_max)
                "options": ["rest", "ball", "approach_half", "approach"],
                # every interception with an end velocity is charged brake_cost per s of its coast and brake (the
                # formation is busy and ends up past the point), so the end velocity is only used where it pays off;
                # the next interception starts from where the brake ends
                "brake_cost": 0.3,
                # late_only: a ball that can still be reached at rest (with the early-arrival margin) is caught at rest;
                # end velocities are offered only for balls no rest interception can reach in time (the coast and brake
                # after a moving catch carry the formation 1-1.5 m past the point and cost the following balls)
                "late_only": False}
# Planner settings chosen on medium-tier trial batches (11 seeds each, out of the final studies; see the report):
# - limits "relaxed" (PLANNER_RELAXED values); the "fast" set (4.5 m/s^2) caught fewer balls (54-67 % vs 70 % at 1 s)
#   with the rotor-disc clearance below 30 mm in every trial and aborts after ball strikes
# - with balls in the net the acceleration limits are scaled by loaded_accel_factor 0.7 and the return to the standby
#   point by standby_accel_factor 0.5: without them 10 of 11 trials (2 s interval) broke the 90 % motor rule (climbing
#   back with 4-5 balls, horizontal 3 m/s^2 with 3-4 balls); with them none, at 96 instead of 98 % of balls caught
#   (with t_hold 0.15 s)
# - t_hold_s 0.15 s of standstill after a ball reaches the net (0.4: 86 %, 0.25: 94 %, 0.15: 96 % caught, no ball
#   thrown out in either); t_early_s 0.25 s (0.15 did not help)
# - net-centre heights from 0.7 m (0.6 m let the lowest knot reach 0.3 m in 3 of 11 trials with 5 balls, 3 s interval)
# geometry scaled with the formation spacing s (1.25 m: r_in 0.40 m, ring 0.45-1.05 m, clear_up 1.1 m, arrival 0.6 m)
# off-centre catch (catch cfg "offcentre"; None: the formation centre always on the ball's crossing point, as before):
# every interception point is also offered with the formation centre moved by up to r (m, the safe off-centre radius
# from the drop test) towards the current reference ("start") and towards the standby-height crossing of each other
# ball in flight ("others"); the rotor-plane crossing must still lie within r_in of the moved centre
CATCH_OFFCENTRE = {"r": 0.2, "toward": ["start", "others"]}
CATCH_MOTOR_LINES = (0.90, 0.95, 0.98)  # motor shares whose exceedance time is reported (safety.motor90.lines)
CATCH_GEOM = {"r_in": 0.32, "ring_in": 0.36, "ring_out": 0.84, "clear_up": 0.88, "arrive": 0.48}
CATCH_REL_V = 0.10                      # m/s, "caught": in contact >= 95 % of the last 1 s, mean speed relative to the
                                        # net surface under it below this, and at least 1 s since the first contact
CATCH_DH_PER_BALL = {1.0: None, 0.5: None}     # m, linear net-centre sink per ball (used without a table)
# increase of the drone-to-net-centre drop with 0, 1, 2 ... balls resting in the net (drop test b: five balls released
# one by one over the hovering formation; out_official/catch/drop/drop_test.json, "h_dn_table_m"); the planner aims
# with it (beyond the table: its last step repeated). Ten balls: out_official/catch_10balls/drop/drop10.json (the first
# entries agree within 0.6 mm; 6-10 balls: 0.1022, 0.1065, 0.1097, 0.1134, 0.1158), passed as h_dn_table in the
# 10-ball settings files
CATCH_DH_TABLE = {1.0: [0.0, 0.0512, 0.0712, 0.0836, 0.0908, 0.0963], 0.5: [0.0, 0.0378, 0.0456, 0.0519, 0.0566, 0.0623]}


def ball_k(p):
    """Quadratic drag per unit mass [1/m]: a = -k |v| v, k = rho Cd A / (2 m)."""
    return 0.5 * p.rho_air * p.ball_cd * np.pi * (p.ball_d / 2) ** 2 / p.ball_m


def flight_rk4(x0, k, dt=0.01, n_max=800, z_stop=0.0, g=G):
    """Ballistic flight with quadratic drag from state x0 = (x, y, z, vx, vy, vz), RK4 in plain floats.
    Returns (N, 7) [t, x, y, z, vx, vy, vz] until z < z_stop while descending or n_max steps."""
    x, y, z, vx, vy, vz = (float(v) for v in x0)
    out = [(0.0, x, y, z, vx, vy, vz)]
    h2, h6 = 0.5 * dt, dt / 6.0
    t = 0.0
    for _ in range(n_max):
        s = k * math.sqrt(vx * vx + vy * vy + vz * vz)
        a1x, a1y, a1z = -s * vx, -s * vy, -g - s * vz
        bx, by, bz = vx + h2 * a1x, vy + h2 * a1y, vz + h2 * a1z
        s = k * math.sqrt(bx * bx + by * by + bz * bz)
        a2x, a2y, a2z = -s * bx, -s * by, -g - s * bz
        cx, cy, cz = vx + h2 * a2x, vy + h2 * a2y, vz + h2 * a2z
        s = k * math.sqrt(cx * cx + cy * cy + cz * cz)
        a3x, a3y, a3z = -s * cx, -s * cy, -g - s * cz
        ex, ey, ez = vx + dt * a3x, vy + dt * a3y, vz + dt * a3z
        s = k * math.sqrt(ex * ex + ey * ey + ez * ez)
        a4x, a4y, a4z = -s * ex, -s * ey, -g - s * ez
        x += h6 * (vx + 2 * bx + 2 * cx + ex)
        y += h6 * (vy + 2 * by + 2 * cy + ey)
        z += h6 * (vz + 2 * bz + 2 * cz + ez)
        vx += h6 * (a1x + 2 * a2x + 2 * a3x + a4x)
        vy += h6 * (a1y + 2 * a2y + 2 * a3y + a4y)
        vz += h6 * (a1z + 2 * a2z + 2 * a3z + a4z)
        t += dt
        out.append((t, x, y, z, vx, vy, vz))
        if z < z_stop and vz < 0:
            break
    return np.array(out)


def crossing(tr, z_c, descending=True):
    """First time and state at which a flight_rk4 trajectory crosses height z_c (descending or ascending branch);
    None if it does not."""
    z, vz = tr[:, 3], tr[:, 6]
    if descending:
        idx = np.flatnonzero((z[:-1] >= z_c) & (z[1:] < z_c) & (vz[1:] < 0))
    else:
        idx = np.flatnonzero((z[:-1] < z_c) & (z[1:] >= z_c) & (vz[1:] > 0))
    if not len(idx):
        return None
    i = idx[0]
    w = (z[i] - z_c) / (z[i] - z[i + 1])
    return (1 - w) * tr[i] + w * tr[i + 1]


class BallEKF:
    """Extended Kalman filter of one ball: state (position, velocity), ballistic model with quadratic drag (k known),
    random-acceleration process noise q_acc, position measurements with standard deviation sig_m."""

    def __init__(self, z, t, v_prior, sig_v0, sig_m, k, q_acc):
        self.x = np.concatenate([np.asarray(z, float), np.asarray(v_prior, float)])
        sm = max(sig_m, 1e-3)
        self.P = np.diag([sm ** 2] * 3 + [sig_v0 ** 2] * 3)
        self.R = np.eye(3) * sm ** 2
        self.t, self.k, self.q, self.n = t, k, q_acc, 1

    def predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        tr = flight_rk4(self.x, self.k, dt=dt, n_max=1, z_stop=-1e9)
        v = self.x[3:]
        sp = max(np.linalg.norm(v), 1e-6)
        J = -self.k * (sp * np.eye(3) + np.outer(v, v) / sp)
        F = np.eye(6)
        F[:3, 3:] = dt * np.eye(3) + 0.5 * dt * dt * J
        F[3:, 3:] = np.eye(3) + dt * J
        q = self.q ** 2
        Q = np.zeros((6, 6))
        Q[:3, :3], Q[:3, 3:], Q[3:, :3], Q[3:, 3:] = (q * dt ** 3 / 3 * np.eye(3), q * dt ** 2 / 2 * np.eye(3),
                                                      q * dt ** 2 / 2 * np.eye(3), q * dt * np.eye(3))
        self.x = tr[-1, 1:].copy()
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, z):
        S = self.P[:3, :3] + self.R
        K = self.P[:, :3] @ np.linalg.inv(S)
        y = np.asarray(z) - self.x[:3]
        self.x = self.x + K @ y
        self.P = self.P - K @ self.P[:3, :]
        self.n += 1
        return float(y @ np.linalg.solve(S, y))


def net_triangles(n):
    """(2 (n-1)^2, 3) knot indices (i * n + j) of the net surface triangles, normals up for the flat net."""
    T = []
    for i in range(n - 1):
        for j in range(n - 1):
            a, b, c, d = i * n + j, (i + 1) * n + j, (i + 1) * n + j + 1, i * n + j + 1
            T += [(a, b, c), (a, c, d)]
    return np.array(T)


def net_contact_forces(kx, kv, P, V, side, tri, r, h, C=NET_CONTACT):
    """Penalty contact of M balls (centres P, velocities V, side +-1) with the net triangles.
    Returns (force on each ball (M, 3), force on each knot (N, 3), in contact (M,), normal force (M,),
    side as seen now (M,), nan where undetermined, ball speed relative to the net surface under it (M,),
    normal-force weighted, nan without contact)."""
    A, B, Cc = kx[tri[:, 0]], kx[tri[:, 1]], kx[tri[:, 2]]
    e1, e2 = B - A, Cc - A
    nr = np.empty_like(e1)                     # np.cross(e1, e2), same operations
    nr[:, 0] = e1[:, 1] * e2[:, 2] - e1[:, 2] * e2[:, 1]
    nr[:, 1] = e1[:, 2] * e2[:, 0] - e1[:, 0] * e2[:, 2]
    nr[:, 2] = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    nh = nr / np.maximum(np.linalg.norm(nr, axis=1), 1e-12)[:, None]
    d00, d01, d11 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    den = np.maximum(d00 * d11 - d01 * d01, 1e-18)
    w = P[:, None, :] - A[None]
    s = (w * nh[None]).sum(2)                                   # (M, T) signed distance to the plane
    q = w - s[..., None] * nh[None]
    d20, d21 = (q * e1[None]).sum(2), (q * e2[None]).sum(2)
    bv = (d11 * d20 - d01 * d21) / den
    bw = (d00 * d21 - d01 * d20) / den
    bu = 1.0 - bv - bw
    mb = np.minimum(np.minimum(bu, bv), bw)
    wgt = np.clip(0.5 + mb / (2 * C["edge_margin_m"] / h), 0.0, 1.0)
    ss = side[:, None] * s
    delta = r - ss
    act = (wgt > 0) & (delta > 0) & (ss > -C["depth_max_m"])
    # side seen now: the triangle under the ball (projection inside), if the ball is clear of it
    s_in = np.where(mb >= 0, np.abs(s), np.inf)
    j = s_in.argmin(axis=1)
    sj = s[np.arange(len(P)), j]
    side_now = np.where(np.isfinite(s_in.min(axis=1)) & (np.abs(sj) > r), np.sign(sj), np.nan)
    M = len(P)
    if not act.any():
        return np.zeros((M, 3)), np.zeros_like(kx), np.zeros(M, bool), np.zeros(M), side_now, np.full(M, np.nan)
    cu, cv, cw = np.clip(bu, 0, 1), np.clip(bv, 0, 1), np.clip(bw, 0, 1)
    tot = np.maximum(cu + cv + cw, 1e-12)
    cu, cv, cw = cu / tot, cv / tot, cw / tot
    vnet = (cu[..., None] * kv[tri[:, 0]][None] + cv[..., None] * kv[tri[:, 1]][None]
            + cw[..., None] * kv[tri[:, 2]][None])
    vrel = V[:, None, :] - vnet
    ns = side[:, None, None] * nh[None]
    vn = (vrel * ns).sum(2)
    fn = np.where(act, np.maximum(C["k_N_m"] * delta - C["c_N_s_m"] * vn, 0.0) * wgt, 0.0)
    vt = vrel - vn[..., None] * ns
    vtm = np.linalg.norm(vt, axis=2) + 1e-12
    ft = np.where(act, np.minimum(C["ct_N_s_m"] * vtm * wgt, C["mu"] * fn), 0.0)
    F = fn[..., None] * ns - (ft / vtm)[..., None] * vt          # on the ball, per triangle
    fk = np.zeros_like(kx)
    for col, cb in ((0, cu), (1, cv), (2, cw)):
        Ft = -(cb[..., None] * F).sum(0)                          # (T, 3) on that vertex of each triangle
        for a in range(3):
            fk[:, a] += np.bincount(tri[:, col], weights=Ft[:, a], minlength=len(kx))
    fs = fn.sum(1)
    vr = np.where(fs > 0, (fn * np.linalg.norm(vrel, axis=2)).sum(1) / np.maximum(fs, 1e-12), np.nan)
    return F.sum(1), fk, act.any(axis=1), fs, side_now, vr


QUINTIC_TGRID = np.geomspace(0.04, 8.0, 48)


def _quintic(p0, v0, a0, d, T):
    """Coefficients c3, c4, c5 of the quintic from (p0, v0, a0) to (p0 + d, 0, 0) in time T (broadcasting)."""
    return ((20 * d - 12 * v0 * T - 3 * a0 * T ** 2) / (2 * T ** 3),
            (-30 * d + 16 * v0 * T + 3 * a0 * T ** 2) / (2 * T ** 4),
            (12 * d - 6 * v0 * T - a0 * T ** 2) / (2 * T ** 5))


def _quintic_v(p0, v0, a0, d, T, v1):
    """c3, c4, c5 of the quintic from (p0, v0, a0) to (p0 + d, v1, 0) in time T."""
    return ((20 * d - (8 * v1 + 12 * v0) * T - 3 * a0 * T ** 2) / (2 * T ** 3),
            (-30 * d + (14 * v1 + 16 * v0) * T + 3 * a0 * T ** 2) / (2 * T ** 4),
            (12 * d - 6 * (v1 + v0) * T - a0 * T ** 2) / (2 * T ** 5))


def quintic_ok(p0, v0, a0, P1, Tg, lim, ns=21, V1=None):
    """(N, M) bool: the quintic from the reference state to each target P1 (N, 3) in each time Tg (M,) respects the
    limits (horizontal / vertical speed and acceleration, total acceleration; the acceleration limits never below the
    starting acceleration, which the previous segment already had)."""
    d = np.asarray(P1, float)[:, None, :] - p0
    T = np.asarray(Tg, float)[None, :, None]
    if V1 is None:
        c3, c4, c5 = _quintic(p0, v0, a0, d, T)
    else:
        c3, c4, c5 = _quintic_v(p0, v0, a0, d, T, np.asarray(V1, float)[:, None, :])
    tau = (np.linspace(0.0, 1.0, ns)[None, None, :, None] * T[..., None])
    c3, c4, c5 = c3[:, :, None], c4[:, :, None], c5[:, :, None]
    vel = v0 + a0 * tau + 3 * c3 * tau ** 2 + 4 * c4 * tau ** 3 + 5 * c5 * tau ** 4
    acc = a0 + 6 * c3 * tau + 12 * c4 * tau ** 2 + 20 * c5 * tau ** 3
    e = 1.01
    ah0, az0, at0 = np.hypot(a0[0], a0[1]), abs(a0[2]), np.linalg.norm(a0)
    ok = ((np.hypot(vel[..., 0], vel[..., 1]) <= lim["v_h"] * e) & (vel[..., 2] <= lim["v_up"] * e)
          & (-vel[..., 2] <= lim["v_down"] * e)
          & (np.hypot(acc[..., 0], acc[..., 1]) <= max(lim["a_h"], ah0) * e)
          & (np.abs(acc[..., 2]) <= max(lim["a_v"], az0) * e)
          & (np.linalg.norm(acc, axis=-1) <= max(lim["a_tot"], at0) * e))
    return ok.all(axis=2)


def quintic_tmin(p0, v0, a0, P1, lim, V1=None):
    """Shortest grid time (QUINTIC_TGRID) to reach each target at rest (or with end velocity V1) within the limits
    (inf: none)."""
    ok = quintic_ok(p0, v0, a0, P1, QUINTIC_TGRID, lim, V1=V1)
    first = np.where(ok.any(axis=1), ok.argmax(axis=1), -1)
    return np.where(first >= 0, QUINTIC_TGRID[np.maximum(first, 0)], np.inf), first


def quintic_tmin_fine(p0, v0, a0, p1, lim, v1=None):
    V1 = None if v1 is None else np.asarray(v1, float)[None]
    T, i = quintic_tmin(p0, v0, a0, np.asarray(p1)[None], lim, V1=V1)
    if not np.isfinite(T[0]):
        return 8.0
    if i[0] == 0:
        return float(T[0])
    lo, hi = QUINTIC_TGRID[i[0] - 1], QUINTIC_TGRID[i[0]]
    for _ in range(10):
        mid = 0.5 * (lo + hi)
        if quintic_ok(p0, v0, a0, np.asarray(p1)[None], np.array([mid]), lim, ns=41, V1=V1)[0, 0]:
            hi = mid
        else:
            lo = mid
    return float(hi)


def t_rest_to_rest(Pa, Pb, lim):
    """(N, M) minimum-jerk rest-to-rest times between points (analytic: peak acceleration 10/sqrt(3) L / T^2,
    peak speed 1.875 L / T, per component and in total)."""
    d = Pb[None, :, :] - Pa[:, None, :]
    dh, dz = np.hypot(d[..., 0], d[..., 1]), d[..., 2]
    k = 10 / np.sqrt(3)
    vz = np.where(dz > 0, lim["v_up"], lim["v_down"])
    return np.maximum.reduce([np.sqrt(k * dh / lim["a_h"]), 1.875 * dh / lim["v_h"], np.sqrt(k * np.abs(dz) / lim["a_v"]),
                              1.875 * np.abs(dz) / vz, np.sqrt(k * np.linalg.norm(d, axis=-1) / lim["a_tot"])])


# ---- jerk-limited ("S-curve") moves (catch cfg "traj"; None: the quintic above, as before)
# A straight move to the target with a time-optimal jerk-limited speed profile along the line (acceleration ramps at the
# jerk limit, holds at the limit, ramps back; cruise at the speed limit; the same down to rest), so the acceleration
# stays at its limit longer than in the quintic (rest to rest at the same peak acceleration: T = sqrt(4 L / a) for an
# infinite jerk limit instead of sqrt(5.77 L / a)). A start velocity / acceleration across the line is brought to zero by
# a quintic over the same time; above v_perp_max / a_perp_max the move falls back to the quintic.
CATCH_TRAJ = {"jerk": 20.0,           # m/s^3, jerk limit along the path (the quintic's peak jerk for 1-3 m moves at
                                      # 3 m/s^2 is 13-22 m/s^3)
              "acc_factor": 1.0,      # acceleration limit x this while speeding up ...
              "brake_factor": 1.0,    # ... and while braking (asymmetric limits)
              "axial_factor": 1.0,    # horizontal acceleration limit (and a_tot) x this for moves along the formation's
              "diag_factor": 1.0,     # axes / along its diagonals (towards a drone), interpolated with |sin 2 heading|
              "v_perp_max": 0.3,      # m/s
              "a_perp_max": 1.0}      # m/s^2


def _path_limits(u, lim, tj):
    """Speed limit and acceleration limits (speeding up, braking) along unit directions u (N, 3)."""
    uh, uz = np.hypot(u[:, 0], u[:, 1]), u[:, 2]
    w = np.abs(np.sin(2.0 * np.arctan2(u[:, 1], u[:, 0])))
    df = tj["axial_factor"] + (tj["diag_factor"] - tj["axial_factor"]) * w
    with np.errstate(divide="ignore", invalid="ignore"):
        A = np.minimum.reduce([np.where(uh > 1e-9, lim["a_h"] * df / np.maximum(uh, 1e-12), np.inf),
                               np.where(np.abs(uz) > 1e-9, lim["a_v"] / np.maximum(np.abs(uz), 1e-12), np.inf),
                               lim["a_tot"] * df])
        V = np.minimum(np.where(uh > 1e-9, lim["v_h"] / np.maximum(uh, 1e-12), np.inf),
                       np.where(uz > 1e-9, lim["v_up"] / np.maximum(uz, 1e-12),
                                np.where(uz < -1e-9, lim["v_down"] / np.maximum(-uz, 1e-12), np.inf)))
    return V, A * tj["acc_factor"], A * tj["brake_factor"]


def _vchange(v0, a0, v1, A, J):
    """Minimum-time jerk-limited change (v0, a0) -> (v1, 0) with |a| <= max(A, |a0|), |jerk| <= J (arrays).
    Returns (s, t1, th, t3): jerk s J for t1 (acceleration to s ap), hold for th, jerk -s J for t3."""
    dvf = a0 * np.abs(a0) / (2 * J)
    s = np.where(v1 - v0 - dvf >= 0, 1.0, -1.0)
    aa0, Dv = s * a0, s * (v1 - v0)
    Am = np.maximum(A, np.abs(a0))
    ap = np.sqrt(np.maximum(J * Dv + 0.5 * aa0 ** 2, 0.0))
    cap = ap > Am
    ap = np.where(cap, Am, ap)
    th = np.where(cap, np.maximum(Dv - (2 * ap ** 2 - aa0 ** 2) / (2 * J), 0.0) / np.maximum(ap, 1e-12), 0.0)
    return s, np.maximum((ap - aa0) / J, 0.0), th, ap / J


def _vchange_dist(v0, a0, s, t1, th, t3, J):
    """Distance and duration of a _vchange profile."""
    j = s * J
    x1 = v0 * t1 + a0 * t1 ** 2 / 2 + j * t1 ** 3 / 6
    v1 = v0 + a0 * t1 + j * t1 ** 2 / 2
    a1 = a0 + j * t1
    x2 = v1 * th + a1 * th ** 2 / 2
    v2 = v1 + a1 * th
    x3 = v2 * t3 + a1 * t3 ** 2 / 2 - j * t3 ** 3 / 6
    return x1 + x2 + x3, t1 + th + t3


def _scurve_1d(v0, a0, L, V, Aa, Ab, J, n_iter=48):
    """Time-optimal jerk-limited 1-D move from (0, v0, a0) to (L, 0, 0) (arrays): change the speed to vp (acceleration
    back to 0), cruise, brake to rest. Returns (T, vp, cruise); T is nan where even vp = -V overshoots."""
    with np.errstate(invalid="ignore", over="ignore"):
        return _scurve_1d_(v0, a0, L, V, Aa, Ab, J, n_iter)


def _scurve_1d_(v0, a0, L, V, Aa, Ab, J, n_iter):
    def D(vp):
        s1 = np.sign(vp - v0 - a0 * np.abs(a0) / (2 * J))
        A1 = np.where(s1 * v0 >= 0, Aa, Ab)                  # speeding up (or from rest): Aa; braking: Ab
        c1 = _vchange(v0, a0, vp, A1, J)
        x1, T1 = _vchange_dist(v0, a0, *c1, J)
        c3 = _vchange(vp, np.zeros_like(vp), np.zeros_like(vp), Ab, J)
        x3, T3 = _vchange_dist(vp, np.zeros_like(vp), *c3, J)
        return x1 + x3, T1 + T3
    Dm, Tm = D(V)
    full = Dm <= L
    lo, hi = -V.copy(), V.copy()
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        dm, _ = D(mid)
        big = dm > L
        hi = np.where(big, mid, hi)
        lo = np.where(big, lo, mid)
    vp = np.where(full, V, lo)
    Dv, Tv = D(vp)
    cruise = np.where(full, (L - Dm) / np.maximum(V, 1e-12), 0.0)
    T = Tv + cruise
    bad = (~full) & (D(-V)[0] > L)
    return np.where(bad, np.nan, T), vp, cruise


def _scurve_dirs(c0, v0, a0, P1, lim, tj):
    d = np.asarray(P1, float) - c0
    L = np.linalg.norm(d, axis=1)
    u = d / np.maximum(L, 1e-12)[:, None]
    vpar, apar = u @ v0, u @ a0
    vperp = np.linalg.norm(v0[None] - vpar[:, None] * u, axis=1)
    aperp = np.linalg.norm(a0[None] - apar[:, None] * u, axis=1)
    V, Aa, Ab = _path_limits(u, lim, tj)
    ok = (L > 1e-4) & (vperp <= tj["v_perp_max"]) & (aperp <= tj["a_perp_max"])
    return d, L, u, vpar, apar, V, Aa, Ab, ok


def scurve_tmin(c0, v0, a0, P1, lim, tj):
    """Duration of the jerk-limited move from the reference state to each target P1 (N, 3); nan where the move would
    fall back to the quintic."""
    tj = {**CATCH_TRAJ, **tj}
    d, L, u, vpar, apar, V, Aa, Ab, ok = _scurve_dirs(c0, v0, a0, P1, lim, tj)
    T, _, _ = _scurve_1d(vpar, apar, L, V, Aa, Ab, float(tj["jerk"]))
    return np.where(ok, T, np.nan)


def scurve_rest_to_rest(Pa, Pb, lim, tj):
    """(N, M) rest-to-rest durations of the jerk-limited moves (the counterpart of t_rest_to_rest)."""
    tj = {**CATCH_TRAJ, **tj}
    d = (Pb[None, :, :] - Pa[:, None, :]).reshape(-1, 3)
    L = np.linalg.norm(d, axis=1)
    u = d / np.maximum(L, 1e-12)[:, None]
    V, Aa, Ab = _path_limits(u, lim, tj)
    z = np.zeros(len(L))
    T, _, _ = _scurve_1d(z, z, L, V, Aa, Ab, float(tj["jerk"]))
    return np.where(L > 1e-9, T, 0.0).reshape(len(Pa), len(Pb))


def _poly_shift(c, s):
    """Coefficients (6, 3) of p(s + tau) for p(tau) = sum c_k tau^k."""
    from math import comb
    out = np.zeros_like(c)
    for m in range(6):
        for k in range(m, 6):
            out[m] += c[k] * comb(k, m) * s ** (k - m)
    return out


def scurve_segments(t, c0, v0, a0, target, lim, tj, C0):
    """Reference segments [(t_start, T, b (6, 3))] of the jerk-limited move (offsets from C0) and the total duration;
    None where the move falls back to the quintic."""
    tj = {**CATCH_TRAJ, **tj}
    J = float(tj["jerk"])
    d, L, u, vpar, apar, V, Aa, Ab, ok = _scurve_dirs(c0, v0, a0, np.asarray(target, float)[None], lim, tj)
    if not ok[0]:
        return None
    T, vp, cruise = _scurve_1d(vpar, apar, L, V, Aa, Ab, J)
    if not np.isfinite(T[0]):
        return None
    u, vpar, apar, vp, cruise = u[0], float(vpar[0]), float(apar[0]), vp, float(cruise[0])
    s1 = np.sign(vp - vpar - apar * abs(apar) / (2 * J))
    A1 = np.where(s1 * vpar >= 0, Aa, Ab)
    sa, t1, th, t3 = (float(x[0]) for x in _vchange(np.array([vpar]), np.array([apar]), vp, A1, J))
    sb, u1, uh, u3 = (float(x[0]) for x in _vchange(vp, np.zeros(1), np.zeros(1), Ab, J))
    phases = [(t1, sa * J), (th, 0.0), (t3, -sa * J), (cruise, 0.0), (u1, sb * J), (uh, 0.0), (u3, -sb * J)]
    Ttot = sum(p_[0] for p_ in phases)
    vperp, aperp = v0 - vpar * u, a0 - apar * u
    q3, q4, q5 = _quintic(np.zeros(3), vperp, aperp, np.zeros(3), Ttot)
    q = np.array([np.zeros(3), vperp, 0.5 * aperp, q3, q4, q5])
    segs, x, v, a, s = [], 0.0, vpar, apar, 0.0
    for dur, j in phases:
        if dur <= 1e-9:
            continue
        b = _poly_shift(q, s)
        b[0] += c0 - C0 + x * u
        b[1] += v * u
        b[2] += 0.5 * a * u
        b[3] += j / 6 * u
        segs.append((t + s, dur, b))
        x, v, a = x + v * dur + a * dur ** 2 / 2 + j * dur ** 3 / 6, v + a * dur + j * dur ** 2 / 2, a + j * dur
        s += dur
    return segs, Ttot


# ---- net dynamics in the planner (catch cfg "netdyn"; None: off, as before) -- out_official/catch_netdyn
# Reduced model, identified from point-to-point moves (catch_netdyn/ident, ident_fit_ref.py). The formation is not a
# perfect servo (its centre lags the reference by up to ~50 mm in a 1 m move: drone attitude / rotor lag, position loop
# poles at -2 rad/s), so the model is a cascade per axis: the formation centre's offset from the formation REFERENCE r
# is a sum of damped modes driven by the reference acceleration and velocity, and the net centre's offset from the
# formation centre a sum of damped modes driven by the formation's acceleration and velocity:
#     d_i'' + 2 zeta_i w_i d_i' + w_i^2 d_i = -g_i a - c_i v      (a, v: r'', r' / formation acceleration, velocity)
# Per ball count n: "fh" / "fv" formation modes (horizontal x and y / vertical), "h" / "v" net modes; each mode
# [w rad/s, zeta, g, c 1/s]. Values from catch_netdyn/ident/model_table.json (CATCH_NETDYN_MODEL below).
CATCH_NETDYN_MODEL = {
    "x1": [
        {"fh": [[2.0, 0.8185, 0.49623, 0.55268], [7.7717, 0.3523, 1.03228, -6.9266]], "fv": [[2.0, 1.0445, 0.1412, 0.04543]], "h": [[9.621, 0.0413, 0.32829, 2.87364], [8.646, 0.7452, 0.95218, -0.68929]], "v": [[5.372, 0.2115, 0.69115, 0.72551]]},  # n = 0
        {"fh": [[2.0, 1.2345, 1.79206, 1.35344], [7.2386, 0.5505, 0.22365, -17.55278]], "fv": [[12.1268, 0.3073, 0.75974, 3.83183]], "h": [[8.745, 0.6719, 1.71884, 1.04026]], "v": [[5.278, 0.1875, 0.70853, 0.4883]]},  # n = 1
        {"fh": [[2.0, 1.2241, 1.7528, 1.32754], [7.1334, 0.56, 0.19948, -16.75145]], "fv": [[11.5708, 0.2916, 0.73075, 3.51844]], "h": [[7.303, 0.5973, 1.28517, 0.62059]], "v": [[5.272, 0.1703, 0.71053, 0.54507]]},  # n = 2
        {"fh": [[2.0, 1.2334, 1.8118, 1.35008], [6.9408, 0.5746, 0.0627, -16.21916]], "fv": [[11.0876, 0.2761, 0.70324, 3.2449]], "h": [[6.553, 0.4978, 1.06855, 0.47839]], "v": [[5.244, 0.1581, 0.70995, 0.51952]]},  # n = 3
        {"fh": [[2.0, 1.2364, 1.85943, 1.3675], [6.7416, 0.5906, -0.04707, -15.5858]], "fv": [[10.8473, 0.2788, 0.71934, 3.11705]], "h": [[6.102, 0.4098, 0.92977, 0.38519]], "v": [[5.228, 0.1452, 0.70331, 0.5432]]},  # n = 4
        {"fh": [[2.0, 1.2166, 1.96997, 1.42973], [6.5178, 0.6208, -0.17716, -15.30759]], "fv": [[10.5758, 0.2779, 0.71778, 2.99169]], "h": [[5.868, 0.3432, 0.93049, 0.43929]], "v": [[5.22, 0.1384, 0.70399, 0.6018]]},  # n = 5
    ],
    "x5": [
        {"fh": [[2.0, 0.8169, 0.49879, 0.55813], [7.887, 0.3554, 1.06697, -7.22261]], "fv": [[12.807, 0.1452, 0.61057, 4.95834]], "h": [[9.633, 0.0396, 0.34031, 3.03441], [9.275, 0.8419, 1.07473, -1.00732]], "v": [[5.424, 0.203, 0.69402, 0.79144]]},  # n = 0
        {"fh": [[2.0, 1.0, 0.91893, 0.83973], [7.6342, 0.4584, 0.90474, -11.33215]], "fv": [[12.1489, 0.2437, 0.67802, 4.16277]], "h": [[8.428, 0.668, 1.66599, 1.42128]], "v": [[5.423, 0.1851, 0.7313, 1.04838]]},  # n = 1
        {"fh": [[2.0, 1.1693, 1.48786, 1.16653], [7.2467, 0.5275, 0.42601, -14.93867]], "fv": [[11.6833, 0.2495, 0.66971, 3.83363]], "h": [[7.124, 0.6078, 1.27044, 0.79498]], "v": [[5.37, 0.1712, 0.73181, 1.00652]]},  # n = 2
        {"fh": [[2.0, 1.2136, 1.66749, 1.25179], [7.0571, 0.5492, 0.21911, -15.39403]], "fv": [[11.1897, 0.2441, 0.6492, 3.49709]], "h": [[6.431, 0.5179, 1.05685, 0.57771]], "v": [[5.295, 0.1592, 0.72793, 0.81393]]},  # n = 3
        {"fh": [[2.0, 1.225, 1.71467, 1.26493], [6.9116, 0.5621, 0.12376, -15.00918]], "fv": [[10.8876, 0.2483, 0.65667, 3.3118]], "h": [[6.012, 0.4324, 0.91109, 0.45804]], "v": [[5.268, 0.1465, 0.71786, 0.80442]]},  # n = 4
        {"fh": [[2.0, 1.2399, 1.80361, 1.30161], [6.7439, 0.58, 0.00201, -14.81212]], "fv": [[10.6094, 0.2522, 0.66046, 3.16643]], "h": [[5.783, 0.3678, 0.8422, 0.40623]], "v": [[5.259, 0.1377, 0.71337, 0.85139]]},  # n = 5
        {"fh": [[2.0, 1.2403, 1.96467, 1.40799], [6.5875, 0.6129, -0.1509, -15.37713]], "fv": [[10.3784, 0.2617, 0.67644, 3.08829]], "h": [[5.66, 0.3251, 0.83408, 0.43431]], "v": [[5.242, 0.1306, 0.70572, 0.84429]]},  # n = 6
        {"fh": [[2.0, 1.2225, 2.20267, 1.59155], [6.3963, 0.6545, -0.37697, -16.42162]], "fv": [[10.1033, 0.2684, 0.68921, 2.9726]], "h": [[5.571, 0.2959, 0.86255, 0.52803]], "v": [[5.235, 0.124, 0.69668, 0.87767]]},  # n = 7
        {"fh": [[2.0, 1.1738, 2.35312, 1.78045], [6.2398, 0.7011, -0.54361, -17.45864]], "fv": [[9.8455, 0.2808, 0.70783, 2.85009]], "h": [[5.51, 0.2728, 0.89882, 0.62078]], "v": [[5.232, 0.1196, 0.67966, 0.94421]]},  # n = 8
        {"fh": [[2.0, 1.0857, 2.36365, 1.98001], [6.1799, 0.767, -0.56019, -18.93778]], "fv": [[9.4172, 0.2979, 0.72163, 2.60555]], "h": [[5.437, 0.2548, 0.93698, 0.72731]], "v": [[5.221, 0.116, 0.63713, 0.97471]]},  # n = 9
        {"fh": [[2.0, 1.0, 2.37744, 2.27655], [6.0923, 0.8583, -0.58579, -21.02706]], "fv": [[8.9522, 0.3158, 0.72931, 2.35329]], "h": [[5.369, 0.2358, 0.96652, 0.8268]], "v": [[5.231, 0.1084, 0.57057, 1.08162]]},  # n = 10
    ],
}
# planner switch: the formation reference of every move (_go) is a polynomial chosen so that the model ends at rest
# with the net centre (and the formation centre) on the target when the move ends (terminal-state equality
# constraints, least acceleration energy); its duration is the shortest on a grid for which the model-predicted checks
# pass: the existing speed / acceleration limits on the formation reference, drone tilt and motor share (predicted
# formation acceleration, string tensions from the predicted net acceleration and the static spread), no slack string,
# formation reference inside the box and above the height floor. Candidate interception times use the same durations.
CATCH_NETDYN = {"state": "model",       # replan start state: "model" (open-loop model, propagated with the reference)
                                        # or "measured" (net centre measured with noise, e.g. a motion-capture marker on
                                        # the net centre, and the drones' own position measurement; steady-state Kalman
                                        # filter on the model)
                "meas_noise_m": 0.002,  # net-centre measurement noise ("measured")
                "ff": False,            # string-force feedforward (m_net + balls) / 4 x predicted net acceleration
                "lead_s": 0.07,         # acceleration lead feedforward: m_drone (r''(t + lead) - r''(t)) per drone
                "drag_ff": True,        # air-drag feedforward at the reference velocity: the drone's own body drag
                                        # (Params.drag, level attitude) plus a quarter of the net's drag
                "plan": True,           # net-dynamics moves (False: the moves as without netdyn, only the feedforward)
                "K": 13,                # polynomial degree of the reference
                "ns": 33,               # check samples per move
                "tilt_margin_deg": 2.0, # predicted drone tilt <= the controller's tilt limit minus this
                "motor_line": 0.90,     # predicted thrust share |F| / (4 f_max) <= this (None: no check)
                "slack_min": 0.25,      # predicted string tension >= this x the static tension
                "vertical": True,       # vertical axis with its model (False: no vertical modes)
                "form": False,          # formation modes in the model (True: the identified formation lag modes too;
                                        # inverted they give references that overshoot by 0.1-0.4 m: not used)
                "model": None,          # None: CATCH_NETDYN_MODEL for the thrust (x1 / x5); or such a table
                "m_net_model": 0.020,   # kg, net mass assumed by the checks (the identified net; None: the actual one)
                "t_min_s": 0.06, "t_max_s": 8.0, "n_t": 140}
CATCH_NETDYN_CQ = 0.018                 # N s^2/m^2, net air drag carried by the strings (identification regression)
CATCH_NETDYN_Q_ACC = 0.5                # m/s^2, Kalman filter process noise per mode ("measured")
_ND_CACHE = {}


def _nd_modes(table, n, part):
    rows = table[min(n, len(table) - 1)]
    return [tuple(map(float, m)) for m in rows.get(part, [])]


class NDModel:
    """Linear model of one axis class, a cascade: formation modes (formation-centre offset from the reference, driven
    by u = [r'', r']) and net modes (net-centre offset from the formation centre, driven by the formation's
    acceleration and velocity). State = [d_i, d_i'] per mode, formation modes first. Outputs (offsets from the
    reference): formation centre Cf, net centre Cn (= formation offset + net offset from the formation); rates (C.d)
    and accelerations (C.a s + D.a u)."""

    def __init__(self, nm, fm):
        self.nm, self.fm = list(nm), list(fm)
        nf_, nn_ = 2 * len(self.fm), 2 * len(self.nm)
        ns = nf_ + nn_
        A = np.zeros((ns, ns))
        B = np.zeros((ns, 2))
        for i, (w, z, g, d) in enumerate(self.fm + self.nm):
            A[2 * i, 2 * i + 1] = 1.0
            A[2 * i + 1, 2 * i] = -w * w
            A[2 * i + 1, 2 * i + 1] = -2 * z * w
        Cf, Cfd = np.zeros(ns), np.zeros(ns)
        Cf[0:nf_:2], Cfd[1:nf_:2] = 1.0, 1.0
        Cfa, Dfa = np.zeros(ns), np.zeros(2)
        for i, (w, z, g, d) in enumerate(self.fm):
            B[2 * i + 1] = (-g, -d)
            Cfa += A[2 * i + 1]
            Dfa += B[2 * i + 1]
        # net modes: input formation acceleration a_f = r'' + Cfa s + Dfa u, velocity v_f = r' + Cfd s
        for i, (w, z, g, d) in enumerate(self.nm):
            j = nf_ + 2 * i + 1
            A[j] += -g * Cfa - d * Cfd
            B[j] = -g * (np.array([1.0, 0.0]) + Dfa) - d * np.array([0.0, 1.0])
        self.A, self.B, self.ns = A, B, ns
        Cr, Crd, Cra, Dra = np.zeros(ns), np.zeros(ns), np.zeros(ns), np.zeros(2)
        Cr[nf_::2], Crd[nf_ + 1::2] = 1.0, 1.0
        for j in range(nf_ + 1, ns, 2):
            Cra += A[j]
            Dra += B[j]
        self.Cf, self.Cfd, self.Cfa, self.Dfa = Cf, Cfd, Cfa, Dfa              # formation offset from the reference
        self.Cr, self.Crd = Cr, Crd                                            # net offset from the formation
        self.Cn, self.Cnd = Cf + Cr, Cfd + Crd                                 # net offset from the reference
        self.Cna, self.Dna = Cfa + Cra, Dfa + Dra
        self.key = (tuple(self.nm), tuple(self.fm))

    def disc(self, h):
        """Exact discretisation for a step h with the input held (ZOH)."""
        from scipy.linalg import expm
        M = np.zeros((self.ns + 2, self.ns + 2))
        M[:self.ns, :self.ns], M[:self.ns, self.ns:] = self.A, self.B
        E = expm(M * h)
        return E[:self.ns, :self.ns], E[:self.ns, self.ns:]


def _nd_tables(model, K, ns_s, Tg, nf=240):
    """Per grid time T: linear maps from x = [p0, v0, a0, p1, s0 (model state)] to the polynomial coefficients (in
    sigma = tau / T) and to the sampled reference (pos, vel, acc), net-centre offset, net-centre and formation
    accelerations. The model must end at rest (all states 0) with the reference at p1. Cached per model."""
    key = (model.key, K, ns_s, tuple(np.round(Tg, 9)), nf)
    if key in _ND_CACHE:
        return _ND_CACHE[key]
    from scipy.linalg import expm
    ns = model.ns
    nx = 4 + ns
    kk = np.arange(K + 1)
    H = np.zeros((K + 1, K + 1))                    # acceleration energy in sigma
    for k in range(2, K + 1):
        for l in range(2, K + 1):
            H[k, l] = k * (k - 1) * l * (l - 1) / (k + l - 3)
    H += 1e-9 * np.eye(K + 1)
    sig_s = np.linspace(0.0, 1.0, ns_s)
    sig_f = (np.arange(nf) + 0.5) / nf
    idx_s = np.round(sig_s * nf).astype(int)

    def pw(s_, d):
        return np.where(kk[None] >= d, np.prod([kk[None] - j for j in range(d)], axis=0) if d else 1.0, 0.0) \
            * s_[:, None] ** np.maximum(kk[None] - d, 0)
    V0, V1, V2 = pw(sig_s, 0), pw(sig_s, 1), pw(sig_s, 2)
    W1, W2 = pw(sig_f, 1), pw(sig_f, 2)
    out = {k: [] for k in ("Ma", "Mpos", "Mvel", "Macc", "Mne", "Mna", "Mfa")}
    Mx = np.zeros((ns + 2, ns + 2))
    Mx[:ns, :ns], Mx[:ns, ns:] = model.A, model.B
    for T in Tg:
        E = expm(Mx * (T / nf))
        Ph, Gh = E[:ns, :ns], E[:ns, ns:]
        Sa = np.zeros((ns, K + 1))
        Ss = np.eye(ns)
        rec_a, rec_s = [None] * ns_s, [None] * ns_s
        j_s = 0
        for j in range(nf + 1):
            while j_s < ns_s and idx_s[j_s] == j:
                rec_a[j_s], rec_s[j_s] = Sa.copy(), Ss.copy()
                j_s += 1
            if j == nf:
                break
            Sa = Ph @ Sa + Gh @ np.vstack([W2[j] / T ** 2, W1[j] / T])
            Ss = Ph @ Ss
        Eq = np.zeros((6 + ns, K + 1))
        L = np.zeros((6 + ns, nx))
        Eq[0, 0], L[0, 0] = 1.0, 1.0
        Eq[1, 1], L[1, 1] = 1.0, T
        Eq[2, 2], L[2, 2] = 2.0, T * T
        Eq[3], L[3, 3] = 1.0, 1.0
        Eq[4] = kk
        Eq[5] = kk * (kk - 1)
        Eq[6:] = Sa
        L[6:, 4:] = -Ss
        KKT = np.block([[2 * H, Eq.T], [Eq, np.zeros((6 + ns, 6 + ns))]])
        Ma = np.linalg.lstsq(KKT, np.vstack([np.zeros((K + 1, nx)), L]), rcond=None)[0][:K + 1]
        St = np.einsum("ijk,kx->ijx", np.array(rec_a), Ma)
        St[:, :, 4:] += np.array(rec_s)
        acc = V2 @ Ma / T ** 2
        vel = V1 @ Ma / T
        out["Ma"].append(Ma)
        out["Mpos"].append(V0 @ Ma)
        out["Mvel"].append(vel)
        out["Macc"].append(acc)
        out["Mne"].append(np.einsum("j,ijx->ix", model.Cn, St))
        out["Mna"].append(np.einsum("j,ijx->ix", model.Cna, St) + model.Dna[0] * acc + model.Dna[1] * vel + acc)
        out["Mfa"].append(np.einsum("j,ijx->ix", model.Cfa, St) + model.Dfa[0] * acc + model.Dfa[1] * vel + acc)
    res = {k: np.array(v) for k, v in out.items()}
    res["Tg"] = np.asarray(Tg, float)
    _ND_CACHE[key] = res
    return res


class NDSeg:
    """Reference segment of a net-dynamics move: polynomial in tau (coefficients (K + 1, 3), offset from C0)."""

    def __init__(self, t0, T, c):
        self.t0, self.T, self.c = t0, T, c

    def eval(self, t):
        tau = min(max(t - self.t0, 0.0), self.T)
        c = self.c
        K = len(c) - 1
        pos = c[K].copy()
        for k in range(K - 1, -1, -1):
            pos = pos * tau + c[k]
        if t - self.t0 >= self.T:
            return pos, np.zeros(3), np.zeros(3)
        vel = K * c[K]
        for k in range(K - 1, 0, -1):
            vel = vel * tau + k * c[k]
        acc = K * (K - 1) * c[K]
        for k in range(K - 1, 1, -1):
            acc = acc * tau + k * (k - 1) * c[k]
        return pos, vel, acc


def nd_eval(tabs, X):
    """X: {axis: (N, nx)} -> per axis dict of arrays (N, nT, ns)."""
    return {ax: {k: np.einsum("tsx,nx->nts", tabs[ax][k], x) for k in ("Mpos", "Mvel", "Macc", "Mna", "Mfa")}
            for ax, x in X.items()}


def nd_checks(P, lim, a0, phys, box, tol, floor):
    """Feasibility (N, nT) of the predicted moves P (nd_eval, axes 0, 1, 2), predicted peak tilt and thrust share."""
    pos = np.stack([P[a]["Mpos"] for a in range(3)], axis=-1)       # (N, nT, ns, 3)
    vel = np.stack([P[a]["Mvel"] for a in range(3)], axis=-1)
    acc = np.stack([P[a]["Macc"] for a in range(3)], axis=-1)
    an = np.stack([P[a]["Mna"] for a in range(3)], axis=-1)          # net-centre acceleration
    af = np.stack([P[a]["Mfa"] for a in range(3)], axis=-1)          # formation-centre acceleration
    e = 1.01
    ah0, az0, at0 = np.hypot(a0[0], a0[1]), abs(a0[2]), np.linalg.norm(a0)
    ok = ((np.hypot(vel[..., 0], vel[..., 1]) <= lim["v_h"] * e) & (vel[..., 2] <= lim["v_up"] * e)
          & (-vel[..., 2] <= lim["v_down"] * e)
          & (np.hypot(acc[..., 0], acc[..., 1]) <= max(lim["a_h"], ah0) * e)
          & (np.abs(acc[..., 2]) <= max(lim["a_v"], az0) * e)
          & (np.linalg.norm(acc, axis=-1) <= max(lim["a_tot"], at0) * e)).all(axis=2)
    g = np.array([0.0, 0.0, G])
    # strings (static directions U, corner to hook): min-norm tensions carrying m_tot (a_net + g); slack below
    # slack_min x the static tension
    T = np.einsum("kj,ntsj->ntsk", phys["Upinv"], phys["m_tot"] * (an + g))
    ok &= (T.min(axis=(2, 3)) >= phys["slack_min"] * phys["T0"])
    # drones: thrust F_k = m_d (a_formation + g) + T_k u_k
    tilt_max, mot_max = np.zeros(ok.shape), np.zeros(ok.shape)
    for k in range(4):
        F = phys["m_d"] * (af + g) + T[..., k, None] * phys["U"][:, k]
        tilt_max = np.maximum(tilt_max, np.degrees(np.arctan2(np.hypot(F[..., 0], F[..., 1]), F[..., 2])).max(axis=2))
        mot_max = np.maximum(mot_max, (np.linalg.norm(F, axis=-1) / (4 * phys["f_max"])).max(axis=2))
    ok &= tilt_max <= phys["tilt_lim"]
    if phys["motor_line"] is not None:
        ok &= mot_max <= phys["motor_line"]
    ok &= (np.all(pos >= box[0] - tol, axis=(2, 3)) & np.all(pos <= box[1] + tol, axis=(2, 3))
           & np.all(pos[..., 2] >= floor, axis=2))
    return ok, tilt_max, mot_max


def nd_first(ok):
    """Index of the first feasible grid time per row (-1: none)."""
    return np.where(ok.any(axis=1), ok.argmax(axis=1), -1)


def catch_box_limits(spacing):
    """Formation-centre box (sim coordinates) that keeps the guards 0.3 m from the side walls and the tops 0.3 m below
    the ceiling (the floor margin is the net-centre height range)."""
    lat = spacing / 2 + DRONE_R + 0.3
    lo = np.array([lat, lat, 0.0]) - BOX_OFFSET
    hi = np.array([BOX_SIZE - lat, BOX_SIZE - lat, BOX_SIZE - 0.3 - DRONE_HALF_H]) - BOX_OFFSET
    return lo, hi


class CatchPlan(Plan):
    """Takeoff and spread as in the hold scenario, a move to the standby point, then interception of the balls.

    observe() (100 Hz): measurements, one EKF per ball, the planner's view of each ball (flight / arrived / in_net /
    lost), replanning at replan_hz: for every ball in flight, candidate interception points on its predicted
    descending branch (net-centre heights z_net_range, aim height = net centre + aim_alpha * drone-to-net-centre
    drop, the ball must cross the rotor plane within r_in of the formation centre and its ascending branch must stay
    clear_up away), the shortest quintic from the current reference state to each (limits), then a search over the
    order of the balls (each ball caught at most once, in time order; t_hold_s of standstill after each catch unless
    the next point is within 0.1 m) for the plan with the most catches, ties by the margins (time to spare, capped
    at 0.8 s per catch) minus 0.3 per m of height change; the formation goes to the first point of that plan as fast
    as the limits allow (arrival at least t_early_s before the ball). Hysteresis: the current choice is kept unless
    the best plan is better by hyst. No ball in flight: back to the standby point. pre_step() (1 kHz): launches,
    ball air drag, net contact forces, contacts and clearances of the balls."""

    def __init__(self, p, cfg, launches, standby, h_dn, ff, spacing, t_go0=13.0, t_arm=0.5):
        self.p, self.cfg, self.ff, self.spacing = p, cfg, ff, spacing
        p2 = cfg.get("p2p")
        t_last = p2["t_move"] if p2 is not None else max(l["t"] for l in launches)
        super().__init__([(t_arm, 4.5, 0.012, p.z_hover)], [(6.5, 12.5, p.d_takeoff, spacing)],
                         t_end=t_last + (cfg["t_after_last_max_s"] if p2 is None else p2.get("t_after_s", 6.0)),
                         t_arm=t_arm)
        self.p2p_moved, self.p2p_t, self.p2p_level = False, None, False
        self.C0 = np.array([0.0, 0.0, p.z_hover])
        self.pattern = np.array([[sx * spacing / 2, sy * spacing / 2, 0.0] for sx, sy in CORNERS])
        self.t_go0, self.L, self.nb = t_go0, launches, len(launches)
        self.h_dn0, self.dh = h_dn, cfg["h_dn_per_ball_m"] or 0.0
        self.standby = np.asarray(standby, float)
        self.k = ball_k(p)
        lim = cfg["limits"]
        self.lim0 = dict(CATCH_LIMITS_SETS[lim] if isinstance(lim, str) else {**CATCH_LIMITS, **lim})
        self.box = catch_box_limits(spacing)
        self.seg, self.target, self.t_arrive_plan = None, None, t_go0
        self.t_rest_plan, self.v1_plan = t_go0, None
        self.choice_hist = [[] for _ in launches]
        self.done, self.sim, self.abort = False, None, None
        self.rng = np.random.default_rng(10_000 + cfg["seed"])
        nb = self.nb
        self.launched = np.zeros(nb, bool)
        self.t_launch = np.full(nb, np.nan)
        self.status = ["wait"] * nb
        self.t_status = np.zeros(nb)
        self.ekf = [None] * nb
        self.plan_hist = [[] for _ in range(nb)]
        self.pred_hist = [[] for _ in range(nb)]
        self.side = np.ones(nb)
        self.contact = np.zeros(nb, bool)
        self.fn = np.zeros(nb)
        self.vrel = np.full(nb, np.nan)
        self.contact_any = np.zeros(nb, bool)       # latched between two observe() calls (1 kHz -> 100 Hz)
        self.fn_max = np.zeros(nb)
        self.shell_hits = [[] for _ in range(nb)]  # (t, drone)
        self.gap_disc = np.full(nb, np.inf)
        self.gap_shell = np.full(nb, np.inf)
        self.gap_string = np.full(nb, np.inf)
        self.hold_until, self.next_replan = -1.0, t_go0
        self.chosen, self.n_net = None, 0
        self.ff_add = np.zeros((4, 3))
        self.tri = net_triangles(p.n)
        self.h_cell = p.L / (p.n - 1)
        self.ncon_other = 0
        self.lg = {k: [] for k in ("ball_pos", "ball_vel", "ball_meas", "ball_est", "ball_contact", "ball_fn",
                                   "ball_vrel", "target", "t_arrive_plan", "chosen", "n_net_est", "ncon_other", "ref_c")}
        self.events = []
        self.t_end_req = None
        self.replan_cost = 0.0
        self.n_replan = 0
        self.n_brake = 0
        self.n_brake_latched = 0
        self.n_brake_kept = 0
        self.n_floor_kept = 0
        self.brake_until = None
        self.zlo_hist = []                  # (t, height floor of the formation centre) at each new reference plan
        self.goal_z = None                  # height of the last planned goal (not a braking stop point)
        self.nd = None if cfg.get("netdyn") is None else {**CATCH_NETDYN, **(cfg["netdyn"] or {})}
        if self.nd is not None:
            self._nd_init()

    # ---- snapshot (state that changes while the plan runs; everything else follows from the configuration)
    _DYN = ("seg", "target", "t_arrive_plan", "t_rest_plan", "v1_plan", "done", "abort", "launched", "t_launch",
            "status", "t_status", "ekf", "plan_hist", "pred_hist", "choice_hist", "side", "contact", "fn", "vrel",
            "contact_any", "fn_max", "shell_hits", "gap_disc", "gap_shell", "gap_string", "hold_until",
            "next_replan", "chosen", "n_net", "ff_add", "ncon_other", "lg", "events", "t_end_req", "replan_cost",
            "n_replan", "n_brake", "p2p_moved", "p2p_t", "p2p_level", "nd_s", "nd_kf", "nd_t", "nd_n", "nd_ff",
            "nd_rng", "nd_stats", "nd_hist", "n_brake_latched", "n_brake_kept", "brake_until", "zlo_hist",
            "goal_z", "n_floor_kept")

    def snapshot_state(self):
        st = {k: getattr(self, k) for k in self._DYN if hasattr(self, k)}
        st["rng_state"] = self.rng.bit_generator.state
        return st

    def restore_state(self, st):
        st = dict(st)
        self.rng.bit_generator.state = st.pop("rng_state")
        for k, v in st.items():
            setattr(self, k, v)

    # ---- reference
    def _seg_eval(self, t):
        if self.seg is None:
            return np.zeros(3), np.zeros(3), np.zeros(3)
        if isinstance(self.seg, NDSeg):           # net-dynamics move (netdyn)
            return self.seg.eval(t)
        if isinstance(self.seg, list):           # moving catch: approach, coast, brake
            t0, T, b = [s for s in self.seg if t >= s[0]][-1] if t >= self.seg[0][0] else self.seg[0]
        else:
            t0, T, b = self.seg                  # b: (6, 3) offset polynomial in tau
        tau = min(max(t - t0, 0.0), T)
        pos = b[0] + tau * (b[1] + tau * (b[2] + tau * (b[3] + tau * (b[4] + tau * b[5]))))
        if t - t0 >= T:
            return pos, np.zeros(3), np.zeros(3)
        vel = b[1] + tau * (2 * b[2] + tau * (3 * b[3] + tau * (4 * b[4] + tau * 5 * b[5])))
        acc = 2 * b[2] + tau * (6 * b[3] + tau * (12 * b[4] + tau * 20 * b[5]))
        return pos, vel, acc

    def ref(self, t):
        pos, vel, d = Plan.ref(self, t)
        off, voff, aoff = self._seg_eval(t)
        ms = self.cfg.get("move_spacing")
        if ms is not None and t >= self.t_go0:
            # spacing change while moving: spacing + (move spacing - spacing) * smoothstep(horizontal speed / v_full)
            sp = math.hypot(voff[0], voff[1])
            x = min(sp / ms["v_full"], 1.0)
            dd_ = (ms["spacing"] - self.spacing)
            dsp = dd_ * (3 * x ** 2 - 2 * x ** 3)
            rate = (dd_ * (6 * x - 6 * x ** 2) * (voff[0] * aoff[0] + voff[1] * aoff[1]) / max(sp, 1e-9) / ms["v_full"]
                    if 0.0 < x < 1.0 else 0.0)
            pos = pos + np.array([[sx * dsp / 2, sy * dsp / 2, 0.0] for sx, sy in CORNERS])
            vel = vel + np.array([[sx * rate / 2, sy * rate / 2, 0.0] for sx, sy in CORNERS])
            d = d + dsp
        return pos + off, vel + voff, d

    def acc(self, t):
        a = np.zeros((4, 3))
        a[:] = self._seg_eval(t)[2]
        return a

    # ---- net dynamics (cfg netdyn)
    def _nd_init(self):
        p, nd = self.p, self.nd
        tab = nd["model"]
        if tab is None:
            tab = CATCH_NETDYN_MODEL["x5" if p.f_max > 2.0 * OfficialParams().f_max else "x1"]
        self.nd_table = tab
        self.nd_Tg = np.geomspace(nd["t_min_s"], nd["t_max_s"], int(nd["n_t"]))
        self.nd_models, self.nd_disc = {}, {}
        self.nd_n = 0
        self.nd_s = [np.zeros(self._nd_model(0, ax).ns) for ax in range(3)]
        self.nd_kf = None
        if nd["state"] == "measured":           # model state + offsets of the net and formation measurements
            self.nd_kf = [np.zeros(self._nd_model(0, ax).ns + 2) for ax in range(3)]
        self.nd_t = None
        self.nd_ff = np.zeros(3)
        self.nd_rng = np.random.default_rng(20_000 + int(self.cfg["seed"]))
        self.nd_stats = {"go": 0, "fallback": 0, "replan_nd": 0, "err_n": 0, "err_model_ss": 0.0, "err_kf_ss": 0.0,
                         "err_model_max": 0.0, "err_kf_max": 0.0}
        self.nd_hist = []

    def _nd_model(self, n, ax):
        nd = self.nd
        if ax == 2 and not nd["vertical"]:
            parts = (None, None)
        else:
            parts = ("v", "fv") if ax == 2 else ("h", "fh")
        key = (n, parts)
        if key not in self.nd_models:
            nm = _nd_modes(self.nd_table, n, parts[0]) if parts[0] else []
            fm = _nd_modes(self.nd_table, n, parts[1]) if (parts[1] and nd["form"]) else []
            self.nd_models[key] = NDModel(nm, fm)
        return self.nd_models[key]

    def _nd_tabs(self, n=None):
        n = self.nd_n if n is None else n
        return {ax: _nd_tables(self._nd_model(n, ax), int(self.nd["K"]), int(self.nd["ns"]), self.nd_Tg)
                for ax in range(3)}

    def _nd_phys(self, n=None):
        n = self.nd_n if n is None else n
        p = self.p
        m_net = hanging_mass(p) if self.nd["m_net_model"] is None else float(self.nd["m_net_model"])
        m_tot = m_net + n * p.ball_m
        pull = (self.ff[0](self.spacing) if self.ff is not None else 0.0) * m_tot / hanging_mass(p)
        U = np.array([[sx * pull / np.sqrt(2), sy * pull / np.sqrt(2), m_tot * G / 4] for sx, sy in CORNERS]).T
        T0 = float(np.linalg.norm(U[:, 0]))
        U = U / T0
        return {"m_d": p.m_drone, "m_tot": m_tot, "pull": pull, "f_max": p.f_max, "U": U, "Upinv": np.linalg.pinv(U),
                "T0": T0, "tilt_lim": p.max_tilt_deg - self.nd["tilt_margin_deg"], "motor_line": self.nd["motor_line"],
                "slack_min": self.nd["slack_min"]}

    def _nd_state(self):
        """Model state per axis for a replan: open-loop model, or the Kalman estimate ("measured")."""
        if self.nd_kf is not None:
            return [kf[:-2].copy() for kf in self.nd_kf]
        return [s_.copy() for s_ in self.nd_s]

    def _nd_x(self, c0, v0, a0, P1, S0):
        """{axis: (N, nx)} for targets P1 (N, 3) from the reference state and the model state."""
        P1 = np.atleast_2d(P1)
        N_ = len(P1)
        return {ax: np.column_stack([np.full(N_, c0[ax]), np.full(N_, v0[ax]), np.full(N_, a0[ax]), P1[:, ax],
                                     np.repeat(S0[ax][None], N_, axis=0)]) for ax in range(3)}

    def _nd_feasible(self, t, P1, lim, c0, v0, a0, tol, floor):
        tabs = self._nd_tabs()
        X = self._nd_x(c0, v0, a0, P1, self._nd_state())
        P = nd_eval(tabs, X)
        ok, tl, mo = nd_checks(P, lim, a0, self._nd_phys(), self.box, tol, floor)
        return tabs, X, ok, tl, mo

    def _nd_step(self, t, P, net_c):
        """Propagate the model to t (input: the formation reference); "measured": Kalman update with the measured
        net centre and formation centre (offsets from the reference); then the optional string-force feedforward."""
        if self.nd_t is None:
            self.nd_t = t
            return
        h = t - self.nd_t
        if h <= 0:
            return
        self.nd_t = t
        _, vr, ar = self.ref_centre(t - h)
        n = self.nd_n
        rc = self.ref_centre(t)[0]
        for ax in range(3):
            mdl = self._nd_model(n, ax)
            key = (mdl.key, round(h, 6))
            if key not in self.nd_disc:
                self.nd_disc[key] = mdl.disc(h)
            Ph, Gh = self.nd_disc[key]
            u = np.array([ar[ax], vr[ax]])
            self.nd_s[ax] = Ph @ self.nd_s[ax] + Gh @ u
            if self.nd_kf is not None:
                x = self.nd_kf[ax]
                xs = np.concatenate([Ph @ x[:-2] + Gh @ u, x[-2:]])
                Kg, Hm = self._nd_gain(mdl, h)
                y = np.array([net_c[ax] + self.nd_rng.normal(0.0, self.nd["meas_noise_m"]) - rc[ax],
                              P[:, ax].mean() + self.nd_rng.normal(0.0, self.p.sigma_p / 2.0) - rc[ax]])
                self.nd_kf[ax] = xs + Kg @ (y - Hm @ xs)
        if t >= self.t_go0:             # statistics: horizontal net-centre offset from the reference, state vs truth
            st = self.nd_stats
            tru = net_c[:2] - rc[:2]
            em = np.hypot(*(np.array([self._nd_model(n, ax).Cn @ self.nd_s[ax] for ax in (0, 1)]) - tru))
            st["err_n"] += 1
            st["err_model_ss"] += float(em * em)
            st["err_model_max"] = max(st["err_model_max"], float(em))
            if self.nd_kf is not None:
                ek = np.hypot(*(np.array([self._nd_model(n, ax).Cn @ self.nd_kf[ax][:-2] + self.nd_kf[ax][-2]
                                          for ax in (0, 1)]) - tru))
                st["err_kf_ss"] += float(ek * ek)
                st["err_kf_max"] = max(st["err_kf_max"], float(ek))
        ff = np.zeros(3)
        if self.nd["ff"]:               # (m_tot a_net + c |v_net| v_net) / 4 per drone
            S = self._nd_state()
            _, vr, ar = self.ref_centre(t)
            m_tot = hanging_mass(self.p) + n * self.p.ball_m
            a_n, v_n = np.zeros(3), np.zeros(3)
            for ax in range(3):
                mdl = self._nd_model(n, ax)
                a_n[ax] = ar[ax] + mdl.Cna @ S[ax] + mdl.Dna @ np.array([ar[ax], vr[ax]])
                v_n[ax] = vr[ax] + mdl.Cnd @ S[ax]
            ff = ff + (m_tot * a_n + CATCH_NETDYN_CQ * np.linalg.norm(v_n) * v_n) / 4.0
        if self.nd["drag_ff"]:          # body drag (drag < 0: opposing) and the net's share at the reference velocity
            vr_ = self._seg_eval(t)[1]
            dr = np.array(getattr(self.p, "drag", (0.0, 0.0, 0.0)), float) if is_official(self.p) else np.zeros(3)
            ff = ff - dr * vr_ + CATCH_NETDYN_CQ * np.linalg.norm(vr_) * vr_ / 4.0
        if self.nd["lead_s"]:           # the attitude / rotor lag: acceleration taken from lead_s ahead
            ar0 = self._seg_eval(t)[2]
            ar1 = self._seg_eval(t + self.nd["lead_s"])[2]
            ff = ff + self.p.m_drone * (ar1 - ar0)
        self.nd_ff = ff

    def _nd_gain(self, mdl, h):
        """Steady-state Kalman gain: model state plus constant offsets of the two measurements (net centre, formation
        centre, both relative to the reference)."""
        key = ("kf", mdl.key, round(h, 6), self.nd["meas_noise_m"])
        if key in self.nd_disc:
            return self.nd_disc[key]
        Ph, Gh = self.nd_disc[(mdl.key, round(h, 6))]
        ns = mdl.ns
        F = np.eye(ns + 2)
        F[:ns, :ns] = Ph
        Hm = np.zeros((2, ns + 2))
        Hm[0, :ns], Hm[0, ns] = mdl.Cn, 1.0
        Hm[1, :ns], Hm[1, ns + 1] = mdl.Cf, 1.0
        Q = np.zeros((ns + 2, ns + 2))
        for i in range(ns // 2):
            Q[2 * i + 1, 2 * i + 1] = (CATCH_NETDYN_Q_ACC * h) ** 2
        Q[ns, ns] = Q[ns + 1, ns + 1] = 1e-8
        R = np.diag([self.nd["meas_noise_m"] ** 2, (self.p.sigma_p / 2.0) ** 2])
        Pc = np.eye(ns + 2) * 1e-4
        Kg = np.zeros((ns + 2, 2))
        for _ in range(3000):
            Pp = F @ Pc @ F.T + Q
            Kg = Pp @ Hm.T @ np.linalg.inv(Hm @ Pp @ Hm.T + R)
            Pc = Pp - Kg @ Hm @ Pp
        self.nd_disc[key] = (Kg, Hm)
        return Kg, Hm

    def _nd_switch(self, n):
        """Ball count changed: map the model state (net-centre and formation offsets and rates, put on the first mode
        of each group) to the new model."""
        if n == self.nd_n:
            return
        for ax in range(3):
            old, new = self._nd_model(self.nd_n, ax), self._nd_model(n, ax)
            for which in ("s", "kf"):
                if which == "kf" and self.nd_kf is None:
                    continue
                src = self.nd_s[ax] if which == "s" else self.nd_kf[ax][:-2]
                dst = np.zeros(new.ns)
                if new.fm and old.fm:
                    dst[0], dst[1] = old.Cf @ src, old.Cfd @ src
                if new.nm and old.nm:
                    i0 = 2 * len(new.fm)
                    dst[i0], dst[i0 + 1] = old.Cr @ src, old.Crd @ src
                if which == "s":
                    self.nd_s[ax] = dst
                else:
                    self.nd_kf[ax] = np.concatenate([dst, self.nd_kf[ax][-2:]])
        self.nd_n = n

    def _go_netdyn(self, t, target, lim, c0, v0, a0, zfl, tol, floor):
        tabs, X, ok, tl, mo = self._nd_feasible(t, target[None], lim, c0, v0, a0, tol, floor)
        i = int(nd_first(ok)[0])
        self.nd_stats["go"] += 1
        if i < 0:
            self.nd_stats["fallback"] += 1
            return False
        T = float(self.nd_Tg[i])
        coef = np.zeros((int(self.nd["K"]) + 1, 3))
        for ax in range(3):
            a = tabs[ax]["Ma"][i] @ X[ax][0]
            coef[:, ax] = a / T ** np.arange(len(a))
        coef[0] -= self.C0
        self.zlo_hist.append((t, zfl))
        self.goal_z = float(target[2])
        self.seg = NDSeg(t, T, coef)
        self.target, self.t_arrive_plan = target, t + T
        self.t_rest_plan, self.v1_plan = t + T, None
        if len(self.nd_hist) < 4000:
            self.nd_hist.append((round(t, 3), round(T, 3), round(float(tl[0, i]), 1), round(float(mo[0, i]), 3)))
        return True

    @staticmethod
    def _seg_eval_list(segs, t):
        t0, T, b = [s for s in segs if t >= s[0]][-1] if t >= segs[0][0] else segs[0]
        tau = min(max(t - t0, 0.0), T)
        return (b[0] + tau * (b[1] + tau * (b[2] + tau * (b[3] + tau * (b[4] + tau * b[5])))),)

    def ref_centre(self, t):
        off, voff, aoff = self._seg_eval(t)
        return self.C0 + off, voff, aoff

    def lim(self):
        f = self.cfg["loaded_accel_factor"] if self.n_net > 0 else 1.0
        return {k: (v * f if k.startswith("a_") else v) for k, v in self.lim0.items()}

    def h_dn(self, n=None):
        n = self.n_net if n is None else n
        tab = self.cfg.get("h_dn_table")
        if tab:
            step = tab[-1] - tab[-2] if len(tab) > 1 else 0.0
            return self.h_dn0 + (tab[n] if n < len(tab) else tab[-1] + (n - len(tab) + 1) * step)
        return self.h_dn0 + self.dh * n

    def _go(self, t, target, scale=1.0):
        bfv = self.cfg.get("boxfix", True)
        bf = isinstance(bfv, dict) or bool(bfv)         # off only with false / None / 0
        bfo = {**CATCH_BOXFIX, **(bfv if isinstance(bfv, dict) else {})}
        if bf and self.brake_until is not None:     # boxfix: keep the braking plan until it reaches rest
            if t < self.brake_until - 1e-9:
                return
            self.brake_until = None
        target = np.clip(np.asarray(target, float), self.box[0], self.box[1])
        c0, v0, a0 = self.ref_centre(t)
        # height floor of the formation centre: net centre at the lowest candidate height with the balls in the net now,
        # or lower where the reference already is or was heading (the last planned goal: after a catch the floor rises
        # by the new ball's sink while the formation may still be finishing its descent)
        zfl = min(self.cfg["z_net_range"][0] + self.h_dn(), float(c0[2]),
                  float(self.goal_z) if self.goal_z is not None else np.inf)
        tol = bfo["tol_m"] if bf else 0.05
        fl = zfl - bfo["floor_tol_m"] if (bf and bfo["floor"]) else -np.inf

        def bad(path):                          # (leaves the box, dips below the height floor (boxfix only))
            return (bool(np.any(path < self.box[0] - tol) or np.any(path > self.box[1] + tol)),
                    bool(np.any(path[:, 2] < fl)))

        if self.target is not None and np.linalg.norm(target - self.target) < 0.003:
            return
        if np.linalg.norm(target - c0) < 1e-4 and np.linalg.norm(v0) < 1e-4 and np.linalg.norm(a0) < 1e-4:
            self.target = target
            return
        if np.linalg.norm(v0) > 0.5:          # gentle limits (standby return) only from near rest: braking first
            scale = 1.0
        lim = {k: (v * scale if k.startswith("a_") else v) for k, v in self.lim().items()}
        if self.nd is not None and self.nd["plan"]:     # net-dynamics move (falls back to the moves below)
            if self._go_netdyn(t, target, lim, c0, v0, a0, zfl, tol, fl):
                return
        if self.cfg.get("traj") is not None:            # jerk-limited straight move (falls back to the quintic below)
            sc = scurve_segments(t, c0, v0, a0, target, lim, self.cfg["traj"], self.C0)
            if sc is not None:
                segs, T = sc
                ts = np.linspace(t, t + T, 41)
                path = np.array([self.C0 + self._seg_eval_list(segs, x)[0] for x in ts])
                if not any(bad(path)):
                    self.zlo_hist.append((t, zfl))
                    self.goal_z = float(target[2])
                    self.seg = segs
                    self.target, self.t_arrive_plan = target, t + T
                    self.t_rest_plan, self.v1_plan = t + T, None
                    return
        T = quintic_tmin_fine(c0, v0, a0, target, lim)
        c3, c4, c5 = _quintic(c0, v0, a0, target - c0, T)
        tau = np.linspace(0.0, T, 41)[:, None]
        path = c0 + v0 * tau + 0.5 * a0 * tau ** 2 + c3 * tau ** 3 + c4 * tau ** 4 + c5 * tau ** 5
        out_box, low = bad(path)
        if out_box:
            # the new target lies behind the current motion and the path would leave the box: stop ahead first
            # (min-jerk braking distance 0.5 v T_b, T_b = 1.875 v / a), the next replans continue from rest
            if bf and bfo["horizontal_stop"]:   # horizontal stop point at the current height (not below the floor)
                vh = float(np.linalg.norm(v0[:2]))
                stop = c0 + np.array([v0[0], v0[1], 0.0]) * 0.5 * 1.875 * vh / lim["a_h"]
                stop[2] = min(max(c0[2], zfl), self.box[1][2])
                target = np.clip(stop, self.box[0], self.box[1])
            else:
                vh = float(np.linalg.norm(v0))
                target = np.clip(c0 + v0 * 0.5 * 1.875 * vh / lim["a_h"], self.box[0], self.box[1])
            T = quintic_tmin_fine(c0, v0, a0, target, lim)
            c3, c4, c5 = _quintic(c0, v0, a0, target - c0, T)
            if bf and bfo["keep_plan"] and self.seg is not None:
                tau = np.linspace(0.0, T, 41)[:, None]
                bp_ = c0 + v0 * tau + 0.5 * a0 * tau ** 2 + c3 * tau ** 3 + c4 * tau ** 4 + c5 * tau ** 5
                if any(bad(bp_)):
                    # cannot stop inside the box from here: keep the current plan (checked inside the box when it
                    # was made); the next replans try again as it slows down
                    self.n_brake_kept = getattr(self, "n_brake_kept", 0) + 1
                    return
            self.n_brake = getattr(self, "n_brake", 0) + 1
            if bf and bfo["latch"]:
                self.brake_until = t + T
                self.n_brake_latched += 1
        elif low and self.seg is not None:
            # boxfix: the new path would dip below the height floor (a replan with a downward velocity): keep the
            # current plan (checked against the floor when it was made) and try again at the next replan
            self.n_floor_kept += 1
            return
        else:
            self.goal_z = float(target[2])
        self.zlo_hist.append((t, zfl))
        self.seg = (t, T, np.array([c0 - self.C0, v0, 0.5 * a0, c3, c4, c5]))
        self.target, self.t_arrive_plan = target, t + T
        self.t_rest_plan, self.v1_plan = t + T, None

    def _go_moving(self, t, target, v1, t_aim):
        """Interception with an end velocity: quintic to (target, v1) arriving at t_aim (or as soon as the limits
        allow), coast at v1 for coast_s, brake to rest at brake_factor x the limits. Falls back to _go (rest) when the
        approach would leave the box."""
        mv = {**CATCH_MOVING, **self.cfg["moving"]}
        c0, v0, a0 = self.ref_centre(t)
        lim = self.lim()
        T = max(t_aim - t, quintic_tmin_fine(c0, v0, a0, target, lim, v1=v1), 0.05)
        c3, c4, c5 = _quintic_v(c0, v0, a0, target - c0, T, v1)
        tau = np.linspace(0.0, T, 41)[:, None]
        path = c0 + v0 * tau + 0.5 * a0 * tau ** 2 + c3 * tau ** 3 + c4 * tau ** 4 + c5 * tau ** 5
        p1 = target + v1 * mv["coast_s"]
        limb = {k: (v * mv["brake_factor"] if k.startswith("a_") else v) for k, v in lim.items()}
        sp = float(np.linalg.norm(v1))
        p2 = np.clip(p1 + v1 * 0.5 * 1.875 * sp / max(limb["a_h"], 1e-6), self.box[0], self.box[1])
        if (np.any(path < self.box[0] - 0.05) or np.any(path > self.box[1] + 0.05)
                or np.any(p1 < self.box[0] - 0.05) or np.any(p1 > self.box[1] + 0.05)):
            self._go(t, target)
            return
        Tb = quintic_tmin_fine(p1, v1, np.zeros(3), p2, limb)
        b3, b4, b5 = _quintic(p1, v1, np.zeros(3), p2 - p1, Tb)
        t1 = t + T
        z = np.zeros(3)
        self.seg = [(t, T, np.array([c0 - self.C0, v0, 0.5 * a0, c3, c4, c5])),
                    (t1, mv["coast_s"], np.array([target - self.C0, v1, z, z, z, z])),
                    (t1 + mv["coast_s"], Tb, np.array([p1 - self.C0, v1, z, b3, b4, b5]))]
        self.target, self.t_arrive_plan = target, t1
        self.t_rest_plan, self.v1_plan = t1 + mv["coast_s"] + Tb, v1.copy()

    def standby_now(self):
        s = self.standby.copy()
        s[2] = self.cfg["z_net0"] + self.h_dn()
        return s

    # ---- simulation hooks
    def attach(self, sim):
        self.sim = sim
        m = sim.model
        nid = lambda t_, name: mujoco.mj_name2id(m, t_, name)
        B, Gm = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_GEOM
        self.b_body = np.array([nid(B, f"ball{b}") for b in range(self.nb)])
        self.b_q = [int(m.jnt_qposadr[m.body_jntadr[x]]) for x in self.b_body]
        self.b_v = [int(m.jnt_dofadr[m.body_jntadr[x]]) for x in self.b_body]
        self.b_geom = np.array([nid(Gm, f"ball{b}") for b in range(self.nb)])
        self.geom_ball = {int(g): b for b, g in enumerate(self.b_geom)}
        self.shell = {nid(Gm, f"shell{k}"): k for k in range(4)}
        n = self.p.n
        self.knot_body = np.array([nid(B, f"n{i}_{j}") for i in range(n) for j in range(n)])
        self.park = [np.array(CATCH_PARK) + np.array([-0.2 * b, 0.0, 0.0]) for b in range(self.nb)]
        self.rot = rotor_layout(self.p)

    def ball_state(self, b):
        d = self.sim.data
        return d.qpos[self.b_q[b]:self.b_q[b] + 3].copy(), d.qvel[self.b_v[b]:self.b_v[b] + 3].copy()

    def pre_step(self, t):
        sim, d, p = self.sim, self.sim.data, self.p
        for b in range(self.nb):
            if self.launched[b]:
                continue
            qa, va = self.b_q[b], self.b_v[b]
            if t >= self.L[b]["t"] - 1e-9:
                d.qpos[qa:qa + 3] = self.L[b]["p0"]
                self.launched[b], self.t_launch[b] = True, t
                self.status[b], self.t_status[b] = "flight", t
                d.qvel[va:va + 3] = self.L[b]["v0"]
            else:
                d.qpos[qa:qa + 3] = self.park[b]
                d.qvel[va:va + 3] = 0.0
            d.qpos[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
            d.qvel[va + 3:va + 6] = 0.0
        idx = np.flatnonzero(self.launched)
        if not len(idx):
            return
        bp = np.array([d.qpos[self.b_q[b]:self.b_q[b] + 3] for b in idx])
        bv = np.array([d.qvel[self.b_v[b]:self.b_v[b] + 3] for b in idx])
        sp = np.linalg.norm(bv, axis=1)
        d.xfrc_applied[self.b_body[idx], :3] += -(p.ball_m * self.k * sp)[:, None] * bv
        kx = sim.knots_now()
        r = p.ball_d / 2
        lo, hi = kx.min(axis=0) - 0.1, kx.max(axis=0) + 0.1
        near = np.all((bp >= lo) & (bp <= hi), axis=1)
        self.contact[:] = False
        self.fn[:] = 0.0
        self.vrel[:] = np.nan
        if near.any():
            ii = idx[near]
            kv = d.qvel[sim.knot_dof]
            fb, fk, act, fn, side_now, vr = net_contact_forces(kx, kv, bp[near], bv[near], self.side[ii], self.tri, r,
                                                           self.h_cell)
            d.xfrc_applied[self.b_body[ii], :3] += fb
            d.xfrc_applied[self.knot_body, :3] += fk
            self.contact[ii], self.fn[ii] = act, fn
            self.vrel[ii] = vr
            upd = ~act & np.isfinite(side_now)
            self.side[ii[upd]] = side_now[upd]
        self.contact_any |= self.contact
        self.fn_max = np.maximum(self.fn_max, self.fn)
        # balls near the drones: rotor discs, airframe cylinder, strings (geometry only; contacts below)
        Pd = np.array([d.qpos[q:q + 3] for q in sim.qadr])
        dd = np.linalg.norm(bp[:, None, :] - Pd[None], axis=2)
        close = dd.min(axis=1) < 0.4
        if close.any():
            for jb in np.flatnonzero(close):
                b = idx[jb]
                x = bp[jb]
                for kk in np.flatnonzero(dd[jb] < 0.4):
                    R = sim.drone_state(kk)[2]
                    rel = R.T @ (x - Pd[kk])                         # body frame
                    radial = math.hypot(rel[0], rel[1])
                    g_sh = max(radial - DRONE_R, abs(rel[2]) - DRONE_HALF_H) - r
                    self.gap_shell[b] = min(self.gap_shell[b], g_sh)
                    for c in self.rot:
                        v_ = rel - c
                        rho = math.hypot(v_[0], v_[1])
                        g_d = math.hypot(max(rho - p.prop_radius, 0.0), v_[2]) - r
                        self.gap_disc[b] = min(self.gap_disc[b], g_d)
                    hook = Pd[kk] + p.hook_z * R[:, 2]
                    ck = kx[sim.corner_knot[kk]]
                    sv = ck - hook
                    u = np.clip((x - hook) @ sv / max(sv @ sv, 1e-12), 0.0, 1.0)
                    self.gap_string[b] = min(self.gap_string[b], float(np.linalg.norm(x - hook - u * sv)) - r)
        nc = d.ncon
        if nc:
            g1, g2 = d.contact.geom1[:nc], d.contact.geom2[:nc]
            for a_, b_ in zip(g1, g2):
                a_, b_ = int(a_), int(b_)
                if a_ in self.shell and b_ in self.geom_ball:
                    self.shell_hits[self.geom_ball[b_]].append((t, self.shell[a_]))
                elif b_ in self.shell and a_ in self.geom_ball:
                    self.shell_hits[self.geom_ball[a_]].append((t, self.shell[b_]))
            balls = np.isin(g1, self.b_geom) | np.isin(g2, self.b_geom)
            self.ncon_other = int((~balls).sum())
        else:
            self.ncon_other = 0

    def ff_extra(self, t, k):
        if self.nd is not None and (self.nd["ff"] or self.nd["lead_s"] or self.nd["drag_ff"]):
            return self.ff_add[k] + self.nd_ff
        return self.ff_add[k]

    def _set_ff(self):
        n, p = self.n_net, self.p
        self.ff_add = np.zeros((4, 3))
        if not self.cfg["ff_balls"] or n == 0:
            return
        mb = n * p.ball_m
        pull = self.ff[0](self.spacing) * mb / hanging_mass(p) if self.ff is not None else 0.0
        for k, (sx, sy) in enumerate(CORNERS):
            self.ff_add[k] = (sx * pull / np.sqrt(2), sy * pull / np.sqrt(2), mb * G / 4)

    def observe(self, t, P, net_c, sag):
        sim, p, cfg = self.sim, self.p, self.cfg
        ref, _, _ = self.ref(t)
        if t > self.t_arm + 1.0 and self.abort is None:
            tilt = max(np.degrees(np.arccos(np.clip(sim.drone_state(k)[2][2, 2], -1, 1))) for k in range(4))
            zerr = float(np.abs(P[:, 2] - ref[:, 2]).max())
            if tilt > TRANSFER_CRIT["tilt_deg"] or zerr > TRANSFER_CRIT["alt_err_m"]:
                self.abort = {"t_s": round(t, 3), "tilt_deg": round(float(tilt), 1), "alt_err_m": round(zerr, 3)}
                self.done = True
        if self.nd is not None:
            self._nd_step(t, P, net_c)
        nb = self.nb
        bp, bv, meas, est = (np.full((nb, 3), np.nan) for _ in range(4))
        est = np.full((nb, 6), np.nan)
        fc = P.mean(axis=0)
        for b in range(nb):
            if not self.launched[b]:
                continue
            x, v = self.ball_state(b)
            bp[b], bv[b] = x, v
            z = x + (self.rng.normal(0.0, cfg["noise_m"], 3) if cfg["noise_m"] > 0 else 0.0)
            meas[b] = z
            if self.ekf[b] is None:
                vp = self.L[b]["v_prior"]
                self.ekf[b] = BallEKF(z, t, vp, cfg["ekf_sig_v0"], cfg["noise_m"], self.k, cfg["ekf_q_acc"])
            else:
                self.ekf[b].predict(t)
                self.ekf[b].update(z)
            est[b] = self.ekf[b].x
            # the planner's view of the ball (from measurements and the drones' own positions)
            st = self.status[b]
            rh = math.hypot(z[0] - fc[0], z[1] - fc[1])
            ra = CATCH_GEOM["arrive"] * self.spacing
            in_region = rh < ra and fc[2] - self.h_dn() - 0.35 < z[2] < fc[2] + 0.05
            if st == "flight":
                if z[2] < fc[2] - 0.03 and rh < ra:
                    self.status[b], self.t_status[b] = "arrived", t
                    self.events.append((round(t, 3), b, "arrived"))
                    self.hold_until = t + cfg["t_hold_s"]
                elif z[2] < 0.15 or (rh > ra and z[2] < cfg["z_net_range"][0] + cfg["aim_alpha"] * self.h_dn() - 0.1
                                     and self.ekf[b].x[5] < 0):
                    self.status[b], self.t_status[b] = "lost", t
                    self.events.append((round(t, 3), b, "lost"))
            elif st == "arrived" and t - self.t_status[b] > 0.3:
                self.status[b], self.t_status[b] = ("in_net" if in_region else "lost"), t
                self.events.append((round(t, 3), b, self.status[b]))
            elif st == "in_net" and not in_region:
                self.status[b], self.t_status[b] = "lost", t
                self.events.append((round(t, 3), b, "lost"))
        n_net = sum(s in ("arrived", "in_net") for s in self.status)
        if n_net != self.n_net:
            self.n_net = n_net
            self._set_ff()
            if self.nd is not None:
                self._nd_switch(n_net)
        if t >= self.next_replan - 1e-9 and t >= self.t_go0 - 1e-9:
            self.next_replan = t + 1.0 / cfg["replan_hz"]
            w0 = time.time()
            if self.seg is None:
                self._go(t, self.standby_now())
            else:
                self.replan(t)
            self.replan_cost += time.time() - w0
            self.n_replan += 1
        p2 = cfg.get("p2p")
        if p2 is not None and p2.get("n_drop") and not self.p2p_level and t >= p2["t_move"] - 2.0 - 1e-9:
            self._go(t, self.standby_now())     # balls dropped in: climb by their sink before the move
            self.p2p_level = True
        if p2 is not None and not self.p2p_moved and t >= p2["t_move"] - 1e-9:
            self._go(t, self.standby_now() + np.asarray(p2["d"], float))
            self.p2p_moved, self.p2p_t = True, t
        c_ref = self.ref_centre(t)[0]
        L = self.lg
        L["ball_pos"].append(bp)
        L["ball_vel"].append(bv)
        L["ball_meas"].append(meas)
        L["ball_est"].append(est)
        L["ball_contact"].append(self.contact_any.copy())
        L["ball_fn"].append(self.fn_max.copy())
        L["ball_vrel"].append(self.vrel.copy())
        L["target"].append(np.full(3, np.nan) if self.target is None else self.target.copy())
        L["t_arrive_plan"].append(self.t_arrive_plan)
        L["chosen"].append(-1 if self.chosen is None else self.chosen[0])
        L["n_net_est"].append(self.n_net)
        L["ncon_other"].append(self.ncon_other)
        L["ref_c"].append(c_ref)
        self.contact_any[:] = False
        self.fn_max[:] = 0.0
        # end: every ball launched and resolved (on the floor and slow, or in contact with the net for the last 1.5 s
        # and slow relative to it over the last 1 s), and the formation's reference at rest for 1 s
        if self.nb and cfg.get("p2p") is None and self.launched.all() and t > max(l["t"] for l in self.L) + 2.0 and t > self.t_rest_plan + 1.0:
            cont = np.array(L["ball_contact"][-150:])
            vr = np.array(L["ball_vrel"][-100:])
            ok = []
            for b in range(nb):
                on_floor = bp[b, 2] < p.ball_d / 2 + 0.01 and np.linalg.norm(bv[b]) < 0.3
                in_net = (len(cont) >= 150 and cont[:, b].all() and np.isfinite(vr[:, b]).all()
                          and vr[:, b].mean() < CATCH_REL_V)
                ok.append(on_floor or in_net)
            if all(ok):
                if self.t_end_req is None:
                    self.t_end_req = t + 0.2
                elif t >= self.t_end_req:
                    self.done = True
            else:
                self.t_end_req = None

    # ---- planning
    def _candidates(self, t, pend):
        cfg = self.cfg
        h = self.h_dn()
        zn = np.arange(cfg["z_net_range"][0], cfg["z_net_range"][1] + 1e-9, cfg["z_net_step"])
        lo, hi = self.box
        rows = []
        prev_trajs = getattr(self, "trajs", {})
        self.trajs = {}
        for b in [b for b in range(self.nb) if self.status[b] == "flight" and self.ekf[b] is not None
                  and self.ekf[b].n >= cfg["n_meas_min"]]:
            e = self.ekf[b]
            tr = flight_rk4(e.x, self.k, dt=0.01, n_max=700, z_stop=0.0)
            tr[:, 0] += e.t
            self.trajs[b] = tr
            if b not in pend:
                continue
            apex = tr[:, 3].max()
            ia = int(tr[:, 3].argmax())
            asc, desc = tr[:ia + 1], tr[ia:]
            if len(desc) < 2:
                continue
            zd = desc[::-1, 3]                                   # increasing
            z_aim, z_rp = zn + cfg["aim_alpha"] * h, zn + h
            okz = (z_rp < apex - 0.05) & (z_aim > zd[0])
            if not okz.any():
                continue
            ta = np.interp(z_aim, zd, desc[::-1, 0])
            xa = np.interp(z_aim, zd, desc[::-1, 1])
            ya = np.interp(z_aim, zd, desc[::-1, 2])
            xr = np.interp(z_rp, zd, desc[::-1, 1])
            yr = np.interp(z_rp, zd, desc[::-1, 2])
            off = np.hypot(xr - xa, yr - ya)
            mv = {**CATCH_MOVING, **cfg["moving"]} if cfg["moving"] is not None else None
            opts = [(None, xa, ya, off)]
            oc = cfg.get("offcentre")
            if oc is not None and not mv:            # off-centre catch: formation centre moved towards a reference point
                oc = {**CATCH_OFFCENTRE, **oc}
                refs = []
                if "start" in oc["toward"]:
                    refs.append(self.ref_centre(t)[0][:2])
                if "others" in oc["toward"]:
                    z_s = cfg["z_net0"] + cfg["aim_alpha"] * h
                    for b2, tr2 in prev_trajs.items():
                        if b2 != b and self.status[b2] == "flight":
                            c2 = crossing(tr2, z_s)
                            if c2 is not None:
                                refs.append(np.array([c2[1], c2[2]]))
                for R_ in refs:
                    dx_, dy_ = R_[0] - xa, R_[1] - ya
                    dn = np.hypot(dx_, dy_)
                    s_ = np.minimum(oc["r"], dn) / np.maximum(dn, 1e-9)
                    xk, yk = xa + s_ * dx_, ya + s_ * dy_
                    opts.append((None, xk, yk, np.hypot(xr - xk, yr - yk)))
            if mv:                                   # end velocities, net trail, crossing offset relative to the moving net
                vbx = np.interp(z_aim, zd, desc[::-1, 4])
                vby = np.interp(z_aim, zd, desc[::-1, 5])
                vbz = np.interp(z_aim, zd, desc[::-1, 6])
                tr_ = np.interp(z_rp, zd, desc[::-1, 0])
                vz1 = -np.minimum(mv["vz_frac"] * np.abs(vbz), mv["vz_max"])
                c0_ = self.ref_centre(t)[0]
                ap = np.column_stack([xa - c0_[0], ya - c0_[1]])
                apn = ap / np.maximum(np.hypot(ap[:, 0], ap[:, 1]), 1e-9)[:, None]
                opts = []
                for kind in mv["options"]:
                    if kind == "rest":
                        V1 = np.zeros((len(zn), 3))
                    elif kind == "ball":
                        V1 = np.column_stack([mv["v_frac"] * vbx, mv["v_frac"] * vby, vz1])
                    else:
                        sp_ = mv["v_max"] * (0.5 if kind == "approach_half" else 1.0)
                        V1 = np.column_stack([sp_ * apn[:, 0], sp_ * apn[:, 1], vz1])
                    sh = np.hypot(V1[:, 0], V1[:, 1])
                    V1[:, :2] *= np.minimum(1.0, mv["v_max"] / np.maximum(sh, 1e-9))[:, None]
                    sh = np.hypot(V1[:, 0], V1[:, 1])
                    lag = mv["k_lag"] * sh[:, None] * V1[:, :2]      # net centre trails by -lag: put the formation ahead
                    xk, yk = xa + lag[:, 0], ya + lag[:, 1]
                    ok_ = np.hypot(xr - xk - V1[:, 0] * (tr_ - ta), yr - yk - V1[:, 1] * (tr_ - ta))
                    opts.append((V1, xk, yk, ok_))
            r_in = cfg["r_in_m"] or CATCH_GEOM["r_in"] * self.spacing
            clear_up = cfg["clear_up_m"] or CATCH_GEOM["clear_up"] * self.spacing
            for V1, xa, ya, off in opts:
                for i in np.flatnonzero(okz):
                    if ta[i] < t + 0.05 or off[i] > r_in:
                        continue
                    c = np.array([xa[i], ya[i], z_rp[i]])
                    if np.any(c < lo) or np.any(c > hi):
                        continue
                    band = (asc[:, 3] > zn[i] - 0.1) & (asc[:, 3] < z_rp[i] + 0.15)
                    if band.any() and np.hypot(asc[band, 1] - c[0], asc[band, 2] - c[1]).min() < clear_up:
                        continue
                    rows.append((b, zn[i], ta[i], c, None if V1 is None else V1[i]))
            # prediction record: aim point at the standby height and the floor crossing
            g = crossing(tr, self.p.ball_d / 2)
            s_c = crossing(tr, cfg["ref_z_net"] + cfg["aim_alpha"] * self.h_dn0)   # fixed height (convergence record)
            self.pred_hist[b].append((round(t, 3), None if g is None else [float(v) for v in g[:4]],
                                      None if s_c is None else [float(v) for v in s_c[:4]]))
        return rows

    def _unsafe(self, ball, zn, ta, C, th):
        """Candidates where another ball in flight is predicted to pass through the drones (0.25 m around each, rotor
        plane +-0.15 m) or the rim of the net (0.45-1.05 m from the centre, between the net centre height and the
        rotor plane) while the formation waits there (0.5 s before the catch to t_hold + 0.2 s after)."""
        uns = np.zeros(len(ball), bool)
        done = {}
        for j in range(len(ball)):
            key = (int(ball[j]), round(float(zn[j]), 3))
            if self.cfg.get("offcentre") is not None:     # off-centre options of one point differ in the centre
                key = key + (round(float(C[j][0]), 4), round(float(C[j][1]), 4))
            if key in done:                       # velocity options of one point share the check
                uns[j] = uns[done[key]]
                continue
            done[key] = j
            c, z_rp = C[j], C[j][2]
            for b2, tr in self.trajs.items():
                if b2 == ball[j]:
                    continue
                w = (tr[:, 0] >= ta[j] - 0.5) & (tr[:, 0] <= ta[j] + th + 0.2) & (tr[:, 3] > zn[j] - 0.05) \
                    & (tr[:, 3] < z_rp + 0.15)
                if not w.any():
                    continue
                q = tr[w]
                rel = q[:, 1:3] - c[:2]
                rr = np.hypot(rel[:, 0], rel[:, 1])
                ring = (rr > CATCH_GEOM["ring_in"] * self.spacing) & (rr < CATCH_GEOM["ring_out"] * self.spacing) \
                    & (q[:, 3] < z_rp + 0.1)
                near = (np.abs(q[:, 3] - z_rp) < 0.15)[:, None] & (
                    np.hypot(rel[:, None, 0] - self.pattern[None, :, 0], rel[:, None, 1] - self.pattern[None, :, 1])
                    < 0.25)
                if ring.any() or near.any():
                    uns[j] = True
                    break
        return uns

    def replan(self, t):
        cfg = self.cfg
        if cfg["planner_off"] or cfg.get("p2p") is not None:
            return
        pend = [b for b in range(self.nb) if self.status[b] == "flight" and self.ekf[b] is not None
                and self.ekf[b].n >= cfg["n_meas_min"]]
        flying = [b for b in range(self.nb) if self.status[b] == "flight"]
        if t < self.hold_until:
            if pend:
                self._candidates(t, pend)                       # keep the prediction record going
            return
        if not pend:
            if cfg["return_standby"] and not flying:
                self.chosen = None
                self._go(t, self.standby_now(), scale=cfg["standby_accel_factor"])
            return
        rows = self._candidates(t, pend)
        if not rows:
            for b in pend:
                self.plan_hist[b].append((round(t, 3), "no_candidate"))
                self.choice_hist[b].append((round(t, 3), None, 0, 0, None, None))
            return
        mv = {**CATCH_MOVING, **cfg["moving"]} if cfg["moving"] is not None else None
        if mv and self.chosen is not None and self.v1_plan is not None and self.chosen[0] in pend \
                and self.t_arrive_plan - t < mv["freeze_s"]:
            return                                              # moving catch: the last 0.2 s are not replanned
        ball = np.array([r[0] for r in rows])
        zn = np.array([r[1] for r in rows])
        ta = np.array([r[2] for r in rows])
        C = np.array([r[3] for r in rows])
        c0, v0, a0 = self.ref_centre(t)
        lim = self.lim()
        V1 = np.array([r[4] for r in rows]) if mv else None
        Tn, _ = quintic_tmin(c0, v0, a0, C, lim, V1=V1)
        tj = cfg.get("traj")
        if tj is not None and not mv:                       # jerk-limited moves: their durations where they apply
            Ts = scurve_tmin(c0, v0, a0, C, lim, tj)
            Tn = np.where(np.isfinite(Ts), Ts, Tn)
        if self.nd is not None and self.nd["plan"] and not mv:  # net-dynamics moves: their durations
            bfv = cfg.get("boxfix", True)
            bf_ = isinstance(bfv, dict) or bool(bfv)
            bfo_ = {**CATCH_BOXFIX, **(bfv if isinstance(bfv, dict) else {})}
            zfl_ = min(cfg["z_net_range"][0] + self.h_dn(), float(c0[2]),
                       float(self.goal_z) if self.goal_z is not None else np.inf)
            fl_ = zfl_ - bfo_["floor_tol_m"] if (bf_ and bfo_["floor"]) else -np.inf
            _, _, okn, _, _ = self._nd_feasible(t, C, lim, c0, v0, a0, bfo_["tol_m"] if bf_ else 0.05, fl_)
            i_ = nd_first(okn)
            Tn = np.where(i_ >= 0, self.nd_Tg[np.maximum(i_, 0)], Tn)
            self.nd_stats["replan_nd"] += 1
        te, th = cfg["t_early_s"], cfg["t_hold_s"]
        if mv:
            moving_ = np.linalg.norm(V1, axis=1) > 1e-6
            te = np.where(moving_, mv["t_early_s"], te)
        slack0 = ta - t - Tn - te
        N = len(rows)
        unsafe = self._unsafe(ball, zn, ta, C, th)
        if unsafe.all():
            unsafe[:] = False                                   # nowhere safe: ignore the other balls
        # ball order: by the time of the candidate nearest the middle of its height range
        bt = {b: float(np.median(ta[ball == b])) for b in set(ball.tolist())}
        rank = np.array([sorted(bt, key=bt.get).index(b) for b in ball])
        if mv:                                      # moving catches end at rest past the point (coast + brake)
            sp_ = np.linalg.norm(V1, axis=1)
            t_after = np.where(sp_ > 1e-6, mv["coast_s"] + 1.875 * sp_ / (lim["a_h"] * mv["brake_factor"]), 0.0)
            Prest = C + V1 * mv["coast_s"] + 0.5 * V1 * (t_after - mv["coast_s"]).clip(0)[:, None]
            Trr = t_rest_to_rest(Prest, C, lim)
            pen = mv["brake_cost"] * t_after
        else:
            Trr = t_rest_to_rest(C, C, lim) if tj is None else scurve_rest_to_rest(C, C, lim, tj)
            pen = np.zeros(len(rows))
        dist = np.linalg.norm(C[None] - C[:, None], axis=2)
        hold = np.where(dist < 0.1, 0.0, th)
        if mv:                                      # after a moving catch: coast and brake before the next move
            sp = np.linalg.norm(V1, axis=1)
            hold = np.where(moving_[:, None], np.maximum(hold, (mv["coast_s"] + 1.875 * sp
                                                                / (lim["a_h"] * mv["brake_factor"]))[:, None]), hold)
            te_j = te[None, :]
        else:
            te_j = te
        sl = ta[None, :] - ta[:, None] - hold - Trr - (te_j if mv else te)          # (i -> j)
        edge = (sl >= 0) & (rank[:, None] < rank[None, :]) & (ta[:, None] < ta[None, :]) & ~unsafe[None, :]
        dzc = np.abs(C[:, 2][None, :] - C[:, 2][:, None])
        sc, dzp = cfg["slack_cap"], cfg["dz_penalty"]
        gain = np.minimum(sl, sc) - dzp * dzc - pen[None, :]
        start_gain = np.minimum(slack0, sc) - dzp * np.abs(C[:, 2] - c0[2]) - pen
        order = np.argsort(ta)

        def dp(start_ok):
            val = np.full(N, -np.inf)
            prev = np.full(N, -1)
            for j in order:
                if start_ok[j]:
                    val[j], prev[j] = 1000.0 + start_gain[j], -1
                pr = np.flatnonzero(edge[:, j] & np.isfinite(val))
                if len(pr):
                    v_ = val[pr] + 1000.0 + gain[pr, j]
                    k_ = int(np.argmax(v_))
                    if v_[k_] > val[j]:
                        val[j], prev[j] = v_[k_], pr[k_]
            if not np.isfinite(val).any():
                return -np.inf, []
            j = int(np.argmax(val))
            path = [j]
            while prev[path[-1]] >= 0:
                path.append(int(prev[path[-1]]))
            return float(val[j]), path[::-1]

        if mv and mv["late_only"]:                  # moving options only for balls no rest interception reaches
            rest_ok = ~moving_ & (slack0 >= 0) & ~unsafe
            for b_ in set(ball[rest_ok].tolist()):
                unsafe = unsafe | ((ball == b_) & moving_)
            edge = edge & ~unsafe[None, :]
        feas = (slack0 >= 0) & ~unsafe
        best, path = dp(feas)
        if self.chosen is not None and path:
            cb, cz = self.chosen
            keep = feas & (ball == cb) & (np.abs(zn - cz) < 1e-6)
            if keep.any() and ball[path[0]] != cb or (keep.any() and abs(zn[path[0]] - cz) > 1e-6):
                v_k, p_k = dp(keep)
                if p_k and v_k >= best - cfg["hyst"] and int(v_k // 1000) >= int(best // 1000):
                    best, path = v_k, p_k
        status = {}
        if path:
            j = path[0]
            for q_, jj in enumerate(path):
                status[int(ball[jj])] = "target" if q_ == 0 else "planned"
        else:
            j = int(np.argmax(np.where(unsafe, -1e9, 0.0) + slack0 - pen))   # best effort: the least late safe point
            status[int(ball[j])] = "best_effort"
        for b in pend:
            if b not in status:
                status[b] = ("dropped_conflict" if (feas & (ball == b)).any() else
                             "unsafe" if ((slack0 >= 0) & (ball == b)).any() else "infeasible")
            self.plan_hist[b].append((round(t, 3), status[b]))
            m_ = ball == b
            fz = zn[m_ & feas]
            self.choice_hist[b].append((round(t, 3), float(zn[j]) if int(ball[j]) == b else None, int(m_.sum()),
                                        int(len(fz)), float(fz.min()) if len(fz) else None,
                                        float(fz.max()) if len(fz) else None,
                                        float(np.linalg.norm(V1[j][:2])) if (mv and int(ball[j]) == b) else None))
        self.chosen = (int(ball[j]), float(zn[j]))
        if mv and np.linalg.norm(V1[j]) > 1e-6:
            self._go_moving(t, C[j], V1[j], ta[j])
        else:
            self._go(t, C[j])


def solve_launch(L0, P, T, k, v_guess=None):
    """Launch velocity from L0 that puts the ball at P after T seconds (Newton on the drag model, RK4 5 ms)."""
    L0, P = np.asarray(L0, float), np.asarray(P, float)
    v = np.asarray(v_guess, float) if v_guess is not None else (P - L0 + 0.5 * np.array([0, 0, G]) * T ** 2) / T
    n = int(round(T / 0.005))
    dt = T / n

    def f(v_):
        return flight_rk4(np.concatenate([L0, v_]), k, dt=dt, n_max=n, z_stop=-1e9)[-1, 1:4] - P
    for _ in range(30):
        r = f(v)
        if np.linalg.norm(r) < 1e-4:
            return v, True
        J = np.column_stack([(f(v + 1e-4 * e) - r) / 1e-4 for e in np.eye(3)])
        v = v - np.linalg.solve(J, r)
    return v, bool(np.linalg.norm(f(v)) < 1e-3)


def launcher_sim():
    return np.array(CATCH_LAUNCHER_BOX, float) - BOX_OFFSET


def barrel_pos(i, n):
    return launcher_sim() + np.array([0.0, (i - (n - 1) / 2) * CATCH_BARREL_M, 0.0])


def launch_velocity(el, az, v):
    e, a = np.radians(el), np.radians(az)
    return v * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def in_box_flight(p0, v0, k, r=0.02):
    """Whole free flight (to the floor) inside the 10 m box (sim coordinates)."""
    tr = flight_rk4(np.concatenate([p0, v0]), k, dt=0.005, n_max=2000, z_stop=r)
    lo, hi = -BOX_OFFSET + r, np.array([BOX_SIZE] * 3) - BOX_OFFSET - r
    lo[2] = -1.0                                   # the floor ends the flight (last point just below z = r)
    return bool(np.all(tr[:, 1:4] >= lo) and np.all(tr[:, 1:4] <= hi)), tr


def tier_ranges(cfg):
    t = CATCH_TIERS[cfg["tier"]] if cfg.get("tier") else {}
    return {k: tuple(cfg[k]) if cfg.get(k) is not None else tuple(t[k]) for k in ("el_deg", "az_deg", "speed_m_s")}


def prior_velocity(rg):
    """Mean launch velocity over the ranges (the EKF's velocity prior before the first measurements)."""
    el = np.radians(np.linspace(*rg["el_deg"], 7))
    az = np.radians(np.linspace(*rg["az_deg"], 7))
    v = np.linspace(*rg["speed_m_s"], 7)
    E, A, V = np.meshgrid(el, az, v, indexing="ij")
    return np.array([(V * np.cos(E) * np.cos(A)).mean(), (V * np.cos(E) * np.sin(A)).mean(), (V * np.sin(E)).mean()])


def range_scatter(rg, k, z_aim, n=600, seed=0):
    """Monte-Carlo of the launch ranges: crossing points of height z_aim on the descending branch (sim coordinates),
    crossing times, and the share of samples whose flight leaves the box."""
    rng = np.random.default_rng(seed)
    pts, out = [], 0
    for _ in range(n):
        v0 = launch_velocity(rng.uniform(*rg["el_deg"]), rng.uniform(*rg["az_deg"]), rng.uniform(*rg["speed_m_s"]))
        ok, tr = in_box_flight(launcher_sim(), v0, k)
        out += not ok
        c = crossing(tr, z_aim)
        if c is not None:
            pts.append(c[:4])
    return np.array(pts), out / n


def catch_standby(rg, k, cfg, h_dn):
    """Standby formation centre: mean crossing point of the descending branches at the standby aim height."""
    z_xy = cfg.get("standby_xy_z_net") or cfg["z_net0"]
    pts, _ = range_scatter(rg, k, z_xy + cfg["aim_alpha"] * h_dn, n=300, seed=12345)
    c = pts[:, 1:3].mean(axis=0)
    return np.array([c[0], c[1], cfg["z_net0"] + h_dn])


def sample_launches(cfg, rg, k, t0):
    """Random launches of one trial: number of balls, per-ball elevation, azimuth and speed (uniform in the ranges,
    redrawn while the flight would leave the box), launch times (sequence: fixed or uniform random interval)."""
    rng = np.random.default_rng(cfg["seed"])
    if cfg.get("count_draw") is not None:       # consume the count draw of the reference range, then use n_balls
        rng.integers(cfg["count_draw"][0], cfg["count_draw"][1] + 1)
        n = int(cfg["n_balls"][0])
    else:
        n = int(rng.integers(cfg["n_balls"][0], cfg["n_balls"][1] + 1))
    out, t = [], t0
    for i in range(n):
        for _ in range(500):
            el, az, v = rng.uniform(*rg["el_deg"]), rng.uniform(*rg["az_deg"]), rng.uniform(*rg["speed_m_s"])
            p0 = barrel_pos(i, n)
            v0 = launch_velocity(el, az, v)
            if in_box_flight(p0, v0, k)[0]:
                break
        if i > 0 and cfg["launch"] == "sequence":
            t += (rng.uniform(*cfg["interval_range_s"]) if cfg["interval_range_s"] else cfg["interval_s"])
        out.append({"t": round(float(t), 3), "p0": p0, "v0": v0, "el_deg": float(el), "az_deg": float(az),
                    "speed_m_s": float(v), "barrel": i})
    return out


_SRC_HASH = []


def catch_snapshot_key(p, cfg, plan, ff_path):
    """(key, step) of the pre-launch snapshot: a hash of everything the run depends on before the first launch (all
    parameters including the class-level ones, the seed (drone sensor noise), the formation and standby settings,
    the plan's limits and timing, the feedforward table, the number and launch time of the first ball, and this source
    file); the snapshot is taken at the last 10 ms boundary at least 50 ms before the first launch."""
    import hashlib
    if not _SRC_HASH:
        with open(os.path.abspath(__file__), "rb") as f:
            _SRC_HASH.append(hashlib.sha256(f.read()).hexdigest())
    with open(ff_path, "rb") as f:
        ffh = hashlib.sha256(f.read()).hexdigest()
    t1 = min(l["t"] for l in plan.L) if plan.L else cfg["p2p"]["t_move"]
    step = int(np.floor((t1 - 0.05) / p.dt / 10.0)) * 10
    cls = {k: getattr(p, k) for k in ("net_drag", "net_drag_keep_linear", "thread_d", "payload_mass", "payload_d",
                                      "cd_payload", "n_balls", "ball_d", "ball_m", "ball_cd", "rho_air", "cd_thread",
                                      "rho_thread", "att_ki_factor", "string_each")}
    pre = {k: cfg.get(k) for k in ("spacing", "h_dn", "h_dn_table", "h_dn_per_ball_m", "standby_accel_factor",
                                   "return_standby", "replan_hz", "z_net0", "planner_off", "seed", "ref_z_net",
                                   "traj", "move_spacing", "boxfix", "netdyn")}
    blob = json.dumps({"p": asdict(p), "cls": cls, "cfg": pre, "standby": plan.standby.tolist(), "lim": plan.lim0,
                       "t_go0": plan.t_go0, "first_launch": t1, "n_balls": plan.nb, "step": step, "ff": ffh,
                       "src": _SRC_HASH[0]}, sort_keys=True, default=_json_np)
    return hashlib.sha256(blob.encode()).hexdigest()[:24], step


def catch_params(cfg):
    p = travel_params({**TRAVEL_DEFAULT, "p_over": cfg["p_over"], "keep_linear": False})
    p.seed = int(cfg["seed"])
    return p


def run_catch(cfg, out=None, tag=None, verbose=True, save_log=True):
    """One catch trial (cfg: CATCH_DEFAULT keys). Returns the summary dict; with out, writes summary, log and plot."""
    cfg = {**CATCH_DEFAULT, **cfg}
    p = catch_params(cfg)
    h_dn = hold_drop() if cfg["h_dn"] is None else float(cfg["h_dn"])
    if cfg["h_dn_per_ball_m"] is None:
        cfg["h_dn_per_ball_m"] = CATCH_DH_PER_BALL.get(float(cfg["net_side"])) or 0.0
    if cfg.get("h_dn_table") is None:
        cfg["h_dn_table"] = CATCH_DH_TABLE.get(float(cfg["net_side"]))
    k = ball_k(p)
    rg = tier_ranges(cfg) if cfg["launches"] is None or cfg.get("tier") else None
    standby = np.asarray(cfg["standby"], float) if cfg["standby"] is not None else catch_standby(rg, k, cfg, h_dn)
    T_tr = float(t_rest_to_rest(np.array([[0.0, 0.0, p.z_hover]]), standby[None], CATCH_LIMITS_SETS["relaxed"])[0, 0])
    t0 = max(16.0, 13.0 + T_tr + 2.5)
    if cfg.get("p2p") is not None:          # point-to-point move, no balls ...
        nd = int(cfg["p2p"].get("n_drop") or 0)
        launches = []
        if nd:                              # ... or n_drop balls released at rest just above the net centre first
            gap = float(cfg["p2p"].get("drop_gap_s", 0.4))
            xy = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1), (2, 0), (-2, 0), (0, 2)]
            launches = [{"t": round(t0 + gap * i, 3), "p0": standby + np.array([0.06 * xy[i % 12][0], 0.06 * xy[i % 12][1],
                                                                            0.10 - h_dn]),
                         "v0": np.zeros(3)} for i in range(nd)]
            t0 = t0 + gap * nd + float(cfg["p2p"].get("settle_s", 5.0))
        cfg["p2p"] = {"t_move": t0, **cfg["p2p"]}
    elif cfg["launches"] is None:
        launches = sample_launches(cfg, rg, k, t0)
    else:                                   # explicit: dicts with dt (after t0) or t, p0 and v0 (sim coordinates)
        launches = [{**l, "t": round(t0 + l.get("dt", 0.0), 3) if "t" not in l else l["t"],
                     "p0": np.asarray(l["p0"], float), "v0": np.asarray(l["v0"], float)} for l in cfg["launches"]]
    vp = prior_velocity(rg) if rg else None
    for l in launches:
        l["v_prior"] = vp if vp is not None else l.get("v_prior", np.asarray(l["v0"], float))
    p.n_balls = len(launches)
    ff_path = cfg["ff_path"] or os.path.join(HERE, "out_official", "sweep_summary.json")
    ff = load_ff(p, ff_path, spacing=cfg["spacing"])
    plan = CatchPlan(p, cfg, launches, standby, h_dn, ff, cfg["spacing"])
    sim = Sim(p)
    plan.attach(sim)
    snap_kw, snap_info = {}, None
    if cfg.get("snap_dir"):                     # start from a cached pre-launch state when one exists for this key
        key, step = catch_snapshot_key(p, cfg, plan, ff_path)
        path = os.path.join(cfg["snap_dir"], f"{key}.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                snap_kw["resume"] = f.read()
            snap_info = {"key": key, "used": True, "step": step}
        else:
            def save(b, path=path):
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "wb") as f:
                    f.write(b)
                try:
                    os.replace(tmp, path)           # atomic: concurrent workers never read a half-written file
                except OSError:                     # Windows: another worker holds the same snapshot open (it
                    try:                            # wrote the identical file first); keep that one
                        os.remove(tmp)
                    except OSError:
                        pass
            snap_kw["snapshot"] = (step, save)
            snap_info = {"key": key, "used": False, "step": step}
    w0 = time.time()
    log = sim.run(plan, ff=ff, acc_ff=1.0, extra_log=True, verbose=verbose, **snap_kw)
    wall = time.time() - w0
    for key, v in plan.lg.items():
        log[key] = np.array(v)
    met = (catch_analysis(p, log, plan, cfg, launches, standby, h_dn) if cfg.get("p2p") is None
           else p2p_metrics(p, log, plan, cfg, standby))
    met.update(wall_s=round(wall, 1), replan_wall_s=round(plan.replan_cost, 2), n_replan=plan.n_replan)
    if plan.nd is not None:
        met["netdyn"] = {"stats": plan.nd_stats, "moves": plan.nd_hist[:200]}
    if snap_info:
        met["snapshot"] = snap_info
    if out:
        tag = tag or cfg["label"] or f"catch_{cfg['tier']}_{cfg['launch']}_seed{cfg['seed']}"
        met["files"] = [f"{tag}_summary.json"]
        js = {"mode": "catch", **met, "cfg": _catch_cfg_json({k: v for k, v in cfg.items() if k != "snap_dir"}),
              "net_contact": NET_CONTACT,
              "ball": {"d_m": p.ball_d, "m_kg": p.ball_m, "cd": p.ball_cd, "k_1_m": k,
                       "v_terminal_m_s": float(np.sqrt(G / k))},
              "launcher_box_m": list(CATCH_LAUNCHER_BOX), "barrel_m": CATCH_BARREL_M,
              "box": {"size_m": BOX_SIZE, "offset_m": BOX_OFFSET.tolist()},
              "limits": plan.lim0, "aero": aero_info(p), "params": asdict(p)}
        if save_log:
            np.savez_compressed(os.path.join(out, f"{tag}_log.npz"), **log)
            met["files"].append(f"{tag}_log.npz")
            if cfg.get("p2p") is None:
                plot_catch_trial(p, log, met, plan, os.path.join(out, f"{tag}.png"))
                met["files"].append(f"{tag}.png")
            js["files"] = met["files"]
        with open(os.path.join(out, f"{tag}_summary.json"), "w") as f:
            json.dump(js, f, indent=1, default=_json_np)
    return met


P2P_READY = (0.05, 0.10)      # m: "ready to catch" tolerances (formation centre to target, net-centre lag)


def p2p_metrics(p, log, plan, cfg, standby):
    """Point-to-point move (cfg p2p): planned duration, the time from the move start until the formation is ready to
    catch, clearances, motor and tilt over the move.
    Ready (tolerance e = 0.05 m, also 0.10 m): from that moment to the end of the record the formation centre (mean drone
    position) stays within e of the target (3-D) and the net centre stays within e of the formation centre horizontally
    (net lag / swing)."""
    t = log["t"]
    dt = float(np.median(np.diff(t)))
    pos = log["pos"]
    fc = pos.mean(axis=1)
    net = log["net_c"]
    tm = float(plan.p2p_t)
    tgt = np.asarray(plan.target, float)
    T_plan = float(plan.t_rest_plan - tm)
    W = t >= tm - 1e-9
    ef = np.linalg.norm(fc - tgt, axis=1)
    en = np.hypot(net[:, 0] - fc[:, 0], net[:, 1] - fc[:, 1])
    ent = np.hypot(net[:, 0] - tgt[0], net[:, 1] - tgt[1])
    d = tgt - np.asarray(standby, float) if cfg["p2p"].get("d") is None else np.asarray(cfg["p2p"]["d"], float)
    u = d / max(np.linalg.norm(d), 1e-9)
    out = {"p2p": {"d_m": np.round(d, 4).tolist(), "distance_m": round(float(np.linalg.norm(d)), 3),
                   "heading_deg": round(float(np.degrees(np.arctan2(d[1], d[0]))), 1), "t_move_s": tm,
                   "T_plan_s": round(T_plan, 3), "shape": ("netdyn" if cfg.get("netdyn") is not None else
                             "quintic" if cfg.get("traj") is None else "scurve")}}
    iw = np.flatnonzero(W)
    for tol in P2P_READY:
        bad = (ef > tol) | (en > tol)
        bw = np.flatnonzero(bad & W)
        i_r = iw[0] if not len(bw) else (bw[-1] + 1 if bw[-1] + 1 < len(t) else None)
        out["p2p"][f"t_ready_{int(round(tol * 1e3))}mm_s"] = None if i_r is None else round(float(t[i_r] - tm), 3)
    out["p2p"]["net_lag_max_mm"] = round(float(1e3 * en[W].max()), 1)
    Wa = t >= tm + T_plan - 1e-9                # after the planned end of the move: residual swing of the net
    out["p2p"]["net_swing_after_mm"] = round(float(1e3 * en[Wa].max()), 1) if Wa.any() else None
    out["p2p"]["net_to_target_end_mm"] = round(float(1e3 * ent[-1]), 1)
    out["p2p"]["overshoot_mm"] = round(float(1e3 * max(0.0, ((fc[W] - tgt) @ u).max())), 1)
    # reference acceleration along the move (to tell accelerating / braking phases apart)
    ts = t[W]
    ra = np.array([plan._seg_eval(x)[2] @ u for x in ts])
    rv = np.array([plan._seg_eval(x)[1] for x in ts])
    out["p2p"]["ref_acc_max_m_s2"] = round(float(np.abs(ra).max()), 3)
    out["p2p"]["ref_speed_max_m_s"] = round(float(np.linalg.norm(rv, axis=1).max()), 3)
    C = TRANSFER_CRIT
    tend = float(t[-1])
    phases = [("on_ground", 0.0, 0.5), ("takeoff", 0.5, 6.5), ("spread", 6.5, 12.5), ("cruise", 12.5, tend)]
    cl, cls = clearance_analysis(p, log, phases)
    ft = cls["t"]
    fW = ft >= tm - 1e-9
    fB = (ft >= tm - 2.0) & (ft < tm)
    allc, gc = cls["all"], cls["guard"]

    def where_min(a, mask):
        a_ = np.where(mask[:, None], a, np.inf)
        i, k = np.unravel_index(np.nanargmin(np.where(np.isfinite(a_), a_, np.inf)), a_.shape)
        tt = float(ft[i])
        acc = float(plan._seg_eval(tt)[2] @ u)
        ph = ("after_stop" if tt > tm + T_plan else "accelerating" if acc > 0.2 else "braking" if acc < -0.2
              else "steady")
        return {"mm": round(float(1e3 * a[i, k]), 1), "drone": int(k), "t_after_start_s": round(tt - tm, 3),
                "phase": ph, "ref_acc_along_m_s2": round(acc, 2)}
    out["p2p"]["disc_min"] = where_min(allc, fW)
    out["p2p"]["guard_min"] = where_min(gc, fW)
    out["p2p"]["disc_min_hover_before_mm"] = round(float(1e3 * allc[fB].min()), 1)
    out["p2p"]["guard_min_hover_before_mm"] = round(float(1e3 * np.nanmin(gc[fB])), 1)
    out["p2p"]["disc_min_per_drone_mm"] = [round(float(1e3 * allc[fW, k].min()), 1) for k in range(4)]
    mf = log["motor"] / p.f_max
    out["p2p"]["motor_max"] = round(float(mf[W].max()), 4)
    hi90 = W & (mf > C["motor_frac"]).any(axis=(1, 2))
    a_, b_ = _runs(hi90)
    out["p2p"]["motor90_longest_s"] = round(float((b_ - a_).max() * dt) if len(a_) else 0.0, 3)
    out["p2p"]["tilt_max_deg"] = round(float(log["tilt"][W].max()), 2)
    out["p2p"]["alt_err_max_mm"] = round(float(1e3 * np.abs(pos[W, :, 2] - log["ref"][W, :, 2]).max()), 1)
    out["p2p"]["at_min_thrust_s"] = round(float((W[:, None, None] & (log["motor_raw"] <= getattr(p, "f_min", 0.0)))
                                                .any(axis=(1, 2)).sum() * dt), 3)
    ok = {"disc30": out["p2p"]["disc_min"]["mm"] >= 1e3 * C["disc_m"],
          "guard": out["p2p"]["guard_min"]["mm"] >= 1e3 * C["guard_m"],
          "motor90": out["p2p"]["motor90_longest_s"] <= C["motor_time_s"] + 1e-9,
          "control": plan.abort is None}
    out["p2p"]["ok"] = ok
    out["p2p"]["clean"] = all(ok.values())
    out["aborted"] = plan.abort is not None
    out["safety_violations"] = [k for k, v in ok.items() if not v]
    return out


def _catch_cfg_json(cfg):
    c = dict(cfg)
    if c.get("launches") is not None:
        c["launches"] = [{k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in l.items()}
                         for l in c["launches"]]
    return c


def catch_pile_support(log, p, levels=6):
    """(samples, balls) bool: the ball is in contact with the net, or touches (centre distance <= d + 3 mm) a ball that
    is supported, up to `levels` balls deep (pile_catch)."""
    cont = log["ball_contact"].astype(bool)
    bp = log["ball_pos"]
    nb = cont.shape[1]
    dmax = p.ball_d + 0.003
    with np.errstate(invalid="ignore"):
        near = {(b, c): np.nan_to_num(np.linalg.norm(bp[:, b] - bp[:, c], axis=1), nan=np.inf) <= dmax
                for b in range(nb) for c in range(nb) if b != c}
    sup = cont.copy()
    for _ in range(levels):
        new = sup.copy()
        for (b, c), nr in near.items():
            new[:, b] |= sup[:, c] & nr
        if (new == sup).all():
            break
        sup = new
    return sup


def catch_ball_losses(log, balls, held, dt, fcv):
    """Balls that touched the net but are not caught at the end: "bounced_out" (never held, i.e. never >= 90 % in the
    net over 0.5 s), "knocked_out" (held, and another ball first touched the net within 0.6 s before it left) or
    "thrown_out" (held, no arrival then: thrown out by the formation's motion). Adds r["loss"] (analysis only)."""
    t = log["t"]
    n5 = max(1, int(round(0.5 / dt)))
    first = {r["ball"]: (float(t[np.flatnonzero(held[:, r["ball"]])[0]]) if held[:, r["ball"]].any() else None)
             for r in balls if r["outcome"] != "not_launched"}
    acc = np.gradient(fcv, t, axis=0)
    for r in balls:
        b = r["ball"]
        if r["outcome"] in ("caught", "not_launched") or not held[:, b].any():
            continue
        hb = held[:, b].astype(float)
        frac = np.convolve(hb, np.ones(n5) / n5, mode="valid")
        held_once = bool(frac.max() >= 0.9)
        i_out = int(np.flatnonzero(held[:, b])[-1])
        t_out = float(t[i_out])
        w = slice(max(0, i_out - int(round(0.1 / dt))), i_out + int(round(0.1 / dt)) + 1)
        q = {"t_out_s": round(t_out, 3), "held": held_once,
             "held_s": round(float(hb.sum() * dt), 2),
             "formation_v_h_m_s": round(float(np.hypot(fcv[i_out, 0], fcv[i_out, 1])), 3),
             "formation_a_h_max_m_s2": round(float(np.hypot(acc[w, 0], acc[w, 1]).max()), 3),
             "balls_held_at_out": int(sum(held[i_out, c] for c in range(held.shape[1]) if c != b))}
        hits = [(c, first[c]) for c in first if c != b and first[c] is not None and t_out - 0.6 <= first[c] <= t_out + 0.05]
        if not held_once:
            q["kind"] = "bounced_out"
        elif hits:
            q["kind"] = "knocked_out"
            q["by_ball"] = int(hits[0][0]) + 1
            q["arrival_before_out_s"] = round(t_out - hits[0][1], 3)
        else:
            q["kind"] = "thrown_out"
        r["loss"] = q


def catch_reference_check(log, plan):
    """The planner's formation-centre reference over the catch phase: largest distance outside the formation box and
    largest depth below the height floor in force when the current plan was made (boxfix), and the braking fallbacks."""
    t = log["t"]
    W = t >= plan.t_go0
    rc = log["ref_c"]
    lo, hi = plan.box
    ex = np.maximum(np.maximum(lo - rc, rc - hi), 0.0).max(axis=1)
    out = {"out_of_box_max_m": round(float(ex[W].max()), 4) if W.any() else 0.0,
           "out_of_box_t_s": round(float(t[W][np.argmax(ex[W])]), 3) if W.any() and ex[W].max() >= 5e-5 else None,
           "n_brake": int(plan.n_brake), "n_brake_latched": int(plan.n_brake_latched),
           "n_brake_kept_plan": int(plan.n_brake_kept), "n_floor_kept_plan": int(plan.n_floor_kept)}
    if plan.zlo_hist:
        zt = np.array([x[0] for x in plan.zlo_hist])
        zf = np.array([x[1] for x in plan.zlo_hist])
        i = np.searchsorted(zt, t, side="right") - 1
        fl = np.where(i >= 0, zf[np.clip(i, 0, None)], -np.inf)
        dp = np.maximum(fl - rc[:, 2], 0.0)
        out["below_floor_max_m"] = round(float(dp[W].max()), 4) if W.any() else 0.0
        out["below_floor_t_s"] = round(float(t[W][np.argmax(dp[W])]), 3) if W.any() and dp[W].max() >= 5e-5 else None
    else:
        out["below_floor_max_m"], out["below_floor_t_s"] = 0.0, None
    return out


def catch_analysis(p, log, plan, cfg, launches, standby, h_dn):
    """Per-ball outcome and reason, prediction convergence, net state at each arrival, safety over the catch phase."""
    t = log["t"]
    dt = float(np.median(np.diff(t)))
    nb = len(launches)
    bp, bv = log["ball_pos"], log["ball_vel"]
    cont = log["ball_contact"]
    pos = log["pos"]
    fc = pos.mean(axis=1)
    fcv = log["vel"].mean(axis=1)
    k = ball_k(p)
    tend = float(t[-1])
    n1 = int(round(1.0 / dt))
    sup = catch_pile_support(log, p) if cfg.get("pile_catch") else None
    balls = []
    # balls in contact with the net at each sample (for "balls already in the net")
    for b, l in enumerate(launches):
        r = {"ball": b, "t_launch": l["t"], "el_deg": round(l["el_deg"], 2) if "el_deg" in l else None,
             "az_deg": round(l["az_deg"], 2) if "az_deg" in l else None,
             "speed_m_s": round(l["speed_m_s"], 3) if "speed_m_s" in l else None,
             "p0_box": (np.asarray(l["p0"]) + BOX_OFFSET).round(3).tolist(), "v0": np.round(l["v0"], 4).tolist()}
        lw = t >= l["t"] - 1e-9
        if not lw.any() or not np.isfinite(bp[lw, b, 0]).any():
            r["outcome"] = "not_launched"
            balls.append(r)
            continue
        # truth: free flight from the launch state (no net), for the prediction errors
        tr = flight_rk4(np.concatenate([l["p0"], l["v0"]]), k, dt=0.001, n_max=10000, z_stop=p.ball_d / 2)
        tr[:, 0] += l["t"]
        g_true = tr[-1]
        r["free_flight"] = {"apex_m": round(float(tr[:, 3].max()), 3), "t_floor_s": round(float(g_true[0] - l["t"]), 3),
                            "floor_box": (g_true[1:4] + BOX_OFFSET).round(3).tolist()}
        sb = crossing(tr, cfg["ref_z_net"] + cfg["aim_alpha"] * h_dn)
        if sb is not None:
            r["standby_offset_m"] = round(float(np.hypot(sb[1] - standby[0], sb[2] - standby[1])), 3)
            r["t_standby_cross_s"] = round(float(sb[0] - l["t"]), 3)
        # prediction convergence: predicted floor point and standby-height crossing vs the free flight
        s_true = sb
        conv = []
        for (tp, g, s_) in plan.pred_hist[b]:
            e_g = None if g is None else float(np.hypot(g[1] - g_true[1], g[2] - g_true[2]))
            e_s = None if (s_ is None or s_true is None) else float(np.hypot(s_[1] - s_true[1], s_[2] - s_true[2]))
            et = None if (s_ is None or s_true is None) else float(s_[0] - s_true[0])
            conv.append([round(tp - l["t"], 3), None if e_g is None else round(e_g, 4),
                         None if e_s is None else round(e_s, 4), None if et is None else round(et, 4)])
        r["prediction"] = conv
        r["pred_hist"] = plan.pred_hist[b]        # (t, predicted floor point [t, x, y, z], standby-height crossing)
        # contacts and final state
        cb = cont[:, b] & lw
        r["net_contact_s"] = round(float(cb.sum() * dt), 2)
        r["t_first_contact"] = round(float(t[np.flatnonzero(cb)[0]]), 3) if cb.any() else None
        last = slice(max(0, len(t) - n1), len(t))
        vr = log["ball_vrel"][last, b]
        r["end_rel_speed_m_s"] = round(float(np.nanmean(vr)), 4) if np.isfinite(vr).any() else None
        caught = bool(cont[last, b].mean() >= 0.95 and np.isfinite(vr).any() and np.nanmean(vr) < CATCH_REL_V
                      and tend - (r["t_first_contact"] or tend) >= 1.0)
        if sup is not None and not caught:      # resting on balls in the net (pile_catch)
            sb_ = sup[:, b] & lw
            if sb_.any():
                vnc = np.gradient(log["net_c"][last], t[last], axis=0)
                vrel_p = float(np.linalg.norm(bv[last, b] - vnc, axis=1).mean())
                t_sup = float(t[np.flatnonzero(sb_)[0]])
                if sb_[last].mean() >= 0.95 and vrel_p < CATCH_REL_V and tend - t_sup >= 1.0:
                    caught = True
                    r["caught_on_pile"] = {"t_first_support": round(t_sup, 3), "rel_speed_m_s": round(vrel_p, 4)}
        zend = float(bp[-1, b, 2])
        r["end_box"] = (bp[-1, b] + BOX_OFFSET).round(3).tolist()
        r["shell_hits"] = len(plan.shell_hits[b])
        r["shell_hit_first"] = (None if not plan.shell_hits[b] else
                                {"t_s": round(plan.shell_hits[b][0][0], 3), "drone": int(plan.shell_hits[b][0][1])})
        r["gap_disc_mm"] = round(1e3 * plan.gap_disc[b], 1) if np.isfinite(plan.gap_disc[b]) else None
        r["gap_shell_mm"] = round(1e3 * plan.gap_shell[b], 1) if np.isfinite(plan.gap_shell[b]) else None
        r["gap_string_mm"] = round(1e3 * plan.gap_string[b], 1) if np.isfinite(plan.gap_string[b]) else None
        r["fn_max_N"] = round(float(np.nanmax(log["ball_fn"][:, b])), 4)
        # crossing of the rotor plane (descending, true path) and the formation then
        zb = bp[:, b, 2]
        ia = int(np.nanargmax(np.where(lw, zb, -np.inf)))
        cr = np.flatnonzero((np.arange(len(t)) > ia) & lw & (zb < fc[:, 2]))
        r["t_cross_rotor"] = None
        hs = plan.plan_hist[b]
        if len(cr):
            i = cr[0]
            r["t_cross_rotor"] = round(float(t[i]), 3)
            r["cross_offset_m"] = round(float(np.hypot(bp[i, b, 0] - fc[i, 0], bp[i, b, 1] - fc[i, 1])), 3)
            tgt = log["target"][i]
            r["formation_to_target_m"] = round(float(np.linalg.norm(fc[i] - tgt)), 3) if np.isfinite(tgt).all() else None
            r["ref_arrived"] = bool(log["t_arrive_plan"][i] <= t[i] + 0.02)
            r["chosen_at_cross"] = int(log["chosen"][i])
            others = [bb for bb in range(nb) if bb != b and cont[max(0, i - 20):i + 1, bb].any()]
            r["balls_in_net_at_cross"] = len(others)
            w = slice(max(0, i - n1), i + 1)
            relc = log["net_c"][w] - fc[w]
            r["formation_at_cross"] = {"z_drones_m": round(float(fc[i, 2]), 3),
                                       "net_centre_z_m": round(float(log["net_c"][i, 2]), 3),
                                       "v_h_m_s": round(float(np.hypot(fcv[i, 0], fcv[i, 1])), 3),
                                       "v_z_m_s": round(float(fcv[i, 2]), 3)}
            r["net_at_cross"] = {"sag_m": round(float(log["sag_net"][i]), 4),
                                 "swing_mm": round(float(1e3 * np.linalg.norm(relc - relc.mean(0), axis=1).max()), 1)}
            st_before = [s for (ts, s) in hs if ts <= t[i]]
        else:
            st_before = [s for (_, s) in hs]
        r["plan_status_last"] = st_before[-1] if st_before else None
        r["plan_status_counts"] = {s: sum(1 for _, x in hs if x == s) for s in sorted(set(x for _, x in hs))}
        r["ever_dropped_conflict"] = any(x == "dropped_conflict" for _, x in hs)
        chh = [x for x in plan.choice_hist[b] if r["t_cross_rotor"] is None or x[0] <= r["t_cross_rotor"]]
        tg = [x for x in chh if x[1] is not None]
        x_ = tg[-1] if tg else (chh[-1] if chh else None)
        f_ = tg[0] if tg else None
        r["intercept_first_target"] = (None if f_ is None else
                                       {"t_since_launch_s": round(f_[0] - l["t"], 3), "zn_m": round(f_[1], 3),
                                        "n_candidates": f_[2], "n_feasible": f_[3],
                                        "lowest_feasible": None if f_[4] is None else bool(abs(f_[1] - f_[4]) < 1e-6)})
        r["intercept_max_feasible"] = max((x[3] for x in chh), default=0)
        r["intercept_max_candidates"] = max((x[2] for x in chh), default=0)
        r["intercept_replans_with_feasible_share"] = (round(float(np.mean([x[3] > 0 for x in chh])), 3) if chh else None)
        r["intercept"] = {"zn_target_m": None if not tg else round(tg[-1][1], 3),
                          "v_end_target_m_s": None if not tg or len(tg[-1]) < 7 or tg[-1][6] is None
                          else round(tg[-1][6], 3),
                          "n_candidates": None if x_ is None else x_[2], "n_feasible": None if x_ is None else x_[3],
                          "zn_feasible_min_m": None if x_ is None or x_[4] is None else round(x_[4], 3),
                          "zn_feasible_max_m": None if x_ is None or x_[5] is None else round(x_[5], 3),
                          "target_is_lowest_feasible": (None if not tg or tg[-1][4] is None
                                                        else bool(abs(tg[-1][1] - tg[-1][4]) < 1e-6))}
        if cb.any():
            i1 = np.flatnonzero(cb)[0]
            r["intercept"]["net_centre_z_at_contact_m"] = round(float(log["net_c"][i1, 2]), 3)
            r["intercept"]["v_h_at_contact_m_s"] = round(float(np.hypot(fcv[i1, 0], fcv[i1, 1])), 3)
            r["intercept"]["v_z_at_contact_m_s"] = round(float(fcv[i1, 2]), 3)
        r["ever_target"] = any(x in ("target", "best_effort") for _, x in hs)
        r["planner_view"] = plan.status[b]
        if caught:
            r["outcome"], r["reason"] = "caught", None
        else:
            r["outcome"] = "floor" if zend < 0.05 else "elsewhere"
            first_net = r["t_first_contact"]
            hit = r["shell_hit_first"]
            if hit is not None and (first_net is None or hit["t_s"] < first_net):
                r["reason"] = "hit_drone"
            elif r["net_contact_s"] >= 0.03 and plan.abort is not None and cont[-1, b]:
                r["reason"] = "trial_aborted"
            elif r["net_contact_s"] >= 0.03:
                r["reason"] = "fell_out"
            elif r["ever_dropped_conflict"] and (not len(cr) or r["chosen_at_cross"] != b):
                r["reason"] = "crowded_out"
            elif (r["plan_status_last"] in (None, "infeasible", "best_effort", "no_candidate") or not len(cr)
                  or not r["ref_arrived"] or (r.get("formation_to_target_m") or 0) > 0.10
                  or r["chosen_at_cross"] != b):
                r["reason"] = "late"
            else:
                r["reason"] = "offset"
        balls.append(r)
    catch_ball_losses(log, balls, cont if sup is None else sup, dt, fcv)
    # safety over the catch phase (from the move to the standby point to the end)
    W = t >= plan.t_go0
    C = TRANSFER_CRIT
    mf = log["motor"] / p.f_max
    hi90 = W & (mf > C["motor_frac"]).any(axis=(1, 2))
    a_, b_ = _runs(hi90)
    lgst = float((b_ - a_).max() * dt) if len(a_) else 0.0
    # time above each motor line (any motor; total and longest continuous run) and command saturation (analysis only)
    mlines = {}
    for fr in CATCH_MOTOR_LINES:
        hk = W & (mf > fr).any(axis=(1, 2))
        ak, bk = _runs(hk)
        mlines[f"{fr:.2f}"] = {"total_s": round(float(hk.sum() * dt), 3),
                               "longest_s": round(float((bk - ak).max() * dt) if len(ak) else 0.0, 3)}
    sat = W & ((log["motor_raw"] / p.f_max) >= 1.0).any(axis=(1, 2))
    sa_, sb_ = _runs(sat)
    phases = [("on_ground", 0.0, 0.5), ("takeoff", 0.5, 6.5), ("spread", 6.5, 12.5), ("cruise", 12.5, tend)]
    cl, cls = clearance_analysis(p, log, phases)
    fW = cls["t"] >= plan.t_go0
    dmin, gmin = cls["all"].min(1), np.nanmin(cls["guard"], axis=1)
    sep = np.min([np.linalg.norm(pos[:, i] - pos[:, j], axis=1) for i in range(4) for j in range(i + 1, 4)], axis=0)
    slack = W[:, None] & (log["tension"] <= 0.0)
    lo, hi_ = -BOX_OFFSET, np.array([BOX_SIZE] * 3) - BOX_OFFSET
    db = np.minimum.reduce([(pos[:, :, 0] - DRONE_R - lo[0]).min(1), (hi_[0] - pos[:, :, 0] - DRONE_R).min(1),
                            (pos[:, :, 1] - DRONE_R - lo[1]).min(1), (hi_[1] - pos[:, :, 1] - DRONE_R).min(1),
                            (hi_[2] - pos[:, :, 2] - DRONE_HALF_H).min(1)])
    gd = [b["gap_disc_mm"] for b in balls if b.get("gap_disc_mm") is not None]
    gs = [b["gap_shell_mm"] for b in balls if b.get("gap_shell_mm") is not None]
    saf = {"control": {"ok": plan.abort is None, "abort": plan.abort,
                       "tilt_max_deg": round(float(log["tilt"][W].max()), 2),
                       "alt_err_max_mm": round(float(1e3 * np.abs(pos[W, :, 2] - log["ref"][W, :, 2]).max()), 1),
                       "floor_contacts_other": int(log["ncon_other"][W].max())},
           "motor90": {"ok": lgst <= C["motor_time_s"] + 1e-9, "longest_s": round(lgst, 3),
                       "motor_max": round(float(mf[W].max()), 4),
                       "lines": mlines, "saturated_total_s": round(float(sat.sum() * dt), 3),
                       "saturated_longest_s": round(float((sb_ - sa_).max() * dt) if len(sa_) else 0.0, 3),
                       "command_min_N": round(float(log["motor_raw"][W].min()), 4),
                       "at_min_thrust_s": round(float((W[:, None, None] & (log["motor_raw"] <= getattr(p, "f_min", 0.0)))
                                                      .any(axis=(1, 2)).sum() * dt), 3)},
           "clearance": {"ok": bool(dmin[fW].min() >= C["disc_m"] and gmin[fW].min() >= C["guard_m"]),
                         "disc_min_mm": round(float(1e3 * dmin[fW].min()), 1),
                         "guard_min_mm": round(float(1e3 * gmin[fW].min()), 1)},
           "ball_rotor": {"ok": all(b.get("shell_hits", 0) == 0 for b in balls) and (not gd or min(gd) >= C["disc_m"] * 1e3),
                          "disc_gap_min_mm": min(gd) if gd else None, "shell_gap_min_mm": min(gs) if gs else None,
                          "shell_hits": int(sum(b.get("shell_hits", 0) > 0 for b in balls))},
           "net_low": {"ok": bool(log["net_zmin"][W].min() > C["net_low_m"]),
                       "net_lowest_m": round(float(log["net_zmin"][W].min()), 4)},
           "separation": {"ok": bool(sep[W].min() >= C["separation_m"]), "min_m": round(float(sep[W].min()), 4)},
           "slack": {"ok": float(slack.sum(0).max() * dt) <= C["slack_s"] + 1e-9,
                     "slack_max_s": round(float(slack.sum(0).max() * dt), 3)},
           "box": {"ok": bool(db[W].min() >= 0.0), "min_dist_m": round(float(db[W].min()), 4)}}
    viol = [key for key, v in saf.items() if not v["ok"]]
    ref = catch_reference_check(log, plan)
    nl = sum(b["outcome"] != "not_launched" for b in balls)
    nc = sum(b["outcome"] == "caught" for b in balls)
    return {"n_balls": nb, "n_caught": nc, "all_caught": nc == nl, "caught_share": round(nc / max(nl, 1), 4),
            "safety_violations": viol, "clean": not viol, "all_caught_clean": nc == nl and not viol,
            "aborted": plan.abort is not None, "t_end_s": round(tend, 2), "standby_box": (standby + BOX_OFFSET).round(3).tolist(),
            "h_dn_m": h_dn, "h_dn_per_ball_m": plan.dh, "t_launch0_s": min(l["t"] for l in launches),
            "safety": saf, "balls": balls, "events": plan.events, "clearance_phases": cl["phases"][-1], "reference": ref}


def plot_catch_trial(p, log, met, plan, path):
    """Top and side view (box coordinates) of the ball paths, predicted standby-height crossings and the formation
    path, and the time series of distance, motor and tilt."""
    off = BOX_OFFSET
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    t = log["t"]
    fc = log["pos"].mean(axis=1) + off
    bp = log["ball_pos"] + off
    cols = plt.cm.tab10(np.arange(10))
    a0, a1 = ax[0, 0], ax[0, 1]
    a0.plot(fc[:, 0], fc[:, 1], "k-", lw=1.5, label="formation centre")
    a1.plot(fc[:, 0], fc[:, 2], "k-", lw=1.5, label="formation centre (drones)")
    a1.plot(log["net_c"][:, 0] + off[0], log["net_c"][:, 2] + off[2], "k:", lw=1, label="net centre")
    L0 = np.array(CATCH_LAUNCHER_BOX)
    for a, j in ((a0, 1), (a1, 2)):
        a.plot(L0[0], L0[j], "ks", ms=10, label="launcher")
    for b, r in enumerate(met["balls"]):
        if r["outcome"] == "not_launched":
            continue
        c = cols[b % 10]
        lab = f"ball {b + 1}: {r['outcome']}" + (f" ({r['reason']})" if r.get("reason") else "")
        a0.plot(bp[:, b, 0], bp[:, b, 1], "-", color=c, lw=1.2, label=lab)
        a1.plot(bp[:, b, 0], bp[:, b, 2], "-", color=c, lw=1.2)
        ph = [s for (_, g, s) in plan.pred_hist[b] if s is not None]
        if ph:
            ph = np.array(ph)
            a0.plot(ph[:, 1] + off[0], ph[:, 2] + off[1], ".", color=c, ms=3, alpha=0.6)
        i = np.flatnonzero(np.isfinite(bp[:, b, 0]))
        if len(i):
            a0.plot(bp[i[-1], b, 0], bp[i[-1], b, 1], "o", color=c, ms=6, mfc="none")
    a0.set_title("top view (box coordinates); dots: predicted crossing of the standby aim height")
    a0.set_xlabel("x [m]")
    a0.set_ylabel("y [m]")
    a0.set_aspect("equal")
    a0.set_xlim(0, 10)
    a0.set_ylim(0, 10)
    a0.legend(fontsize=7, loc="upper left")
    a1.set_title("side view x-z")
    a1.set_xlabel("x [m]")
    a1.set_ylabel("z [m]")
    a1.set_xlim(0, 10)
    a1.set_ylim(0, 10)
    a1.legend(fontsize=7)
    a2 = ax[1, 0]
    W = t >= plan.t_go0 - 1
    for b, r in enumerate(met["balls"]):
        if r["outcome"] == "not_launched":
            continue
        dh = np.hypot(bp[:, b, 0] - fc[:, 0], bp[:, b, 1] - fc[:, 1])
        a2.plot(t[W], np.where(log["ball_contact"][W, b], np.nan, dh[W]), "-", color=cols[b % 10], lw=1,
                label=f"ball {b + 1} horizontal distance to formation centre")
        a2.plot(t[W], np.where(log["ball_contact"][W, b], dh[W], np.nan), "-", color=cols[b % 10], lw=4, alpha=0.5)
    v = np.linalg.norm(log["vel"].mean(axis=1), axis=1)
    a2.plot(t[W], v[W], "k-", lw=1, label="formation speed [m/s]")
    a2.set_ylim(0, 4)
    a2.set_xlabel("t [s]")
    a2.set_title("thick: ball in contact with the net")
    a2.legend(fontsize=7)
    a3 = ax[1, 1]
    a3.plot(t[W], (log["motor"][W] / p.f_max).max(axis=(1, 2)), "r-", lw=1, label="motor max (share of f_max)")
    a3.plot(t[W], log["tilt"][W].max(axis=1) / 60.0, "b-", lw=1, label="tilt max / 60 deg")
    a3.plot(t[W], log["sag_net"][W], "g-", lw=1, label="net sag [m]")
    a3.axhline(0.9, color="r", ls=":", lw=0.8)
    a3.set_xlabel("t [s]")
    a3.legend(fontsize=7)
    s = met["safety"]
    fig.suptitle(f"caught {met['n_caught']} / {met['n_balls']}; safety: "
                 + (", ".join(met["safety_violations"]) or "none exceeded")
                 + f"; motor max {s['motor90']['motor_max']:.2f}, disc {s['clearance']['disc_min_mm']} mm, "
                   f"tilt {s['control']['tilt_max_deg']} deg")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def _catch_job(cfg):
    try:
        out, tag = cfg.pop("_out", None), cfg.pop("_tag", None)
        m = run_catch(cfg, out=out, tag=tag, verbose=False, save_log=cfg.pop("_save_log", False) if out else False)
        m["cfg_key"] = {k: cfg.get(k) for k in ("tier", "launch", "interval_s", "noise_m", "seed", "label")}
        return m
    except Exception as exc:                      # keep the batch running; the error is reported
        import traceback
        return {"error": repr(exc), "trace": traceback.format_exc(), "cfg": _catch_cfg_json(cfg)}


# ---------------------------------------------------------------- catch mode: studies
def _pool(jobs):
    import multiprocessing as mp
    return mp.get_context("spawn").Pool(jobs)


def catch_drop_test(out, base=None, verbose=True):
    """Minimal net-contact check, formation hovering (planner off, drones 1.2 m, net centre ~0.8 m):
    (a) one ball released at rest 1 m above the net centre; (b) five balls released one by one every 3 s (a few cm
    apart), sag per ball for the planner's height table; (c) one ball arriving like a catch, 6.5 m/s, 70 deg below
    horizontal. Rebound, drone altitude disturbance, net-centre sink, tension and motor peaks."""
    base = {**CATCH_DEFAULT, **(base or {})}
    os.makedirs(out, exist_ok=True)
    h_dn = hold_drop() if base["h_dn"] is None else float(base["h_dn"])
    p0 = catch_params(base)
    zc = p0.z_hover - h_dn                                          # net centre in the hold
    common = dict(tier=None, planner_off=True, standby=[0.0, 0.0, p0.z_hover], z_net0=p0.z_hover - h_dn,
                  h_dn=h_dn, h_dn_per_ball_m=0.0, h_dn_table=[0.0], return_standby=False)
    e = np.radians(70.0)
    v_imp = 6.5 * np.array([np.cos(e), 0.0, -np.sin(e)])
    tests = {"a_drop_1m": [{"dt": 0.0, "p0": [0.0, 0.0, zc + 1.0], "v0": [0.0, 0.0, 0.0]}],
             "b_five_drops": [{"dt": 3.0 * i, "p0": [0.03 * np.cos(2.4 * i), 0.03 * np.sin(2.4 * i), zc + 1.0],
                               "v0": [0.0, 0.0, 0.0]} for i in range(5)],
             "c_fast_oblique": [{"dt": 0.0, "p0": (np.array([0.0, 0.0, zc + 0.05]) - v_imp / 6.5 * 0.6).tolist(),
                                 "v0": v_imp.tolist()}]}
    res = {}
    for name, launches in tests.items():
        cfg = {**base, **common, "launches": [{**l, "v_prior": l["v0"]} for l in launches],
               "t_after_last_max_s": 6.0, "label": f"drop_{name}", "seed": 1}
        m = run_catch(cfg, out=out, tag=f"drop_{name}", verbose=False)
        L = dict(np.load(os.path.join(out, f"drop_{name}_log.npz")))
        res[name] = drop_metrics(L, m, cfg, launches)
        if verbose:
            print(name, json.dumps(res[name])[:900], flush=True)
    h = res["b_five_drops"]["h_dn_after_each_m"]
    res["h_dn_table_m"] = [round(x - h[0], 4) for x in h]      # drone-to-net-centre drop vs balls in the net
    with open(os.path.join(out, "drop_test.json"), "w") as f:
        json.dump(res, f, indent=1, default=_json_np)
    plot_drop_test(out, res)
    return res


def drop_metrics(L, m, cfg, launches):
    t = L["t"]
    dt = float(np.median(np.diff(t)))
    pos, ref = L["pos"], L["ref"]
    t0 = min(b["t_launch"] for b in m["balls"])
    pre = (t >= t0 - 1.0) & (t < t0)
    zc0 = float(L["net_c"][pre, 2].mean())
    sag0 = float(L["sag_net"][pre].mean())
    dz_pre = float(np.abs(pos[pre, :, 2] - ref[pre, :, 2]).max())
    out = {"net_centre_before_m": round(zc0, 4), "sag_before_m": round(sag0, 4),
           "drone_alt_err_before_max_mm": round(1e3 * dz_pre, 1),
           "tension_before_gf": round(float(L["tension"][pre].mean() / G * 1e3), 2)}
    balls = []
    for b, r in enumerate(m["balls"]):
        tl = r["t_launch"]
        cb = L["ball_contact"][:, b] & (t >= tl)
        bz = L["ball_pos"][:, b, 2]
        q = {"outcome": r["outcome"], "t_first_contact": r["t_first_contact"]}
        if cb.any():
            i1 = np.flatnonzero(cb)[0]
            after = np.arange(len(t)) >= i1
            sep = after & ~cb
            q["time_out_of_contact_after_first_s"] = round(float(sep.sum() * dt), 3)
            # rebound: highest point of the ball while out of contact after the first contact, above its height at
            # the first contact (0 when it never leaves the net)
            q["rebound_mm"] = round(float(1e3 * max(0.0, (bz[sep].max() - bz[i1]) if sep.any() else 0.0)), 1)
            q["impact_speed_m_s"] = round(float(np.linalg.norm(L["ball_vel"][i1 - 1, b])), 3)
            W = (t >= t[i1]) & (t < t[i1] + 2.5)
            nc = L["net_c"][:, 2]
            q["net_centre_sink_peak_mm"] = round(float(1e3 * (zc0 - nc[W].min())), 1)
            q["drone_alt_dev_peak_mm"] = round(float(1e3 * np.abs(pos[W, :, 2] - ref[W, :, 2]).max()), 1)
            q["tilt_peak_deg"] = round(float(L["tilt"][W].max()), 2)
            q["tension_peak_gf"] = round(float(L["tension"][W].max() / G * 1e3), 2)
            q["motor_peak"] = round(float((L["motor"][W] / catch_params(cfg).f_max).max()), 4)
            q["net_bounce_mm_first_2s"] = round(float(1e3 * (nc[W].max() - nc[W].min())), 1)
        balls.append(q)
    out["balls"] = balls
    tend = float(t[-1])
    last = (t >= tend - 1.0)
    out["net_centre_end_m"] = round(float(L["net_c"][last, 2].mean()), 4)
    out["net_centre_sink_end_mm"] = round(1e3 * (zc0 - out["net_centre_end_m"]), 1)
    out["net_bounce_last_1s_mm"] = round(float(1e3 * np.ptp(L["net_c"][last, 2])), 1)
    out["sag_end_m"] = round(float(L["sag_net"][last].mean()), 4)
    out["tension_end_gf"] = round(float(L["tension"][last].mean() / G * 1e3), 2)
    out["all_in_net"] = all(r["outcome"] == "caught" for r in m["balls"])
    if len(m["balls"]) > 1:                 # sag after each drop: mean over the 1 s before the next one (or the end)
        ts = [r["t_launch"] for r in m["balls"]] + [tend + 1e-9]
        out["sag_after_each_m"] = [round(float(L["sag_net"][(t >= ts[i + 1] - 1.0) & (t < ts[i + 1])].mean()), 4)
                                   for i in range(len(m["balls"]))]
        out["sag_after_each_m"] = [round(sag0, 4)] + out["sag_after_each_m"]
        out["drone_z_after_each_m"] = [round(float(pos[(t >= ts[i + 1] - 1.0) & (t < ts[i + 1]), :, 2].mean()), 4)
                                       for i in range(len(m["balls"]))]
        hd = lambda w: float(pos[w, :, 2].mean() - L["net_c"][w, 2].mean())
        out["h_dn_after_each_m"] = [round(hd(pre), 4)] + [round(hd((t >= ts[i + 1] - 1.0) & (t < ts[i + 1])), 4)
                                                          for i in range(len(m["balls"]))]
    out["safety_violations"] = m["safety_violations"]
    return out


def plot_drop_test(out, res):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for a, name in zip(ax, ("a_drop_1m", "b_five_drops", "c_fast_oblique")):
        L = dict(np.load(os.path.join(out, f"drop_{name}_log.npz")))
        t = L["t"]
        W = t >= 15.0
        a.plot(t[W], L["net_c"][W, 2], "k-", lw=1, label="net centre z")
        for b in range(L["ball_pos"].shape[1]):
            a.plot(t[W], L["ball_pos"][W, b, 2] - 0.02, "-", lw=1, label=f"ball {b + 1} bottom z" if b < 5 else None)
        a.plot(t[W], L["pos"][W, :, 2].mean(axis=1) - 0.4, "b--", lw=1, label="drone z - 0.4 m")
        a.set_ylim(0.3, 1.9)
        a.set_title(name)
        a.set_xlabel("t [s]")
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "drop_test.png"), dpi=90)
    plt.close(fig)


REACH_D = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
REACH_T = (1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6)
REACH_DIRS = {"away": 0.0, "lateral": 90.0, "toward": 180.0}       # deg from +x (the launcher fires along +x)
REACH_STANDBY_BOX = (6.0, 5.0)


def reach_cfgs(base, h_dn, k, z_net0):
    sb = np.array([REACH_STANDBY_BOX[0], REACH_STANDBY_BOX[1], 0.0]) - BOX_OFFSET
    sb[2] = z_net0 + h_dn
    z_aim = base["ref_z_net"] + base["aim_alpha"] * h_dn        # the axis height stays 1.0 m + 0.3 drop
    cfgs, cells = [], []
    for dname, ang in REACH_DIRS.items():
        u = np.array([np.cos(np.radians(ang)), np.sin(np.radians(ang)), 0.0])
        for d in REACH_D:
            if d == 0 and dname != "away":
                continue
            for T in REACH_T:
                P = np.array([sb[0], sb[1], z_aim]) + d * u
                cell = {"dir": dname, "d_m": d, "T_s": T}
                blo, bhi = catch_box_limits(base["spacing"])
                if np.any(P[:2] < blo[:2] - 1e-9) or np.any(P[:2] > bhi[:2] + 1e-9):
                    cell["infeasible"] = "outside the formation's box"
                    cells.append(cell)
                    continue
                v0, ok = solve_launch(barrel_pos(0, 1), P, T, k)
                if not ok:
                    cell["infeasible"] = "no launch solution"
                    cells.append(cell)
                    continue
                inbox, tr = in_box_flight(barrel_pos(0, 1), v0, k)
                c = crossing(tr, z_aim)
                if not inbox or c is None or abs(c[0] - T) > 0.02:
                    cell["infeasible"] = "flight leaves the box" if not inbox else "not descending there"
                    cells.append(cell)
                    continue
                sp = float(np.linalg.norm(v0))
                cell.update(speed_m_s=round(sp, 2), el_deg=round(float(np.degrees(np.arcsin(v0[2] / sp))), 1),
                            apex_m=round(float(tr[:, 3].max()), 2),
                            descent_deg=round(float(np.degrees(np.arctan2(-c[6], np.hypot(c[4], c[5])))), 1))
                cells.append(cell)
                cfgs.append({**base, "tier": None, "standby": sb.tolist(), "z_net0": z_net0, "h_dn": h_dn,
                             "launches": [{"dt": 0.0, "p0": barrel_pos(0, 1).tolist(), "v0": v0.tolist(),
                                           "v_prior": v0.tolist()}],
                             "seed": 1000 + len(cfgs), "label": f"reach_{dname}_d{d:g}_T{T:g}",
                             "_cell": len(cells) - 1})
    return cfgs, cells


def catch_reach_map(out, jobs, base=None):
    """Single-ball reachability: landing offset from the standby point (at the standby aim height) x time from launch
    to that crossing, in three directions; one trial per cell (the launch solved for that point and time)."""
    base = {**CATCH_DEFAULT, **(base or {})}
    os.makedirs(out, exist_ok=True)
    h_dn = hold_drop() if base["h_dn"] is None else float(base["h_dn"])
    p = catch_params(base)
    cfgs, cells = reach_cfgs(base, h_dn, ball_k(p), base["z_net0"])
    print(f"reach map: {len(cfgs)} trials ({len(cells) - len(cfgs)} cells without a launch inside the box)", flush=True)
    idx = [c.pop("_cell") for c in cfgs]
    t0 = time.time()
    with _pool(jobs) as pool:
        res = pool.map(_catch_job, cfgs, chunksize=1)
    for i, m in zip(idx, res):
        c = cells[i]
        if "error" in m:
            c["error"] = m["error"]
            continue
        b = m["balls"][0]
        c.update(caught=b["outcome"] == "caught", reason=b.get("reason"), clean=m["clean"],
                 violations=m["safety_violations"], motor_max=m["safety"]["motor90"]["motor_max"],
                 disc_min_mm=m["safety"]["clearance"]["disc_min_mm"], cross_offset_m=b.get("cross_offset_m"),
                 plan_status_last=b.get("plan_status_last"))
    js = {"standby_box": list(REACH_STANDBY_BOX) + [base["z_net0"] + h_dn], "z_net0": base["z_net0"], "h_dn_m": h_dn,
          "aim_height_m": base["ref_z_net"] + base["aim_alpha"] * h_dn, "launcher_box": list(CATCH_LAUNCHER_BOX),
          "D": REACH_D, "T": REACH_T, "dirs": REACH_DIRS, "cells": cells, "limits": base["limits"],
          "wall_s": round(time.time() - t0, 1)}
    with open(os.path.join(out, "reach_map.json"), "w") as f:
        json.dump(js, f, indent=1, default=_json_np)
    plot_reach_map(js, os.path.join(out, "reach_map.png"))
    return js


def plot_reach_map(js, path, tiers=None):
    dirs = list(js["dirs"])
    D, T = js["D"], js["T"]
    fig, ax = plt.subplots(1, len(dirs) + 1, figsize=(5.2 * (len(dirs) + 1), 4.8))
    comb = np.full((len(T), len(D)), np.nan)
    cnt = np.zeros((len(T), len(D)))
    for a, dn in zip(ax, dirs):
        M = np.full((len(T), len(D)), np.nan)
        lab = [["" for _ in D] for _ in T]
        for c in js["cells"]:
            if c["dir"] != dn and not (c["d_m"] == 0):
                continue
            i, j = T.index(c["T_s"]), D.index(c["d_m"])
            if "caught" in c:
                M[i, j] = 1.0 if c["caught"] and c["clean"] else (0.6 if c["caught"] else 0.0)
                lab[i][j] = ("C" if c["caught"] else {"late": "L", "offset": "O", "fell_out": "F", "hit_drone": "H"}
                             .get(c.get("reason"), "x")) + ("" if c["clean"] else "*")
                comb[i, j] = (0 if np.isnan(comb[i, j]) else comb[i, j]) + float(c["caught"])
                cnt[i, j] += 1
            elif c.get("infeasible"):
                lab[i][j] = "-"
        a.imshow(M, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        for i in range(len(T)):
            for j in range(len(D)):
                a.text(j, i, lab[i][j], ha="center", va="center", fontsize=8)
        a.set_xticks(range(len(D)), [f"{d:g}" for d in D])
        a.set_yticks(range(len(T)), [f"{t:g}" for t in T])
        a.set_xlabel("landing offset from standby [m]")
        a.set_ylabel("time from launch to the aim height [s]")
        a.set_title(f"direction: {dn}", fontsize=10)
    a = ax[-1]
    frac = comb / np.maximum(cnt, 1)
    frac[cnt == 0] = np.nan
    a.imshow(frac, origin="lower", cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(len(T)):
        for j in range(len(D)):
            if cnt[i, j]:
                a.text(j, i, f"{int(comb[i, j])}/{int(cnt[i, j])}", ha="center", va="center", fontsize=8)
    if tiers:
        for name, (dd, tt), col in tiers:
            a.scatter(np.interp(dd, D, range(len(D))), np.interp(tt, T, range(len(T))), s=4, color=col, alpha=0.4,
                      label=name)
        a.legend(fontsize=7, loc="upper right")
    a.set_xticks(range(len(D)), [f"{d:g}" for d in D])
    a.set_yticks(range(len(T)), [f"{t:g}" for t in T])
    a.set_xlabel("landing offset from standby [m]")
    a.set_ylabel("time from launch to the aim height [s]")
    a.set_title("all directions: caught / trials" + ("\n(dots: launch ranges of the tiers)" if tiers else ""), fontsize=10)
    fig.suptitle("single ball from the standby hover: C caught (dark green: no safety item exceeded, light: caught with "
                 "one exceeded, *), L late, O offset, F fell out, H hit a drone, - no launch / outside the box", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=90)
    plt.close(fig)


def tier_scatter(rg, k, cfg, h_dn, n=600):
    """Offset from the tier's standby and time from launch at the standby aim height, per Monte-Carlo launch."""
    sb = catch_standby(rg, k, cfg, h_dn)
    pts, out_share = range_scatter(rg, k, cfg["z_net0"] + cfg["aim_alpha"] * h_dn, n=n, seed=99)
    d = np.hypot(pts[:, 1] - sb[0], pts[:, 2] - sb[1])
    return sb, d, pts[:, 0], pts, out_share


def catch_tier_table(base=None):
    base = {**CATCH_DEFAULT, **(base or {})}
    p = catch_params(base)
    k = ball_k(p)
    h_dn = hold_drop() if base["h_dn"] is None else float(base["h_dn"])
    rows = {}
    for name in CATCH_TIERS:
        rg = tier_ranges({**base, "tier": name})
        sb, d, T, pts, out_share = tier_scatter(rg, k, base, h_dn)
        rows[name] = {**{k_: list(v) for k_, v in rg.items()}, "standby_box": (sb + BOX_OFFSET).round(3).tolist(),
                      "offset_m": {"mean": round(float(d.mean()), 3), "p50": round(float(np.median(d)), 3),
                                   "p90": round(float(np.percentile(d, 90)), 3), "max": round(float(d.max()), 3)},
                      "spread_x_m": round(float(np.ptp(pts[:, 1])), 3), "spread_y_m": round(float(np.ptp(pts[:, 2])), 3),
                      "time_s": {"min": round(float(T.min()), 3), "max": round(float(T.max()), 3),
                                 "mean": round(float(T.mean()), 3)},
                      "flight_to_floor_note": "time from launch to the standby aim height (net centre 1.0 m + 0.12 m)",
                      "rejected_out_of_box_share": round(out_share, 3), "_d": d, "_T": T}
    return rows


def catch_intercept_stats(balls):
    """Distribution of the interception heights and formation speeds over the balls (planner target net-centre height,
    actual net-centre height at the first contact, candidate counts)."""
    def dist(v):
        v = [x for x in v if x is not None]
        if not v:
            return None
        v = np.array(v, float)
        return {"n": int(len(v)), "mean": round(float(v.mean()), 3), "p10": round(float(np.percentile(v, 10)), 3),
                "p50": round(float(np.median(v)), 3), "p90": round(float(np.percentile(v, 90)), 3),
                "max": round(float(v.max()), 3)}
    ic = [b.get("intercept") or {} for b in balls]
    cg = [b.get("intercept") or {} for b in balls if b["outcome"] == "caught"]
    low = [x["target_is_lowest_feasible"] for x in ic if x.get("target_is_lowest_feasible") is not None]
    hist = {}
    for x in cg:
        z = x.get("net_centre_z_at_contact_m")
        if z is not None:
            key = f"{np.floor(z * 2) / 2:.1f}"
            hist[key] = hist.get(key, 0) + 1
    ft = [b.get("intercept_first_target") or {} for b in balls]
    lf = [x["lowest_feasible"] for x in ft if x.get("lowest_feasible") is not None]
    return {"first_target_zn_m": dist([x.get("zn_m") for x in ft]),
            "first_target_n_candidates": dist([x.get("n_candidates") for x in ft]),
            "first_target_n_feasible": dist([x.get("n_feasible") for x in ft]),
            "first_target_lowest_feasible_share": round(float(np.mean(lf)), 3) if lf else None,
            "max_candidates": dist([b.get("intercept_max_candidates") for b in balls]),
            "max_feasible": dist([b.get("intercept_max_feasible") for b in balls]),
            "replans_with_feasible_share": dist([b.get("intercept_replans_with_feasible_share") for b in balls]),
            "zn_target_m": dist([x.get("zn_target_m") for x in ic]),
            "v_end_target_m_s": dist([x.get("v_end_target_m_s") for x in ic]),
            "moving_target_share": (round(float(np.mean([x["v_end_target_m_s"] > 1e-6 for x in ic
                                                          if x.get("v_end_target_m_s") is not None])), 3)
                                    if any(x.get("v_end_target_m_s") is not None for x in ic) else None),
            "net_centre_z_at_contact_caught_m": dist([x.get("net_centre_z_at_contact_m") for x in cg]),
            "contact_height_hist_0p5m": dict(sorted(hist.items(), key=lambda kv: float(kv[0]))),
            "n_feasible": dist([x.get("n_feasible") for x in ic]),
            "n_candidates": dist([x.get("n_candidates") for x in ic]),
            "target_is_lowest_feasible_share": round(float(np.mean(low)), 3) if low else None,
            "v_h_at_contact_caught_m_s": dist([x.get("v_h_at_contact_m_s") for x in cg]),
            "v_z_at_contact_caught_m_s": dist([x.get("v_z_at_contact_m_s") for x in cg])}


def catch_agg(ms):
    """Aggregate of trial summaries."""
    ms = [m for m in ms if "error" not in m]
    if not ms:
        return {}
    balls = [b for m in ms for b in m["balls"] if b["outcome"] != "not_launched"]
    nb = len(balls)
    reasons = {}
    for b in balls:
        if b["outcome"] != "caught":
            reasons[b["reason"]] = reasons.get(b["reason"], 0) + 1
    by_idx = {}
    for b in balls:
        by_idx.setdefault(b["ball"] + 1, []).append(b["outcome"] == "caught")
    by_net = {}
    net_state = {}
    for b in balls:
        if b.get("balls_in_net_at_cross") is None:
            continue
        kq = b["balls_in_net_at_cross"]
        by_net.setdefault(kq, []).append(b["outcome"] == "caught")
        net_state.setdefault(kq, []).append((b["net_at_cross"]["sag_m"], b["net_at_cross"]["swing_mm"]))
    viol = {}
    for m in ms:
        for v in m["safety_violations"]:
            viol[v] = viol.get(v, 0) + 1
    S = lambda key, sub: [m["safety"][key][sub] for m in ms if m["safety"][key].get(sub) is not None]
    st = lambda x: ({"mean": round(float(np.mean(x)), 4), "min": round(float(np.min(x)), 4),
                     "max": round(float(np.max(x)), 4)} if len(x) else None)
    by_n = {}
    for m in ms:
        by_n.setdefault(m["n_balls"], []).append(m["all_caught"])
    return {"trials": len(ms), "balls": nb, "caught": sum(b["outcome"] == "caught" for b in balls),
            "caught_share_balls": round(sum(b["outcome"] == "caught" for b in balls) / max(nb, 1), 4),
            "caught_share_trial_mean": round(float(np.mean([m["caught_share"] for m in ms])), 4),
            "all_caught_share": round(float(np.mean([m["all_caught"] for m in ms])), 4),
            "all_caught_clean_share": round(float(np.mean([m["all_caught_clean"] for m in ms])), 4),
            "clean_share": round(float(np.mean([m["clean"] for m in ms])), 4),
            "caught_clean_balls_share": round(sum(b["outcome"] == "caught" for m in ms if m["clean"] for b in m["balls"])
                                              / max(nb, 1), 4),
            "all_caught_by_n_balls": {str(k): [int(sum(v)), len(v)] for k, v in sorted(by_n.items())},
            "by_ball_index": {str(k): [int(sum(v)), len(v), round(float(np.mean(v)), 3)] for k, v in sorted(by_idx.items())},
            "by_balls_in_net_at_cross": {str(k): [int(sum(v)), len(v), round(float(np.mean(v)), 3)]
                                         for k, v in sorted(by_net.items())},
            "net_at_cross_by_balls_in_net": {str(k): {"sag_m_mean": round(float(np.mean([x[0] for x in v])), 4),
                                                      "swing_mm_mean": round(float(np.mean([x[1] for x in v])), 1),
                                                      "n": len(v)} for k, v in sorted(net_state.items())},
            "intercept": catch_intercept_stats(balls),
            "miss_reasons": reasons, "safety_violation_trials": viol,
            "aborted": int(sum(m["aborted"] for m in ms)),
            "motor_max": st(S("motor90", "motor_max")), "motor90_longest_s": st(S("motor90", "longest_s")),
            "disc_min_mm": st(S("clearance", "disc_min_mm")), "guard_min_mm": st(S("clearance", "guard_min_mm")),
            "tilt_max_deg": st(S("control", "tilt_max_deg")), "alt_err_max_mm": st(S("control", "alt_err_max_mm")),
            "net_lowest_m": st(S("net_low", "net_lowest_m")), "slack_max_s": st(S("slack", "slack_max_s")),
            "ball_hits_drone_trials": int(sum(m["safety"]["ball_rotor"]["shell_hits"] > 0 for m in ms)),
            "ball_disc_gap_min_mm": st([x for x in S("ball_rotor", "disc_gap_min_mm")]),
            "wall_s_mean": round(float(np.mean([m["wall_s"] for m in ms])), 1)}


def prediction_curve(ms, times=(0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2, 1.4)):
    """Median / 90th percentile of the predicted floor point and standby-height crossing errors vs time since launch."""
    rows = {"t_since_launch_s": list(times), "floor_err_m": {"p50": [], "p90": [], "n": []},
            "aim_err_m": {"p50": [], "p90": [], "n": []}, "aim_time_err_s": {"p50": [], "p90": [], "n": []}}
    per = [b["prediction"] for m in ms if "error" not in m for b in m["balls"] if b.get("prediction")]
    for tq in times:
        for key, col in (("floor_err_m", 1), ("aim_err_m", 2), ("aim_time_err_s", 3)):
            vals = []
            for pr in per:
                a = np.array([[x[0], np.nan if x[col] is None else abs(x[col])] for x in pr], float)
                j = np.flatnonzero((a[:, 0] >= tq - 0.026) & (a[:, 0] <= tq + 0.026))
                if len(j) and np.isfinite(a[j[0], 1]):
                    vals.append(a[j[0], 1])
            rows[key]["p50"].append(round(float(np.median(vals)), 4) if vals else None)
            rows[key]["p90"].append(round(float(np.percentile(vals, 90)), 4) if vals else None)
            rows[key]["n"].append(len(vals))
    return rows


CATCH_SNAPSHOT_DIR = [None]         # persistent snapshot cache (--snapshot-dir); None: one temporary cache per batch


def run_catch_batch(cfgs, jobs, out, name, save_logs=()):
    """Run trial configs in parallel; trials whose label is in save_logs keep their log, plot and summary in out.
    Trials that share a seed and every pre-launch setting start from one pre-launch snapshot (catch_snapshot_key;
    bit-identical to running them from the start): the cache is CATCH_SNAPSHOT_DIR[0] if set, else a temporary
    folder deleted after the batch (snapshots off: CATCH_SNAPSHOT_DIR[0] = "off")."""
    import shutil
    import tempfile
    tmp = None
    if CATCH_SNAPSHOT_DIR[0] != "off":
        sd = CATCH_SNAPSHOT_DIR[0] or (tmp := tempfile.mkdtemp(prefix="net_sim_snap_"))
        os.makedirs(sd, exist_ok=True)
        for c in cfgs:
            c.setdefault("snap_dir", sd)
    # the configs keep their order (settings outer, seeds inner): the first setting writes the snapshots, the later
    # ones reuse them (a trial that starts before its snapshot exists simply runs from the start: same result)
    for c in cfgs:
        if c.get("label") in save_logs:
            c["_out"], c["_tag"], c["_save_log"] = out, c["label"], True
    t0 = time.time()
    try:
        with _pool(jobs) as pool:
            res = pool.map(_catch_job, cfgs, chunksize=1)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    err = [r for r in res if "error" in r]
    print(f"{name}: {len(res)} trials in {time.time() - t0:.0f} s, {len(err)} errors", flush=True)
    for e in err[:3]:
        print(e["trace"], flush=True)
    return res


def _compact(m):
    keep = ("n_balls", "n_caught", "all_caught", "caught_share", "safety_violations", "clean", "all_caught_clean",
            "aborted", "t_end_s", "standby_box", "t_launch0_s", "wall_s", "cfg_key", "error")
    c = {k: m.get(k) for k in keep if k in m}
    if "safety" in m:
        c["safety"] = m["safety"]
        c["balls"] = [{k: v for k, v in b.items() if k not in ("prediction", "pred_hist")} for b in m["balls"]]
    return c


def catch_study(which, out, jobs, base=None, seeds=30, tiers=None, intervals=None):
    """Studies of the catch mode (see main, --catch-study)."""
    base = {**CATCH_DEFAULT, **(base or {})}
    S = list(range(1, seeds + 1))
    os.makedirs(out, exist_ok=True)
    js = {"which": which, "seeds": S, "base": _catch_cfg_json(base), "tiers": CATCH_TIERS,
          "limits": CATCH_LIMITS_SETS[base["limits"]] if isinstance(base["limits"], str) else base["limits"]}
    if which == "tiers":
        groups = [(t_, mode) for t_ in (tiers or ("easy", "medium", "hard")) for mode in ("sequence", "salvo")]
        cfgs = [{**base, "tier": t_, "launch": mode, "seed": s, "label": f"{t_}_{mode}_seed{s}"}
                for t_, mode in groups for s in S]
        res = run_catch_batch(cfgs, jobs, out, "tiers")
        js["groups"] = {}
        for t_, mode in groups:
            ms = [m for m in res if m.get("cfg_key", {}).get("tier") == t_ and m["cfg_key"]["launch"] == mode]
            js["groups"][f"{t_}_{mode}"] = {"agg": catch_agg(ms), "prediction": prediction_curve(ms),
                                           "trials": [_compact(m) for m in ms]}
        tt = catch_tier_table(base)
        tt = {k: v for k, v in tt.items() if k in (tiers or ("easy", "medium", "hard"))}
        js["tier_table"] = {k: {k2: v2 for k2, v2 in v.items() if not k2.startswith("_")} for k, v in tt.items()}
        rp = os.path.join(os.path.dirname(out), "reach", "reach_map.json")
        if os.path.exists(rp):                 # the tiers' Monte-Carlo launches on the reachability map
            with open(rp) as f:
                rj = json.load(f)
            plot_reach_map(rj, os.path.join(os.path.dirname(out), "reach", "reach_map_tiers.png"),
                           tiers=[(k, (v["_d"], v["_T"]), c) for (k, v), c in
                                  zip(tt.items(), ("tab:blue", "k", "tab:red", "tab:purple", "tab:orange"))])
        plot_catch_tiers(js, os.path.join(out, "tiers.png"))
    elif which == "interval":
        ivs = tuple(intervals or (0.5, 1.0, 2.0, 3.0))
        cfgs = [{**base, "tier": "medium", "launch": "sequence", "interval_s": iv, "seed": s,
                 "label": f"interval{iv:g}_seed{s}"} for iv in ivs for s in S]
        res = run_catch_batch(cfgs, jobs, out, "interval")
        js["groups"] = {f"{iv:g}": {"agg": catch_agg([m for m in res if m.get("cfg_key", {}).get("interval_s") == iv]),
                                    "trials": [_compact(m) for m in res if m.get("cfg_key", {}).get("interval_s") == iv]}
                        for iv in ivs}
        plot_catch_curve(js, os.path.join(out, "interval.png"), "interval between launches [s]")
    elif which == "noise":
        sig = (0.0, 0.005, 0.05)
        cfgs = [{**base, "tier": "medium", "launch": "sequence", "noise_m": sg, "seed": s,
                 "label": f"noise{1e3 * sg:g}mm_seed{s}"} for sg in sig for s in S]
        res = run_catch_batch(cfgs, jobs, out, "noise")
        js["groups"] = {}
        for sg in sig:
            ms = [m for m in res if m.get("cfg_key", {}).get("noise_m") == sg]
            js["groups"][f"{1e3 * sg:g}"] = {"agg": catch_agg(ms), "prediction": prediction_curve(ms),
                                             "trials": [_compact(m) for m in ms]}
        plot_catch_curve(js, os.path.join(out, "noise.png"), "position noise sigma [mm]")
        plot_prediction(js, os.path.join(out, "prediction_convergence.png"))
    elif which == "single":                  # one group with the base settings (e.g. the 0.5 m net)
        cfgs = [{**base, "seed": s, "label": f"{base['tier']}_{base['launch']}_seed{s}"} for s in S]
        res = run_catch_batch(cfgs, jobs, out, "single")
        js["groups"] = {"base": {"agg": catch_agg(res), "prediction": prediction_curve(res),
                                 "trials": [_compact(m) for m in res]}}
    with open(os.path.join(out, f"catch_{which}.json"), "w") as f:
        json.dump(js, f, indent=1, default=_json_np)
    return js


def plot_catch_tiers(js, path):
    G_ = js["groups"]
    names = list(G_)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    x = np.arange(len(names))
    a = ax[0]
    for off, key, lab in ((-0.27, "caught_share_balls", "balls caught"), (-0.09, "all_caught_share", "trials: all caught"),
                          (0.09, "all_caught_clean_share", "all caught, no safety item exceeded"),
                          (0.27, "clean_share", "trials with no safety item exceeded")):
        a.bar(x + off, [G_[n]["agg"][key] for n in names], 0.18, label=lab)
    a.set_xticks(x, names, rotation=30, fontsize=8)
    a.set_ylim(0, 1.05)
    a.legend(fontsize=7)
    a.set_title("share")
    a = ax[1]
    rs = ["late", "offset", "fell_out", "hit_drone", "crowded_out", "trial_aborted"]
    bot = np.zeros(len(names))
    for r in rs:
        v = np.array([G_[n]["agg"]["miss_reasons"].get(r, 0) / max(G_[n]["agg"]["balls"], 1) for n in names])
        a.bar(x, v, 0.6, bottom=bot, label=r)
        bot += v
    a.set_xticks(x, names, rotation=30, fontsize=8)
    a.legend(fontsize=7)
    a.set_title("missed balls by reason (share of all balls)")
    a = ax[2]
    for n in names:
        bi = G_[n]["agg"]["by_ball_index"]
        a.plot([int(k) for k in bi], [v[2] for v in bi.values()], "o-", label=n)
    a.set_xlabel("ball (launch order)")
    a.set_ylabel("caught share")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_catch_curve(js, path, xlabel):
    G_ = js["groups"]
    xs = [float(k) for k in G_]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    a = ax[0]
    for key, lab in (("all_caught_share", "trials: all caught"), ("caught_share_balls", "balls caught"),
                     ("all_caught_clean_share", "all caught, no safety item exceeded")):
        a.plot(xs, [G_[k]["agg"][key] for k in G_], "o-", label=lab)
    a.set_xlabel(xlabel)
    a.set_ylim(0, 1.05)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    a = ax[1]
    rs = ["late", "offset", "fell_out", "hit_drone", "crowded_out", "trial_aborted"]
    bot = np.zeros(len(xs))
    for r in rs:
        v = np.array([G_[k]["agg"]["miss_reasons"].get(r, 0) / max(G_[k]["agg"]["balls"], 1) for k in G_])
        a.bar(range(len(xs)), v, 0.6, bottom=bot, label=r)
        bot += v
    a.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    a.set_xlabel(xlabel)
    a.set_title("missed balls by reason (share of all balls)")
    a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def plot_prediction(js, path):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, g in js["groups"].items():
        pr = g.get("prediction")
        if not pr:
            continue
        tq = pr["t_since_launch_s"]
        for a, key, lab in zip(ax, ("floor_err_m", "aim_err_m", "aim_time_err_s"),
                               ("predicted floor point error [m]", "predicted crossing point error at the standby aim "
                                "height [m]", "predicted crossing time error [s]")):
            l, = a.plot(tq, [np.nan if v is None else v for v in pr[key]["p50"]], "o-", label=f"sigma {k} mm, median")
            a.plot(tq, [np.nan if v is None else v for v in pr[key]["p90"]], "--", color=l.get_color(),
                   label=f"sigma {k} mm, 90th pct")
            a.set_yscale("log")
            a.set_xlabel("time since launch [s]")
            a.set_title(lab, fontsize=9)
            a.grid(alpha=0.3, which="both")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def main_catch(a):
    """--mode catch: one trial (default) or a study (--catch-study)."""
    if a.snapshot_dir:
        CATCH_SNAPSHOT_DIR[0] = a.snapshot_dir
    base = {}
    for key, v in (("launch", a.launch), ("interval_s", a.interval), ("interval_range_s", a.interval_range),
                   ("n_balls", a.n_balls), ("noise_m", None if a.noise_mm is None else a.noise_mm / 1e3),
                   ("el_deg", a.el_range), ("az_deg", a.az_range), ("speed_m_s", a.speed_range), ("limits", a.limits)):
        if v is not None:
            base[key] = v
    base["tier"] = a.tier
    sub = "catch"
    po = {}
    if a.thrust_scale != 1.0:
        po["f_max"] = OfficialParams().f_max * a.thrust_scale
        sub = os.path.join(f"thrust{a.thrust_scale:g}", "catch")
    if a.max_tilt is not None:
        po["max_tilt_deg"] = a.max_tilt
    if a.catch_limits:
        cl = a.catch_limits
        lim = dict(json.load(open(cl)) if os.path.isfile(cl) else json.loads(cl))
        if "max_tilt_deg" in lim:
            po["max_tilt_deg"] = lim.pop("max_tilt_deg")
        lim.pop("note", None)
        base.update(lim)
    if a.net_side not in (None, 1.0):           # another net: Params overrides, spacing, hold height, feedforward table
        base["p_over"] = {"L": a.net_side, "m_net": a.net_mass if a.net_mass is not None else 0.020 * a.net_side}
        if a.takeoff_spacing is not None:
            base["p_over"]["d_takeoff"] = a.takeoff_spacing
        base["net_side"] = a.net_side
        sub = os.path.join(sub, f"net{a.net_side:g}".replace(".", ""))
    if po:
        base["p_over"] = {**(base.get("p_over") or {}), **po}
    base["spacing"] = a.spacing
    if a.hold_summary:
        with open(a.hold_summary) as f:
            base["h_dn"] = float(json.load(f)["hold"]["sag_below_drones_m"])
    if a.ff not in (None, "auto"):
        base["ff_path"] = a.ff
    root = a.out or os.path.join(HERE, "out_official", sub)
    print(f"catch mode, output under {root}: {json.dumps(base)}", flush=True)
    if a.catch_study == "drop":
        catch_drop_test(os.path.join(root, "drop"), base)
    elif a.catch_study == "reach":
        catch_reach_map(os.path.join(root, "reach"), a.jobs, base)
    elif a.catch_study:
        catch_study(a.catch_study, os.path.join(root, a.catch_study), a.jobs, base, seeds=a.seeds, tiers=a.tiers,
                    intervals=a.intervals)
    else:
        out = os.path.join(root, "trials")
        os.makedirs(out, exist_ok=True)
        one = {**base, "seed": a.seed}
        if a.snapshot_dir and a.snapshot_dir != "off":     # single trials use snapshots only with --snapshot-dir
            os.makedirs(a.snapshot_dir, exist_ok=True)
            one["snap_dir"] = a.snapshot_dir
        m = run_catch(one, out=out)
        if "p2p" in m:                          # point-to-point move (cfg p2p, e.g. via --catch-limits)
            print(json.dumps(m["p2p"], indent=1))
            print("files: " + ", ".join(os.path.join(out, x) for x in m["files"]))
            return
        if m.get("snapshot"):
            print(f"pre-launch snapshot {m['snapshot']['key']}: "
                  + ("resumed from it" if m["snapshot"]["used"] else "written for later runs"))
        print(f"caught {m['n_caught']} / {m['n_balls']}, safety items exceeded: {m['safety_violations'] or 'none'}")
        for b in m["balls"]:
            print(f"  ball {b['ball'] + 1}: launch t = {b['t_launch']:.2f} s, el {b['el_deg']} deg, az {b['az_deg']} deg, "
                  f"{b['speed_m_s']} m/s -> {b['outcome']}" + (f" ({b['reason']})" if b.get("reason") else ""))
        print("files: " + ", ".join(os.path.join(out, x) for x in m["files"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hold", "sweep", "travel", "transfer", "mission", "catch"], default="hold")
    ap.add_argument("--spacing", type=float, default=1.25,
                    help="drone spacing during the hold in m (default: 1.25, used for all documented results)")
    ap.add_argument("--drone-model", choices=["official", "simple"], default="official",
                    help="official: firmware + Crazyflow identified Crazyflie 2.1 Brushless (default); "
                         "simple: the earlier hand-assumed model")
    ap.add_argument("--out", default=None,
                    help="output folder (default: out_official/ or out/ next to this file, by drone model)")
    ap.add_argument("--takeoff-spacing", type=float, default=None,
                    help="drone spacing at rest and during the climb in m (default: 1.20, drones outside the "
                         "net corners; 0.80 is the old plan with each drone sitting on the net)")
    ap.add_argument("--clearance-warn", type=float, default=None,
                    help="warn for a flight phase whose rotor-to-net clearance is below this, in m (default 0.030)")
    ap.add_argument("--guard-warn", type=float, default=None,
                    help="same for the propeller-guard clearance, in m (default 0.010)")
    ap.add_argument("--yaw-deg", type=float, default=None,
                    help="heading of all four drones in degrees (initial attitude and attitude reference; "
                         "default 45: corner strings run between two rotors; 0: rotors on the world diagonals, "
                         "the earlier default)")
    ap.add_argument("--hook-z", type=float, default=None,
                    help="string hook height relative to the centre of mass in m (default -0.015; earlier "
                         "-0.010); below "
                         f"{GUARD_FEET_Z} (the guard feet) the real drone needs longer landing legs")
    ap.add_argument("--gif", action="store_true")
    ap.add_argument("--net-mass", type=float, default=None, help="net mass in kg (default 0.020 x net side)")
    ap.add_argument("--net-side", type=float, default=None,
                    help="net side in m (default 1.0); same 9 x 9 knots and thread, so the mass scales with the side "
                         "(0.5 m: 10 g) unless --net-mass is given; corner strings stay 0.20 m")
    ap.add_argument("--payload-mass", type=float, default=0.0,
                    help="payload fixed to the centre knot, in g (default 0: none); a 15 mm sphere with floor contact "
                         "and air drag, counted in the hanging weight of the feedforward")
    ap.add_argument("--land-spacing", type=float, default=None,
                    help="drone spacing during the descent and landing in m (default 1.20, never above the hold spacing)")
    ap.add_argument("--sweep-spacings", type=float, nargs="+", default=None,
                    help="--mode sweep: hold spacings to visit (default 0.90 ... 1.33 m for the 1 m net)")
    ap.add_argument("--drone-mass", type=float, default=None, help="drone mass in kg")
    ap.add_argument("--ff", nargs="?", const="auto", default=None,
                    help="use a net-pull feedforward table: a sweep_summary.json path, "
                         "or no value for the one in the output folder")
    tr = ap.add_argument_group("travel mode (--mode travel; official drone model, feedforward on by default)")
    tr.add_argument("--speed", type=float, default=1.0, help="cruise speed in m/s (default 1.0)")
    tr.add_argument("--accel", type=float, default=1.0, help="acceleration limit in m/s^2 (default 1.0)")
    tr.add_argument("--heading", choices=list(TRAVEL_HEADINGS), default="side",
                    help="side: along a net side (+x, drones 0 and 3 lead); diagonal: along a net diagonal "
                         "(45 deg, drone 0 leads)")
    tr.add_argument("--heading-deg", type=float, default=None, help="any other heading in degrees from +x")
    tr.add_argument("--cruise-time", type=float, default=8.0, help="cruise duration in s (default 8)")
    tr.add_argument("--settle-time", type=float, default=8.0, help="hold after the stop in s (default 8)")
    tr.add_argument("--acc-ff", choices=["on", "off", "net"], default="on",
                    help="reference-acceleration feedforward: m_drone * a_ref (on, default), off, or "
                         "(m_drone + m_net / 4) * a_ref (net)")
    tr.add_argument("--sweep", action="store_true",
                    help="run the full travel test (all headings, accelerations, speeds and the sensitivity "
                         "variants, in parallel processes) instead of one trip; takes several minutes")
    tr.add_argument("--jobs", type=int, default=max(1, min(22, (os.cpu_count() or 2) - 2)),
                    help="parallel processes for --sweep")
    tf = ap.add_argument_group("transfer mode (--mode transfer: shortest A -> B move, net shape required only at B; "
                               "uses --speed (peak), --accel (speeding up), --heading, --sweep, --jobs)")
    tf.add_argument("--distance", type=float, default=10.0, help="distance A -> B in m (default 10)")
    tf.add_argument("--brake-accel", type=float, default=None, help="braking acceleration limit (default: --accel)")
    tf.add_argument("--contract", type=float, default=None,
                    help="contract the spacing to this value (m) while speeding up and cruising (default: none)")
    tf.add_argument("--contract-time", type=float, default=None, help="contraction ramp in s (default: speed-up ramp)")
    tf.add_argument("--respread-start", type=float, default=0.0,
                    help="start of the re-spread to the hold spacing, s after the brake starts (default 0)")
    tf.add_argument("--respread-time", type=float, default=None, help="re-spread ramp in s (default: brake ramp)")
    tf.add_argument("--climb", type=float, default=0.0, help="climb by this much (m) with the contraction (default 0)")
    tf.add_argument("--max-tilt", type=float, default=None, help="controller tilt limit in deg (sensitivity only)")
    ms = ap.add_argument_group("mission mode (--mode mission: random net-centre targets in a 10 x 10 x 10 m box, "
                               "relaxed precision, rigid pattern; one mission by default)")
    ms.add_argument("--seed", type=int, default=1, help="mission seed (default 1)")
    ms.add_argument("--targets", type=int, default=10, help="number of targets (default 10)")
    ms.add_argument("--planner", choices=["relaxed", "strict"], default="relaxed")
    ms.add_argument("--factor", type=float, default=None, help="planner aggressiveness factor (default: the "
                    "passing setting of the study, MISSION_FACTOR)")
    ms.add_argument("--settle", choices=["avg", "inst", "pos"], default="avg",
                    help="settle rule at each target: avg (0.5 s mean of sag and net offset), inst, pos (drones only)")
    ms.add_argument("--tol-pos", type=float, default=None,
                    help="settle tolerance: largest drone distance from its final hold position, mm (default 50)")
    ms.add_argument("--tol-net-off", type=float, default=None,
                    help="settle tolerance: net-centre horizontal offset from the formation centre, mm (default 50)")
    ms.add_argument("--tol-sag", type=float, default=None,
                    help="settle tolerance: net sag change from its hold value, mm (default 20)")
    ms.add_argument("--rotor-check", action="store_true",
                    help="fail a leg on net_below_rotors as well (first study; default: recorded only)")
    ms.add_argument("--first-study", action="store_true",
                    help="the first study's rules: tolerances 10 / 10 / 5 mm and --rotor-check (individual --tol-* "
                         "options still override); output default out_official/missions")
    ms.add_argument("--final-strict", type=float, default=30.0,
                    help="after the last target, hover up to this many s and time the strict 10 / 10 / 5 mm rule "
                         "(default 30; 0: off; inactive when the tolerances are the strict ones)")
    ms.add_argument("--hold-summary", default=None,
                    help="hold summary JSON of this net: its drone height above the net centre sets the formation "
                         "height over each net-centre target (default out_official/hold_ff_summary.json)")
    ms.add_argument("--planner-limits", default=None,
                    help="relaxed-planner limits replacing PLANNER_RELAXED entries: a JSON file or a JSON string")
    ms.add_argument("--targets-from", default=None,
                    help="a mission-study JSON whose targets are reused for the same seed (e.g. to compare nets)")
    ms.add_argument("--compare-study", default=None,
                    help="--study: reference study for the leg-by-leg comparison and the 'ref_rule' runs (its "
                         "relaxed rule at its factor in this configuration; default: the first study)")
    ms.add_argument("--factors", type=float, nargs="+", default=None, help="--study: planner factors to sweep")
    ms.add_argument("--no-strict", action="store_true", help="--study: skip the strict-planner missions")
    ms.add_argument("--no-ref-rule", action="store_true",
                    help="--study: skip the reference study's rule ('ref_rule') missions")
    ms.add_argument("--tol-sweep", type=float, nargs="*", default=None,
                    help="--study: tolerance sweep at the chosen factor, mm (default 10 20 30 50 100; none: no values)")
    ms.add_argument("--characterise", action="store_true",
                    help="step 1: climb / descent grids, oblique and level legs (parallel; several minutes)")
    ms.add_argument("--study", action="store_true",
                    help="the full study: 10 seeds x 10 targets at several factors, the strict planner and the "
                         "settle-rule variants (parallel; tens of minutes)")
    ca = ap.add_argument_group("catch mode (--mode catch: a launcher fires 3-5 balls, the formation intercepts them with "
                               "the net; uses --seed, --net-side, --spacing, --takeoff-spacing, --hold-summary, --ff, "
                               "--jobs)")
    ca.add_argument("--tier", choices=list(CATCH_TIERS), default="medium", help="launch ranges (default medium)")
    ca.add_argument("--launch", choices=["sequence", "salvo"], default="sequence",
                    help="sequence: one ball after another (--interval); salvo: all at once from side-by-side barrels")
    ca.add_argument("--interval", type=float, default=None, help="time between launches in s (default 2.0)")
    ca.add_argument("--interval-range", type=float, nargs=2, default=None, help="random interval, uniform in [a, b] s")
    ca.add_argument("--n-balls", type=int, nargs=2, default=None, help="number of balls, uniform in [a, b] (default 3 5)")
    ca.add_argument("--noise-mm", type=float, default=None, help="ball position noise sigma in mm (default 5)")
    ca.add_argument("--el-range", type=float, nargs=2, default=None, help="elevation range in deg (overrides the tier)")
    ca.add_argument("--az-range", type=float, nargs=2, default=None, help="azimuth range in deg from +x")
    ca.add_argument("--speed-range", type=float, nargs=2, default=None, help="launch speed range in m/s")
    ca.add_argument("--limits", choices=list(CATCH_LIMITS_SETS), default=None, help="formation limits (default relaxed)")
    ca.add_argument("--catch-study", choices=["drop", "reach", "tiers", "interval", "noise", "single"], default=None,
                    help="drop: net-contact check; reach: single-ball reachability map; tiers: 3 tiers x sequence / "
                         "salvo; interval: medium sequence at 0.5 / 1 / 2 / 3 s; noise: sigma 0 / 5 / 50 mm; single: "
                         "the given settings over the seeds (e.g. the 0.5 m net)")
    ca.add_argument("--seeds", type=int, default=30, help="--catch-study: seeds per group (default 30)")
    ca.add_argument("--catch-limits", default=None,
                    help="catch settings replacing CATCH_DEFAULT entries (limits, loaded_accel_factor, "
                         "standby_accel_factor, t_hold_s, ..., and max_tilt_deg for the controller): JSON file or string. "
                         "The planner's box / height-floor fix is on by default; '{\"boxfix\": false}' turns it off "
                         "(results as before 2026-11-23)")
    ca.add_argument("--tiers", nargs="+", default=None, help="--catch-study tiers: tiers to run (default easy medium hard)")
    ca.add_argument("--snapshot-dir", default=None,
                    help="keep pre-launch snapshots in this folder and reuse them across commands (keyed by every "
                         "pre-launch setting, the seed and this source file; results are bit-identical). Studies: "
                         "default a temporary folder per batch, 'off' disables. Single trials: only with this option")
    ca.add_argument("--intervals", type=float, nargs="+", default=None,
                    help="--catch-study interval: intervals in s (default 0.5 1 2 3)")
    ap.add_argument("--thrust-scale", type=float, default=1.0,
                    help="multiply each motor's maximum thrust (f_max) by this (default 1; the thrust and torque maps, "
                         "rotor dynamics, minimum thrust, mass, size, inertia and drag are unchanged); results then go "
                         "to out_official/thrust<scale>/ by default")
    ap.add_argument("--net-drag", choices=["on", "off"], default=None,
                    help="quadratic air drag on the net threads (default: on in travel mode, off in hold / sweep)")
    ap.add_argument("--keep-linear-drag", action="store_true",
                    help="with --net-drag on, keep the linear knot damping (node_drag stand-in) as well")
    ap.add_argument("--thread-d", type=float, default=None,
                    help="net thread diameter in m (default: derived from the net mass, nylon 1150 kg/m^3)")
    a = ap.parse_args()
    if a.mode == "catch":
        main_catch(a)
        return
    if a.mode == "mission":
        net_dir = f"net{a.net_side:g}".replace(".", "") if a.net_side not in (None, 1.0) else None
        if a.payload_mass:
            net_dir = os.path.join("payload", f"net{(a.net_side or 1.0):g}".replace(".", "") + f"_p{a.payload_mass:g}g")
        if a.thrust_scale != 1.0:
            net_dir = os.path.join(f"thrust{a.thrust_scale:g}", net_dir or "")
        a.out = a.out or (os.path.join(HERE, "out_official", net_dir, "missions") if net_dir else
                          os.path.join(HERE, "out_official", "missions" if a.first_study else "missions_tol50"))
        os.makedirs(a.out, exist_ok=True)
        tol = dict(MISSION_TOL_STRICT if a.first_study else MISSION_TOL)
        for key, v in (("pos_m", a.tol_pos), ("net_off_m", a.tol_net_off), ("sag_m", a.tol_sag)):
            if v is not None:
                tol[key] = v / 1000.0
        base = {"tol": tol, "rotor_check": a.rotor_check or a.first_study, "final_strict_s": a.final_strict,
                "planner_over": PLANNER_RELAXED_FIRST if a.first_study else None}
        # another net / formation: Params overrides, spacing, hold height, feedforward table, targets
        over = {k: v for k, v in (("L", a.net_side), ("m_net", a.net_mass if a.net_mass is not None else
                                                      (0.020 * a.net_side if a.net_side is not None else None)),
                                  ("d_takeoff", a.takeoff_spacing),
                                  ("payload_mass", a.payload_mass / 1000.0 if a.payload_mass else None),
                                  ("f_max", OfficialParams().f_max * a.thrust_scale if a.thrust_scale != 1.0 else None),
                                  ("max_tilt_deg", a.max_tilt)) if v is not None}
        if over:
            base["p_over"] = over
        if abs(a.spacing - 1.25) > 1e-12:
            base["spacing"] = a.spacing
        if a.hold_summary:
            with open(a.hold_summary) as f:
                base["h_dn"] = float(json.load(f)["hold"]["sag_below_drones_m"])
        if a.ff not in (None, "auto"):
            base["ff_path"] = a.ff
        if a.targets_from:
            base["targets_from"] = a.targets_from
        if a.planner_limits:
            pl = a.planner_limits
            lim = json.load(open(pl)) if os.path.isfile(pl) else json.loads(pl)
            base["planner_over"] = {**(base["planner_over"] or {}), **lim.get("planner_relaxed", lim)}
            if a.factor is None and lim.get("factor") is not None:          # the file may carry its factor
                a.factor = float(lim["factor"])
        if over or "spacing" in base:
            base["params_note"] = (f"net {over.get('L', 1.0):g} m / {1e3 * over.get('m_net', 0.020):g} g, spacing "
                                   f"{base.get('spacing', 1.25):g} m"
                                   + (f", payload {1e3 * over['payload_mass']:g} g" if over.get("payload_mass") else "")
                                   + (f", thrust x{a.thrust_scale:g}" if a.thrust_scale != 1.0 else "")
                                   + (f", tilt limit {a.max_tilt:g} deg" if a.max_tilt is not None else ""))
            print(f"configuration: {json.dumps({k: base.get(k) for k in ('p_over', 'spacing', 'h_dn', 'ff_path', 'targets_from')})}",
                  flush=True)
        print(f"mission mode, output: {a.out}, settle tolerances {tol}, net_below_rotors "
              f"{'fails a leg' if base['rotor_check'] else 'recorded only'}", flush=True)
        if a.characterise:
            run_characterise(a.out, a.jobs, base=base)
        elif a.study:
            kw = {}
            if a.factors:
                kw["factors"] = tuple(a.factors)
            if a.tol_sweep is not None:
                kw["tol_sweep_mm"] = tuple(a.tol_sweep)
            if a.compare_study:
                kw["compare_path"] = a.compare_study
            if a.no_strict:
                kw["strict"] = False
            if a.no_ref_rule:
                kw["ref_rule"] = False
            run_mission_study(a.out, a.jobs, base=base, **kw)
        else:
            f = a.factor if a.factor is not None else (
                (MISSION_FACTOR_FIRST if a.first_study else MISSION_FACTOR) if a.planner == "relaxed" else 1.0)
            met = run_mission({**MISSION_DEFAULT, "seed": a.seed, "n_targets": a.targets, "planner": a.planner,
                               "factor": f, "settle_mode": a.settle, **base}, out=a.out)
            print_mission(met)
            print("files: " + ", ".join(os.path.join(a.out, x) for x in met["files"]))
        return
    if a.mode == "transfer":
        a.out = a.out or os.path.join(HERE, "out_official", "transfer")
        os.makedirs(a.out, exist_ok=True)
        ff_path = os.path.join(HERE, "out_official", "sweep_summary.json") if a.ff in (None, "auto") else a.ff
        over = {"max_tilt_deg": a.max_tilt} if a.max_tilt is not None else None
        cfg = {**TRANSFER_DEFAULT, "L": a.distance, "heading": a.heading, "heading_deg": a.heading_deg,
               "speed": a.speed, "a_up": a.accel, "a_down": a.brake_accel, "D_c": a.contract, "c_T": a.contract_time,
               "s_off": a.respread_start, "s_T": a.respread_time, "dz": a.climb, "ff_path": ff_path, "p_over": over,
               "strategy": "reshape" if a.contract is not None else "rigid", "keep_linear": a.keep_linear_drag}
        print(f"transfer mode, output: {a.out}", flush=True)
        if a.sweep:
            run_transfer_sweep({"ff_path": ff_path}, a.out, a.jobs)
        else:
            met = run_transfer(cfg, out=a.out)
            print_transfer(met)
            print("files: " + ", ".join(os.path.join(a.out, f) for f in met["files"]))
        return
    if a.mode == "travel":
        if a.drone_model != "official":
            raise SystemExit("travel mode uses the official drone model only")
        a.out = a.out or os.path.join(HERE, "out_official", "travel")
        os.makedirs(a.out, exist_ok=True)
        ff_path = os.path.join(HERE, "out_official", "sweep_summary.json") if a.ff in (None, "auto") else a.ff
        if not os.path.exists(ff_path):
            raise SystemExit(f"feedforward table not found: {ff_path} (run --mode sweep first)")
        over = {k: v for k, v in (("d_takeoff", a.takeoff_spacing), ("clearance_warn", a.clearance_warn),
                                  ("guard_warn", a.guard_warn), ("yaw_deg", a.yaw_deg), ("hook_z", a.hook_z),
                                  ("m_drone", a.drone_mass)) if v is not None}
        p0 = OfficialParams()
        cfg = {**TRAVEL_DEFAULT, "speed": a.speed, "accel": a.accel, "heading": a.heading,
               "heading_deg": a.heading_deg, "t_cruise": a.cruise_time, "t_settle": a.settle_time,
               "spacing": a.spacing, "net_drag": a.net_drag != "off", "keep_linear": a.keep_linear_drag,
               "thread_d": a.thread_d, "net_mass": a.net_mass, "ff_path": ff_path, "p_over": over or None,
               "acc_ff": {"on": 1.0, "off": 0.0,
                          "net": (over.get("m_drone", p0.m_drone) + (a.net_mass or p0.m_net) / 4)
                          / over.get("m_drone", p0.m_drone)}[a.acc_ff]}
        print(f"travel mode, output: {a.out}", flush=True)
        if a.sweep:
            run_travel_sweep({k: cfg[k] for k in ("t_cruise", "t_settle", "spacing", "ff_path", "p_over",
                                                  "thread_d", "net_mass", "keep_linear", "net_drag", "acc_ff")},
                             a.out, a.jobs)
        else:
            met = run_travel(cfg, out=a.out)
            print_travel(met)
            print("files: " + ", ".join(os.path.join(a.out, f) for f in met["files"]))
        return
    if a.out is None and a.thrust_scale != 1.0:             # another thrust never writes over the existing results
        a.out = os.path.join(HERE, "out_official", f"thrust{a.thrust_scale:g}", a.mode)
    if a.out is None and a.payload_mass:                    # a payload never writes over the existing results
        a.out = os.path.join(HERE, "out_official", "payload",
                             f"net{(a.net_side or 1.0):g}".replace(".", "") + f"_p{a.payload_mass:g}g", a.mode)
    if a.out is None and a.net_side not in (None, 1.0):      # another net never writes over the 1 m results
        a.out = os.path.join(HERE, "out_official", f"net{a.net_side:g}".replace(".", ""), a.mode)
    if a.out is None:
        a.out = os.path.join(HERE, "out_official" if a.drone_model == "official" else "out")
    os.makedirs(a.out, exist_ok=True)
    p = OfficialParams() if a.drone_model == "official" else Params()
    if a.net_drag == "on":
        p.net_drag, p.net_drag_keep_linear = True, a.keep_linear_drag
    if a.thread_d is not None:
        p.thread_d = a.thread_d
    if a.net_side is not None:
        p.L = a.net_side
        p.m_net = 0.020 * a.net_side
    if a.net_mass is not None:
        p.m_net = a.net_mass
    if a.land_spacing is not None:
        p.d_land = a.land_spacing
    if a.payload_mass:
        p.payload_mass = a.payload_mass / 1000.0
        print(f"payload: {json.dumps(payload_info(p))}", flush=True)
    if a.drone_mass is not None:
        p.m_drone = a.drone_mass
    if a.thrust_scale != 1.0:
        p.f_max *= a.thrust_scale
        print(f"thrust scale {a.thrust_scale:g}: f_max {p.f_max:.3f} N per motor (minimum {getattr(p, 'f_min', 0.0):.4f} N "
              f"unchanged)", flush=True)
    if a.max_tilt is not None:
        p.max_tilt_deg = a.max_tilt
    if a.takeoff_spacing is not None:
        p.d_takeoff = a.takeoff_spacing
    if a.clearance_warn is not None:
        p.clearance_warn = a.clearance_warn
    if a.guard_warn is not None:
        p.guard_warn = a.guard_warn
    if a.yaw_deg is not None:
        p.yaw_deg = a.yaw_deg
    if a.hook_z is not None:
        p.hook_z = a.hook_z
    hk = hook_check(p)
    print(f"yaw {p.yaw_deg:g} deg, hook {p.hook_z * 1e3:+.1f} mm from the centre of mass "
          f"({hk['hook_above_feet_mm']:+.1f} mm relative to the guard feet: {hk['note']})", flush=True)
    if hk["needs_hardware_change"]:
        print("warning: this configuration needs a hardware change (longer landing legs)", flush=True)
    # at rest the hook must be within string reach of its corner knot (horizontal check)
    gap = np.sqrt(2) * abs(p.d_takeoff / 2 - 0.98 * p.L / 2)
    if gap > p.string:
        print(f"warning: takeoff spacing {p.d_takeoff:.2f} m puts each drone {gap * 1e3:.0f} mm from its net corner, "
              f"more than the {p.string * 1e3:.0f} mm string: the strings start overstretched", flush=True)
    print(f"drone model: {a.drone_model} (mass {p.m_drone * 1e3:.1f} g, {p.f_max / G * 1e3:.1f} gf per motor), "
          f"output: {a.out}", flush=True)
    if a.mode == "sweep":
        rows, wall = run_sweep(p, a.sweep_spacings or [0.90, 1.00, 1.10, 1.16, 1.20, 1.23, 1.25, 1.27, 1.29, 1.31, 1.33],
                               a.out, clearance=a.sweep_spacings is not None)
        print(f"sweep finished in {wall:.1f} s wall time")
        print(f"{'Dref':>5} {'D':>6} {'span':>6} {'sag_net':>8} {'sag_drn':>8} {'theta':>6} {'tilt':>6} {'tilt_p':>6} "
              f"{'T[gf]':>7} {'Tp[gf]':>7} {'mot_max':>7} {'err_mm':>7} {'disc':>6} {'guard':>6}")
        for r in rows:
            print(f"{r['spacing_ref_m']:5.2f} {r['spacing_m']:6.3f} {r['net_corner_span_m']:6.3f} {r['sag_of_net_m']:8.3f} {r['sag_below_drones_m']:8.3f} "
                  f"{r['string_angle_deg']:6.1f} {r['tilt_deg']:6.1f} {r['tilt_pred_deg']:6.1f} "
                  f"{r['tension_N'] / G * 1e3:7.1f} {r['tension_pred_N'] / G * 1e3:7.1f} "
                  f"{r['motor_frac_max']:7.2f} {r['pos_err_mean_mm']:7.1f} {r.get('disc_min_mm', float('nan')):6.1f} "
                  f"{r.get('guard_min_mm', float('nan')):6.1f}")
    else:
        ff_path = os.path.join(a.out, "sweep_summary.json") if a.ff == "auto" else a.ff
        if ff_path and not os.path.exists(ff_path):
            raise SystemExit(f"feedforward table not found: {ff_path} (run --mode sweep first)")
        ff = load_ff(p, ff_path, spacing=a.spacing) if ff_path else None
        if p.net_drag:
            print(f"net drag on: {json.dumps(aero_info(p))}", flush=True)
        s = run_hold(p, a.spacing, a.out, a.gif, ff=ff,
                     tag=("hold_ff" if ff else "hold") + ("_netdrag" if p.net_drag else ""))
        for w in ("first_2s_after_spread", "hold", "after_gust"):
            r = s[w]
            print(f"{w:>22}: D={r['spacing_m']:.4f} m  err_mean={r['pos_err_mean_mm']:.1f} mm  rms={r['pos_rms_mm']}  "
                  f"sag={r['sag_of_net_m']:.3f} m (std {r['sag_std_mm']:.1f} mm)  tilt={r['tilt_deg']:.1f}/{r['tilt_pred_deg']:.1f} deg  "
                  f"T={r['tension_N'] / G * 1e3:.1f} gf  motor_max={r['motor_frac_max']:.2f}")
        print("gust:", s["gust"])
        print("landing:", json.dumps(s["landing"], indent=1))
        tk = s["takeoff"]
        print(f"takeoff at {p.d_takeoff:.2f} m ({tk['window_s'][0]}-{tk['window_s'][1]} s): pos_err_max={tk['pos_err_max_mm']:.1f} mm  "
              f"tilt_max={tk['tilt_max_deg']:.1f} deg  motor_max={tk['motor_frac_max']:.2f}  T_max={tk['tension_max_gf']:.1f} gf  "
              f"own string slack while airborne [s]={tk['own_string_slack_while_airborne_s']}")
        at = s["attitude_hold"]
        print(f"attitude in hold ({at['window_s'][0]:.0f}-{at['window_s'][1]:.0f} s, worst drone): tilt std {at['tilt_std_deg']:.3f} deg, "
              f"content above {at['f_cut_hz']:.0f} Hz {at['tilt_rms_above_3hz_deg']:.3f} deg rms")
        print_clearance(p, s["clearance"])


if __name__ == "__main__":
    main()
