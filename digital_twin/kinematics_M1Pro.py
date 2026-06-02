"""
Dobot M1Pro SCARA Robot — Kinematics Engine
=============================================
Standard DH parameter model for 4-axis SCARA collaborative robot.
J1/J2/J4: revolute joints (degrees)
J3: prismatic lift column at base, coaxial with J1 — raises entire arm plane

Architecture:
  J3 (base lift) → raises arm plane vertically
  J1 (shoulder)  → rotates upper arm in horizontal plane
  J2 (elbow)     → rotates forearm in horizontal plane
  J4 (wrist)     → rotates tool about vertical axis

Coordinate convention (right-handed):
  - X: forward (from base toward work area)
  - Y: left
  - Z: up (vertical, positive = away from table)
  - R: rotation about Z axis at end-effector (degrees, CCW positive)

Joint variable mapping (internal → DH):
  q[0] = J1  → θ1 (deg)  |  q[1] = J2  → θ2 (deg)
  q[2] = J3  → d3 (mm), lift displacement  |  q[3] = J4 → θ4 (deg)

Cartesian pose:
  [X, Y, Z, R]  — X,Y,Z in mm, R in degrees
"""

import math
import json
import os
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict

# ============================================================
# Paths
# ============================================================
_DIR = os.path.dirname(os.path.abspath(__file__))
_OFFSETS_PATH = os.path.join(_DIR, "data", "offsets", "dobot_joint_offsets.json")
_STATE_PATH = os.path.join(_DIR, "data", "runtime", "dobot_state.json")


# ============================================================
# Load offsets from JSON
# ============================================================
def _load_offsets() -> Dict:
    if os.path.exists(_OFFSETS_PATH):
        with open(_OFFSETS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


_offsets = _load_offsets()


# ============================================================
# Physical parameters — Dobot M1Pro (public + estimated)
# ============================================================

# Link lengths (mm).  a1 + a2 ≈ 400mm max reach.
A1 = 250.0          # upper arm length (J1→J2), mm
A2 = 150.0          # forearm length (J2→J4), mm

# Z-axis lift column (J3 coaxial with J1 at base)
# The entire arm plane rises with d3.
# z_tcp = D1 + d3 - D4
D1 = 85.0           # fixed base column height (table → arm plane at d3=0), mm
D4 = 80.0           # tool flange offset (J4 → TCP), mm

# Z-axis lead screw (map protocol J3 degrees → linear d3 mm)
# d3 = D3_BASE + (J3_protocol / 360.0) * SCREW_LEAD  (positive slope)
SCREW_LEAD = 10.0   # mm per full revolution (360°)

# d3 range: 150 ~ 450 mm (300mm stroke, 15 revs each way at 10mm/rev)
D3_MIN = 150.0
D3_BASE = 300.0     # d3 at J3_protocol = 0° (mid-stroke)
D3_MAX = 450.0

# Joint limits
J1_MIN, J1_MAX = -85.0, 85.0        # degrees
J2_MIN, J2_MAX = -130.0, 130.0      # degrees
J3_DEG_MIN, J3_DEG_MAX = -5400.0, 5400.0   # motor degrees (15 revs each way)
Z_MIN, Z_MAX = 155.0, 455.0         # mm (TCP Z range: z = D1 + d3 - D4)
J4_MIN, J4_MAX = -360.0, 360.0      # degrees

# Cartesian workspace bounds (mm)
X_MIN, X_MAX = -400.0, 400.0
Y_MIN, Y_MAX = -400.0, 400.0

# ============================================================
# Helper functions
# ============================================================

def deg2rad(deg: float) -> float:
    return deg * math.pi / 180.0

def rad2deg(rad: float) -> float:
    return rad * 180.0 / math.pi

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))

def wrap_180(deg: float) -> float:
    """Wrap angle to [-180, 180)."""
    deg = deg % 360.0
    if deg >= 180.0:
        deg -= 360.0
    return deg


# ============================================================
# J3 ↔ d3 conversion
# ============================================================

def j3_deg_to_d3(j3_deg: float) -> float:
    """Convert protocol J3 (degrees) to d3 (mm).
    Positive slope: larger motor angle → larger d3 → higher arm plane."""
    return D3_BASE + (j3_deg / 360.0) * SCREW_LEAD

def d3_to_j3_deg(d3_mm: float) -> float:
    """Convert d3 (mm) back to protocol J3 (degrees)."""
    return (d3_mm - D3_BASE) * 360.0 / SCREW_LEAD


# ============================================================
# Standard DH transformation
# ============================================================

def dh_transform(theta_deg: float, d: float, a: float, alpha_deg: float) -> np.ndarray:
    """
    Standard DH 4×4 homogeneous matrix.
    T = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)
    """
    t = deg2rad(theta_deg)
    al = deg2rad(alpha_deg)
    ct, st = math.cos(t), math.sin(t)
    ca, sa = math.cos(al), math.sin(al)
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0,        sa,      ca,      d],
        [0,         0,       0,      1],
    ], dtype=float)


# ============================================================
# Forward Kinematics
# ============================================================

@dataclass
class JointState:
    """Joint state in internal representation."""
    j1_deg: float   # base rotation
    j2_deg: float   # elbow rotation
    d3_mm: float    # Z-axis displacement
    j4_deg: float   # end-effector rotation

    @classmethod
    def from_protocol(cls, j1: float, j2: float, j3_deg: float, j4: float):
        """Build from protocol joint values (J3 in degrees)."""
        return cls(j1_deg=j1, j2_deg=j2,
                   d3_mm=j3_deg_to_d3(j3_deg), j4_deg=j4)

    def to_protocol(self) -> Tuple[float, float, float, float]:
        """Export as protocol joint values (J3 in degrees)."""
        return (self.j1_deg, self.j2_deg, d3_to_j3_deg(self.d3_mm), self.j4_deg)

    def to_array(self) -> np.ndarray:
        return np.array([self.j1_deg, self.j2_deg, self.d3_mm, self.j4_deg])


@dataclass
class CartesianPose:
    """End-effector pose in Cartesian space."""
    x: float    # mm
    y: float    # mm
    z: float    # mm (height from table)
    r: float    # degrees (rotation about Z)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x, self.y, self.z, self.r)


def forward_kinematics(q: JointState) -> Tuple[CartesianPose, List[np.ndarray]]:
    """
    Compute end-effector pose and all link transforms.

    DH table (SCARA):
        i   θ_i       d_i    a_i    α_i
        1   θ1*       d1     a1     0°
        2   θ2*       0      a2     180°
        3   0          d3*    0      0°
        4   θ4*       d4     0      0°

    Returns (CartesianPose, [T01, T02, T03, T04])
    """
    T01 = dh_transform(q.j1_deg, D1, A1, 0)
    T12 = dh_transform(q.j2_deg, 0, A2, 180)
    T23 = dh_transform(0, q.d3_mm, 0, 0)
    T34 = dh_transform(q.j4_deg, D4, 0, 0)

    T02 = T01 @ T12
    T03 = T02 @ T23
    T04 = T03 @ T34

    x, y = T04[0, 3], T04[1, 3]
    # z = arm plane height - tool offset
    #   = (D1 + d3) - D4 = D1 + d3 - D4
    z = D1 + q.d3_mm - D4

    # Orientation: total Z-rotation = θ1 + θ2 + θ4
    r = q.j1_deg + q.j2_deg + q.j4_deg
    r = wrap_180(r)

    return CartesianPose(x=x, y=y, z=z, r=r), [T01, T02, T03, T04]


def fk_compact(j1: float, j2: float, d3_mm: float, j4: float) -> CartesianPose:
    """Convenience: forward kinematics without T matrices."""
    q = JointState(j1_deg=j1, j2_deg=j2, d3_mm=d3_mm, j4_deg=j4)
    pose, _ = forward_kinematics(q)
    return pose


# ============================================================
# Inverse Kinematics
# ============================================================

@dataclass
class IKSolution:
    """Single IK solution branch."""
    j1_deg: float
    j2_deg: float
    d3_mm: float
    j4_deg: float
    elbow_up: bool
    valid: bool = True
    reason: str = ""

    def to_joint_state(self) -> JointState:
        return JointState(j1_deg=self.j1_deg, j2_deg=self.j2_deg,
                          d3_mm=self.d3_mm, j4_deg=self.j4_deg)


def inverse_kinematics(x: float, y: float, z: float, r: float,
                       elbow_up: bool = True,
                       near_joints: Optional[JointState] = None) -> IKSolution:
    """
    Analytical inverse kinematics for 4-axis SCARA.

    Returns the elbow_up solution by default. If near_joints is given,
    picks the branch closest to those joint values.
    """
    # --- Z-axis (J3 / d3) ---
    # z = D1 + d3 - D4  →  d3 = z - D1 + D4
    d3 = z - D1 + D4
    if not (D3_MIN <= d3 <= D3_MAX):
        return IKSolution(0, 0, d3, 0, elbow_up, valid=False,
                          reason=f"d3={d3:.1f} out of range [{D3_MIN}, {D3_MAX}]")

    # --- θ2 ---
    cos_q2 = (x*x + y*y - A1*A1 - A2*A2) / (2.0 * A1 * A2)
    if cos_q2 < -1.0 or cos_q2 > 1.0:
        return IKSolution(0, 0, d3, 0, elbow_up, valid=False,
                          reason=f"Target ({x:.1f},{y:.1f}) unreachable (|cos_θ2|>1)")

    q2_up = math.acos(cos_q2)          # elbow up   (positive)
    q2_down = -math.acos(cos_q2)       # elbow down (negative)

    # --- θ1 ---
    def solve_q1(q2: float) -> float:
        return math.atan2(y, x) - math.atan2(A2 * math.sin(q2), A1 + A2 * math.cos(q2))

    q1_up = solve_q1(q2_up)
    q1_down = solve_q1(q2_down)

    # --- θ4 ---
    q4_up = wrap_180(r - rad2deg(q1_up) - rad2deg(q2_up))
    q4_down = wrap_180(r - rad2deg(q1_down) - rad2deg(q2_down))

    q1_up_d = rad2deg(q1_up)
    q1_down_d = rad2deg(q1_down)
    q2_up_d = rad2deg(q2_up)
    q2_down_d = rad2deg(q2_down)

    # --- Choose branch ---
    if near_joints is not None:
        # Pick branch closest to near_joints
        def dist(q1d, q2d, q4d):
            d1 = (wrap_180(q1d - near_joints.j1_deg))**2
            d2 = (wrap_180(q2d - near_joints.j2_deg))**2
            d4 = (wrap_180(q4d - near_joints.j4_deg))**2
            return d1 + d2 + d4 + (d3 - near_joints.d3_mm)**2 * 0.01

        if dist(q1_up_d, q2_up_d, q4_up) <= dist(q1_down_d, q2_down_d, q4_down):
            q1_f, q2_f, q4_f, eu = q1_up_d, q2_up_d, q4_up, True
        else:
            q1_f, q2_f, q4_f, eu = q1_down_d, q2_down_d, q4_down, False
    elif elbow_up:
        q1_f, q2_f, q4_f, eu = q1_up_d, q2_up_d, q4_up, True
    else:
        q1_f, q2_f, q4_f, eu = q1_down_d, q2_down_d, q4_down, False

    # --- Joint limit check ---
    if not (J1_MIN <= q1_f <= J1_MAX):
        return IKSolution(q1_f, q2_f, d3, q4_f, eu, valid=False,
                          reason=f"J1={q1_f:.1f}° out of [{J1_MIN},{J1_MAX}]")
    if not (J2_MIN <= q2_f <= J2_MAX):
        return IKSolution(q1_f, q2_f, d3, q4_f, eu, valid=False,
                          reason=f"J2={q2_f:.1f}° out of [{J2_MIN},{J2_MAX}]")
    if not (J4_MIN <= q4_f <= J4_MAX):
        return IKSolution(q1_f, q2_f, d3, q4_f, eu, valid=False,
                          reason=f"J4={q4_f:.1f}° out of [{J4_MIN},{J4_MAX}]")

    return IKSolution(j1_deg=q1_f, j2_deg=q2_f, d3_mm=d3, j4_deg=q4_f,
                      elbow_up=eu, valid=True)


# ============================================================
# Jacobian
# ============================================================

def compute_jacobian(q: JointState) -> np.ndarray:
    """
    Compute 4×4 geometric Jacobian for SCARA.
    J maps joint velocity → Cartesian velocity: [vx, vy, vz, ωz]^T = J · [θ̇1, θ̇2, ḋ3, θ̇4]^T

    For SCARA with base lift (J3 raises entire arm plane):
        vx = -a1 s1 θ̇1 - a2 s12 (θ̇1 + θ̇2)
        vy =  a1 c1 θ̇1 + a2 c12 (θ̇1 + θ̇2)
        vz = +ḋ3   (arm plane rises with d3, z = D1 + d3 - D4)
        ωz = θ̇1 + θ̇2 + θ̇4
    """
    t1 = deg2rad(q.j1_deg)
    t2 = deg2rad(q.j2_deg)
    c1, s1 = math.cos(t1), math.sin(t1)
    c12 = math.cos(t1 + t2)
    s12 = math.sin(t1 + t2)

    J = np.zeros((4, 4))
    J[0, 0] = -A1 * s1 - A2 * s12    # ∂x/∂θ1
    J[0, 1] = -A2 * s12              # ∂x/∂θ2
    J[0, 2] = 0                       # ∂x/∂d3
    J[0, 3] = 0                       # ∂x/∂θ4

    J[1, 0] = A1 * c1 + A2 * c12     # ∂y/∂θ1
    J[1, 1] = A2 * c12               # ∂y/∂θ2
    J[1, 2] = 0                       # ∂y/∂d3
    J[1, 3] = 0                       # ∂y/∂θ4

    J[2, 0] = 0
    J[2, 1] = 0
    J[2, 2] = +1                      # ∂z/∂d3  (z = D1 + d3 - D4)
    J[2, 3] = 0

    J[3, 0] = 1                       # ∂ωz/∂θ1
    J[3, 1] = 1                       # ∂ωz/∂θ2
    J[3, 2] = 0                       # ∂ωz/∂d3
    J[3, 3] = 1                       # ∂ωz/∂θ4

    return J


# ============================================================
# Workspace analysis
# ============================================================

def compute_workspace_boundary(n: int = 360) -> Tuple[List[float], List[float]]:
    """Compute the reachable XY workspace boundary (outer ring).

    Returns (xs, ys) for polar plot of max reach at each angle.
    """
    xs, ys = [], []
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        # At each approach direction, find max extension
        # r_max given J1 ∈ [±J1_MAX], J2 ∈ [±J2_MAX]
        # The reachable radius depends on the approach angle relative to J1
        # Full extension when both arms aligned: r_max = A1 + A2 = 400mm
        xs.append((A1 + A2) * math.cos(angle))
        ys.append((A1 + A2) * math.sin(angle))
    return xs, ys


def compute_workspace_inner(n: int = 360) -> Tuple[List[float], List[float]]:
    """Compute inner deadzone (minimum reach when fully folded).

    Minimum reach = |A1 - A2| = 100mm
    """
    xs, ys = [], []
    for i in range(n):
        angle = 2.0 * math.pi * i / n
        r = abs(A1 - A2)
        xs.append(r * math.cos(angle))
        ys.append(r * math.sin(angle))
    return xs, ys


# ============================================================
# Runtime state persistence
# ============================================================

def load_state() -> JointState:
    """Load runtime state from data/runtime/dobot_state.json.

    Falls back to home position if the file does not exist or is invalid.
    """
    if os.path.exists(_STATE_PATH):
        try:
            with open(_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            joint = data.get("joint", {})
            return JointState(
                j1_deg=joint.get("j1", 0.0),
                j2_deg=joint.get("j2", 0.0),
                d3_mm=joint.get("d3_mm", D3_BASE),
                j4_deg=joint.get("j4", 0.0),
            )
        except (json.JSONDecodeError, KeyError):
            pass
    return JointState(j1_deg=0, j2_deg=0, d3_mm=D3_BASE, j4_deg=0)


def save_state(state: JointState) -> None:
    """Write current state to data/runtime/dobot_state.json."""
    os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
    pose, _ = forward_kinematics(state)
    from datetime import datetime, timezone
    data = {
        "joint": {
            "j1": round(state.j1_deg, 2),
            "j2": round(state.j2_deg, 2),
            "j3_deg": round(d3_to_j3_deg(state.d3_mm), 2),
            "j4": round(state.j4_deg, 2),
            "d3_mm": round(state.d3_mm, 3),
        },
        "tcp": {
            "x": round(pose.x, 3),
            "y": round(pose.y, 3),
            "z": round(pose.z, 3),
            "r": round(pose.r, 2),
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":
    # Home position (J3_deg=0 → d3=D3_BASE=320 → z=325mm)
    q_home = JointState(j1_deg=0, j2_deg=0, d3_mm=D3_BASE, j4_deg=0)
    pose, _ = forward_kinematics(q_home)
    print(f"Home: J=[0,0,d3={D3_BASE:.0f},0] → Pose=({pose.x:.1f}, {pose.y:.1f}, {pose.z:.1f}, {pose.r:.1f}°)")
    print(f"  Expected: (400, 0, 305, 0)")

    # Mid config
    q_mid = JointState(j1_deg=30, j2_deg=-60, d3_mm=300, j4_deg=90)
    pose_mid, _ = forward_kinematics(q_mid)
    print(f"Mid:   J=[30,-60,d3=300,90] → Pose=({pose_mid.x:.1f}, {pose_mid.y:.1f}, {pose_mid.z:.1f}, {pose_mid.r:.1f}°)")

    # IK test
    sol = inverse_kinematics(200, 200, 350, 0, elbow_up=True)
    print(f"IK:    Pose=(200,200,350,0) → J=({sol.j1_deg:.1f}, {sol.j2_deg:.1f}, d3={sol.d3_mm:.1f}, {sol.j4_deg:.1f}°) valid={sol.valid}")

    if sol.valid:
        q2 = sol.to_joint_state()
        p2, _ = forward_kinematics(q2)
        print(f"IK verify: J→Pose=({p2.x:.1f}, {p2.y:.1f}, {p2.z:.1f}, {p2.r:.1f}°)")
        err = math.sqrt((p2.x-200)**2 + (p2.y-200)**2 + (p2.z-350)**2)
        print(f"  Error={err:.3f}mm")

    print(f"\nJ1: [{J1_MIN}, {J1_MAX}]°  J2: [{J2_MIN}, {J2_MAX}]°")
    print(f"d3: [{D3_MIN}, {D3_MAX}]mm  Z_tcp: [{Z_MIN}, {Z_MAX}]mm  J4: [{J4_MIN}, {J4_MAX}]°")
    print(f"SCREW_LEAD={SCREW_LEAD}mm/rev  D3_BASE={D3_BASE}mm  D1={D1}mm")
    print(f"\nArchitecture: J3 lift column (coaxial with J1) → entire arm plane moves vertically")
    print(f"  z_tcp = D1 + d3 - D4 = {D1} + d3 - {D4}")
    print(f"  d3 = D3_BASE + (J3_deg/360)*SCREW_LEAD = {D3_BASE} + J3_deg/360*{SCREW_LEAD}")
    print(f"  J3=-5400°→d3={D3_BASE + (-5400/360)*SCREW_LEAD:.0f}mm  J3=0°→d3={D3_BASE:.0f}mm  J3=+5400°→d3={D3_BASE + (5400/360)*SCREW_LEAD:.0f}mm")
