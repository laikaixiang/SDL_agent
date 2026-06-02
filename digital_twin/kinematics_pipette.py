"""
XYZZ+dual ADP Pipette Arm — Kinematics Engine
===============================================
4-axis linear gantry system: X, Y, Z1, Z2 (all prismatic).

Coordinate convention (after Rx(-90°) pivot, same as Dobot):
  - X : horizontal, mid beam sliding on cross beam (World X)
  - Y : horizontal, cross beam moving along frame (World Z)
  - Z1: vertical (UP), ADP1 lift (World Y)
  - Z2: vertical (UP), ADP2 lift (World Y)

This matches the Dobot convention where Z = vertical up.

Architecture:
  The pipette arm has a fixed frame (group1), a cross beam (group2)
  that slides along the frame in Y, a mid beam (group3) that slides
  on the cross beam in X, and two independent ADP modules (group4/5)
  that move vertically (Z1, Z2).

All axes are pure prismatic — FK is identity, IK is limit clamping.
"""

import math
import json
import os
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Dict


# ============================================================
# Paths
# ============================================================
_DIR = os.path.dirname(os.path.abspath(__file__))
_PARAMS_PATH = os.path.join(_DIR, "data", "offsets", "pipette_kinematic_params.json")
_STATE_PATH = os.path.join(_DIR, "data", "runtime", "pipette_state.json")


# ============================================================
# Load parameters from JSON
# ============================================================
def _load_params() -> Dict:
    if os.path.exists(_PARAMS_PATH):
        with open(_PARAMS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


_params = _load_params()
_joints = _params.get("joints", {})
_adp = _params.get("adp_spacing_mm", 30.2)


# ============================================================
# Physical parameters — from pipette_kinematic_params.json (with hardcoded fallback)
# ============================================================

# X axis: mid beam sliding on cross beam (CAD X, World X)
#   Cross beam X: [4.6, 424.4], mid beam half-width: 39.5mm
X_MIN = _joints.get("X", {}).get("mechanical_range", [44.1, 384.9])[0]
X_MAX = _joints.get("X", {}).get("mechanical_range", [44.1, 384.9])[1]
X_REF = _joints.get("X", {}).get("reference_mm", 300.3)
X_STROKE = X_MAX - X_MIN

# Y axis: cross beam sliding along frame (CAD Y, World Z)
#   Frame Y: [3.1, 423.1], cross beam half-depth: 56.0mm
Y_MIN = _joints.get("Z", {}).get("mechanical_range", [59.1, 367.1])[0]
Y_MAX = _joints.get("Z", {}).get("mechanical_range", [59.1, 367.1])[1]
Y_REF = _joints.get("Z", {}).get("reference_mm", 213.1)
Y_REF_STL = 416.1      # STL assembly position (cross beam exceeds frame by 49mm)
Y_STROKE = Y_MAX - Y_MIN

# Z1/Z2 axis: ADP vertical lift (CAD Z, World Y = UP)
#   Mid beam Z: [59.3, 195.4], ADP half-height: 34.8mm
Z_MIN = _joints.get("Y1", {}).get("mechanical_range", [94.0, 160.7])[0]
Z_MAX = _joints.get("Y1", {}).get("mechanical_range", [94.0, 160.7])[1]
Z_REF = _joints.get("Y1", {}).get("reference_mm", 123.1)
Z_STROKE = Z_MAX - Z_MIN

# ADP spacing in X direction
ADP_SPACING_X = _adp

# Tip offset from ADP center
# Tip center: (284.8, 156.1, 112.3), ADP1 center: (284.6, 298.1, 123.1)
# Offset: tip is below ADP1 by 123.1-112.3=10.8mm in Z, and 298.1-156.1=142mm in Y
TIP_Z_OFFSET = 10.8     # mm, tip below ADP center in Z (CAD)
TIP_Y_OFFSET = 142.0    # mm, tip offset from ADP center in Y (CAD)

# ============================================================
# Helper functions
# ============================================================

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ============================================================
# Data classes
# ============================================================

@dataclass
class PipetteState:
    """4-axis joint state in logical coordinates.

    All values in mm, measured from the Rx(-90°) pivot origin.
    X, Y are horizontal; Z1, Z2 are vertical (same as Dobot Z).
    """
    x:  float   # mid beam X position (mm)
    y:  float   # cross beam Y position (mm)
    z1: float   # ADP1 Z height (mm)
    z2: float   # ADP2 Z height (mm)

    def to_dict(self) -> Dict[str, float]:
        return {'x': round(self.x, 2), 'y': round(self.y, 2),
                'z1': round(self.z1, 2), 'z2': round(self.z2, 2)}

    def to_array(self) -> List[float]:
        return [self.x, self.y, self.z1, self.z2]

    @classmethod
    def home(cls):
        """Return home position (mechanical center of all axes)."""
        return cls(x=X_REF, y=Y_REF, z1=Z_REF, z2=Z_REF)

    @classmethod
    def stl_assembly(cls):
        """Return STL assembly reference position."""
        return cls(x=X_REF, y=Y_REF_STL, z1=Z_REF, z2=Z_REF)


@dataclass
class PipetteTip:
    """Position of a pipette tip in CAD coordinates (pivot-local)."""
    x: float    # CAD X
    y: float    # CAD Y
    z: float    # CAD Z (vertical)

    def to_world(self) -> Tuple[float, float, float]:
        """Convert CAD coordinates to Three.js world coordinates.

        After Rx(-90°): CAD_X→World_X, CAD_Y→World_Z, CAD_Z→World_Y.
        """
        return (self.x, self.z, -self.y)

    def to_dict(self) -> Dict[str, float]:
        return {'x': round(self.x, 2), 'y': round(self.y, 2),
                'z': round(self.z, 2)}


@dataclass
class PipettePose:
    """End-effector pose: positions of both ADP tips."""
    tip1: PipetteTip   # ADP1 tip position
    tip2: PipetteTip   # ADP2 tip position

    def to_dict(self) -> Dict:
        return {
            'tip1': self.tip1.to_dict(),
            'tip2': self.tip2.to_dict(),
            'tip1_world': [round(v, 2) for v in self.tip1.to_world()],
            'tip2_world': [round(v, 2) for v in self.tip2.to_world()],
        }


@dataclass
class PipetteSolution:
    """IK result: clamped axis values + validity flags."""
    x: float
    y: float
    z1: float
    z2: float
    valid: bool = True
    limit_violations: List[str] = field(default_factory=list)

    def to_state(self) -> PipetteState:
        return PipetteState(x=self.x, y=self.y, z1=self.z1, z2=self.z2)

    def to_dict(self) -> Dict:
        return {
            'x': round(self.x, 2), 'y': round(self.y, 2),
            'z1': round(self.z1, 2), 'z2': round(self.z2, 2),
            'valid': self.valid,
            'limit_violations': self.limit_violations,
        }


# ============================================================
# Forward Kinematics
# ============================================================

def forward_kinematics(q: PipetteState) -> PipettePose:
    """Compute tip positions from axis values.

    In CAD coordinates (pivot-local space):
      tip1 = (ADP1_center_X - TIP_Y_OFFSET_in_reverse...)

    Tip position in CAD:
      tip_X = ADP_center_X (with Y-axis offset adjustment)
      tip_Y = ADP_center_Y - TIP_Y_OFFSET (tip is offset from ADP center in Y)
      tip_Z = Z - TIP_Z_OFFSET (tip is below ADP center)

    ADP1 center in CAD at reference: X=284.6, Y=298.1, Z=123.1
    Tip center at reference:         X=284.8, Y=156.1, Z=112.3

    So tip is behind ADP in CAD Y by ~142mm, and below ADP in CAD Z by ~10.8mm.
    The tip's X tracks the ADP's X (negligible 0.2mm difference).
    """
    # ADP1 center in CAD
    adp1_cx = 284.6 + (q.x - X_REF)      # ADP1 X tracks X axis
    adp1_cy = 298.1 - (q.y - Y_REF_STL)   # ADP1 Y tracks Y axis (note: Y is CAD Y direction)
    adp1_cz = q.z1                         # ADP1 Z = Z1 axis value

    # ADP2 center in CAD  (same Y, same Z, offset X by ADP_SPACING)
    adp2_cx = adp1_cx + ADP_SPACING_X
    adp2_cy = adp1_cy
    adp2_cz = q.z2

    # Tip positions (offset from ADP centers)
    tip1 = PipetteTip(
        x=adp1_cx - 0.2,         # negligible X offset
        y=adp1_cy - 142.0,       # tip is ~142mm behind ADP in CAD Y
        z=adp1_cz - TIP_Z_OFFSET  # tip is ~10.8mm below ADP in CAD Z
    )
    tip2 = PipetteTip(
        x=adp2_cx - 0.2,
        y=adp2_cy - 142.0,
        z=adp2_cz - TIP_Z_OFFSET
    )
    return PipettePose(tip1=tip1, tip2=tip2)


def fk_compact(x: float, y: float, z1: float, z2: float) -> PipettePose:
    """Convenience: FK without explicit state object."""
    return forward_kinematics(PipetteState(x=x, y=y, z1=z1, z2=z2))


# ============================================================
# Inverse Kinematics
# ============================================================

def inverse_kinematics(x: float, y: float, z1: float, z2: float) -> PipetteSolution:
    """Constrain target axis values to mechanical limits.

    Since all axes are independent prismatic joints, IK is just
    clamping to limits. Returns the clamped values plus any
    limit violations.
    """
    violations = []
    x_clamped = clamp(x, X_MIN, X_MAX)
    y_clamped = clamp(y, Y_MIN, Y_MAX)
    z1_clamped = clamp(z1, Z_MIN, Z_MAX)
    z2_clamped = clamp(z2, Z_MIN, Z_MAX)

    if x != x_clamped:
        violations.append(f'X={x:.1f} clamped to {x_clamped:.1f} (limit [{X_MIN:.0f}, {X_MAX:.0f}])')
    if y != y_clamped:
        violations.append(f'Y={y:.1f} clamped to {y_clamped:.1f} (limit [{Y_MIN:.0f}, {Y_MAX:.0f}])')
    if z1 != z1_clamped:
        violations.append(f'Z1={z1:.1f} clamped to {z1_clamped:.1f} (limit [{Z_MIN:.0f}, {Z_MAX:.0f}])')
    if z2 != z2_clamped:
        violations.append(f'Z2={z2:.1f} clamped to {z2_clamped:.1f} (limit [{Z_MIN:.0f}, {Z_MAX:.0f}])')

    return PipetteSolution(
        x=x_clamped, y=y_clamped, z1=z1_clamped, z2=z2_clamped,
        valid=len(violations) == 0,
        limit_violations=violations,
    )


def ik_compact(x: float, y: float, z1: float, z2: float) -> PipetteSolution:
    """Convenience: IK without fancy options."""
    return inverse_kinematics(x, y, z1, z2)


# ============================================================
# Workspace
# ============================================================

def compute_workspace_bounds() -> Dict:
    """Return the 3D workspace boundary.

    The workspace is the rectangular prism reachable by both tips.
    In CAD coordinates: X range + Y range + Z range.
    """
    return {
        'x': [X_MIN, X_MAX],
        'y': [Y_MIN, Y_MAX],
        'z': [Z_MIN, Z_MAX],
        'volume_mm3': round(X_STROKE * Y_STROKE * Z_STROKE, 1),
        'tip1_reference': {
            'x': 284.6, 'y': 298.1 - 142.0, 'z': Z_REF - TIP_Z_OFFSET,
        },
        'tip2_reference': {
            'x': 284.6 + ADP_SPACING_X, 'y': 298.1 - 142.0, 'z': Z_REF - TIP_Z_OFFSET,
        },
    }


def get_joint_limits() -> Dict:
    """Return all joint limits and reference positions."""
    return {
        'x':  {'type': 'prismatic', 'min': X_MIN, 'max': X_MAX,
               'ref': X_REF, 'stroke': X_STROKE, 'direction': 'horizontal (World X)'},
        'y':  {'type': 'prismatic', 'min': Y_MIN, 'max': Y_MAX,
               'ref': Y_REF, 'ref_stl': Y_REF_STL, 'stroke': Y_STROKE,
               'direction': 'horizontal (World Z)'},
        'z1': {'type': 'prismatic', 'min': Z_MIN, 'max': Z_MAX,
               'ref': Z_REF, 'stroke': Z_STROKE, 'direction': 'vertical (World Y = UP)'},
        'z2': {'type': 'prismatic', 'min': Z_MIN, 'max': Z_MAX,
               'ref': Z_REF, 'stroke': Z_STROKE, 'direction': 'vertical (World Y = UP)'},
        'adp_spacing_x_mm': ADP_SPACING_X,
    }


# ============================================================
# Runtime state persistence
# ============================================================

def load_state() -> PipetteState:
    """Load runtime state from data/runtime/pipette_state.json.

    Falls back to home position if the file does not exist or is invalid.
    """
    if os.path.exists(_STATE_PATH):
        try:
            with open(_STATE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            axis = data.get("axis", {})
            return PipetteState(
                x=axis.get("x", X_REF),
                y=axis.get("y", Y_REF),
                z1=axis.get("z1", Z_REF),
                z2=axis.get("z2", Z_REF),
            )
        except (json.JSONDecodeError, KeyError):
            pass
    return PipetteState.home()


def save_state(state: PipetteState) -> None:
    """Write current state to data/runtime/pipette_state.json."""
    os.makedirs(os.path.dirname(_STATE_PATH), exist_ok=True)
    data = {
        "axis": state.to_dict(),
        "tip1": {},
        "tip2": {},
        "updated_at": "",
    }
    pose = forward_kinematics(state)
    data["tip1"] = pose.tip1.to_dict()
    data["tip2"] = pose.tip2.to_dict()
    from datetime import datetime, timezone
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ============================================================
# Test
# ============================================================

if __name__ == '__main__':
    # Home position
    q_home = PipetteState.home()
    pose_home = forward_kinematics(q_home)
    print(f"Home:   X={X_REF:.1f} Y={Y_REF:.1f} Z1={Z_REF:.1f} Z2={Z_REF:.1f}")
    print(f"  Tip1 CAD: ({pose_home.tip1.x:.1f}, {pose_home.tip1.y:.1f}, {pose_home.tip1.z:.1f})")
    print(f"  Tip1 World: {pose_home.tip1.to_world()}")
    print(f"  Tip2 CAD: ({pose_home.tip2.x:.1f}, {pose_home.tip2.y:.1f}, {pose_home.tip2.z:.1f})")

    # STL assembly position
    q_stl = PipetteState.stl_assembly()
    print(f"\nSTL:    X={X_REF:.1f} Y={Y_REF_STL:.1f} Z1={Z_REF:.1f} Z2={Z_REF:.1f}")
    print(f"  [WARN] Y={Y_REF_STL:.1f} exceeds mechanical max={Y_MAX:.1f} by {Y_REF_STL-Y_MAX:.1f}mm")

    # IK test
    sol = inverse_kinematics(200, 200, 120, 120)
    print(f"\nIK:     target X=200 Y=200 Z1=120 Z2=120")
    print(f"  valid={sol.valid}, violations={sol.limit_violations}")

    # Out of bounds test
    sol2 = inverse_kinematics(0, 500, 50, 300)
    print(f"\nIK:     target X=0 Y=500 Z1=50 Z2=300 (all out of bounds)")
    print(f"  valid={sol2.valid}")
    for v in sol2.limit_violations:
        print(f"  - {v}")

    # Limits summary
    print(f"\nJoint limits:")
    print(f"  X:  [{X_MIN:.0f}, {X_MAX:.0f}], stroke={X_STROKE:.0f}mm, ref={X_REF:.1f}")
    print(f"  Y:  [{Y_MIN:.0f}, {Y_MAX:.0f}], stroke={Y_STROKE:.0f}mm, ref={Y_REF:.1f} (STL={Y_REF_STL:.1f})")
    print(f"  Z1: [{Z_MIN:.0f}, {Z_MAX:.0f}], stroke={Z_STROKE:.0f}mm, ref={Z_REF:.1f}")
    print(f"  Z2: [{Z_MIN:.0f}, {Z_MAX:.0f}], stroke={Z_STROKE:.0f}mm, ref={Z_REF:.1f}")
    print(f"  ADP spacing: {ADP_SPACING_X:.1f}mm in X")
