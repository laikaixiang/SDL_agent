"""
Extract Dobot M1Pro joint pivot positions from STL assembly geometry.
Outputs dobot_joint_offsets.json for use by index.html Three.js scene.
"""
import struct
import os
import json

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'models', 'dobot_m1pro')

# DH parameters (from kinematics.py)
A1, A2 = 250.0, 150.0  # link lengths (mm)
D1, D4 = 85.0, 80.0    # base height, tool offset (mm)

PARTS = ['M1_AXIS1_1', 'M1_AXIS2_1', 'M1_AXIS3_1', 'M1_AXIS4_1', 'M1_ROTATE_1']


def read_stl_bbox(filepath):
    """Return (center, bbox_min, bbox_max) for an STL file."""
    with open(filepath, 'rb') as f:
        f.read(80)
        tri_count = struct.unpack('<I', f.read(4))[0]
        data = f.read(tri_count * 50)

    xs, ys, zs = [], [], []
    for i in range(tri_count):
        vals = struct.unpack_from('<12f', data, i * 50)
        for j in range(3, 12, 3):
            xs.append(vals[j])
            ys.append(vals[j + 1])
            zs.append(vals[j + 2])

    bmin = (min(xs), min(ys), min(zs))
    bmax = (max(xs), max(ys), max(zs))
    center = ((bmin[0] + bmax[0]) / 2, (bmin[1] + bmax[1]) / 2, (bmin[2] + bmax[2]) / 2)
    return center, bmin, bmax


def main():
    bboxes = {}
    for part in PARTS:
        fpath = os.path.join(MODELS_DIR, part + '.stl')
        center, bmin, bmax = read_stl_bbox(fpath)
        bboxes[part] = {'center': center, 'min': bmin, 'max': bmax}

    # ---- Determine joint positions in assembly coordinates ----
    # Assembly coordinate system: X=forward along arm, Y=up (vertical), Z=lateral
    # DH coordinate system: X=forward, Y=left, Z=up
    # Mapping: assy(X,Y,Z) → dh(X, -Z, Y)

    # All parts share approximate Y-center ~116.3mm (the arm plane)
    # The arm lies primarily in the XZ plane of the assembly.

    # AXIS1 is the base — its bottom is at assy Y≈1.2mm (table level)
    # J1 rotation axis (vertical) passes through the base column center.
    # From AXIS1 bbox: X=[0.8,176.2], Z=[0,687.6]
    # The base column center in XZ ≈ (88.5, 343.8)

    # Direction from base to wrist in the assembly XZ plane:
    c1 = bboxes['M1_AXIS1_1']['center']
    c4 = bboxes['M1_AXIS4_1']['center']
    dx = c4[0] - c1[0]
    dz = c4[2] - c1[2]
    arm_length_assy = (dx**2 + dz**2)**0.5
    print(f'Arm vector in assembly XZ: ({dx:.1f}, {dz:.1f}), length={arm_length_assy:.1f}mm')
    print(f'Expected DH arm length: A1+A2 = {A1+A2}mm')

    # Compute joint positions along the arm direction
    # J1 (base origin in DH) → assembly position at base column center
    j1_assy = (c1[0], c1[1], c1[2])

    # J2 is A1=250mm from J1 along the arm direction
    scale = A1 / arm_length_assy if arm_length_assy > 0 else 1
    j2_ass_x = j1_assy[0] + dx * scale
    j2_ass_z = j1_assy[2] + dz * scale
    j2_assy = (j2_ass_x, c1[1], j2_ass_z)

    # J4 is A1+A2=400mm from J1 along the arm direction
    scale = (A1 + A2) / arm_length_assy if arm_length_assy > 0 else 1
    j4_ass_x = j1_assy[0] + dx * scale
    j4_ass_z = j1_assy[2] + dz * scale
    j4_assy = (j4_ass_x, c1[1], j4_ass_z)

    print(f'\nJoint positions in assembly coords:')
    print(f'  J1: ({j1_assy[0]:.1f}, {j1_assy[1]:.1f}, {j1_assy[2]:.1f})')
    print(f'  J2: ({j2_assy[0]:.1f}, {j2_assy[1]:.1f}, {j2_assy[2]:.1f})')
    print(f'  J4: ({j4_assy[0]:.1f}, {j4_assy[1]:.1f}, {j4_assy[2]:.1f})')

    # For each STL part, compute the offset from its parent joint to the part center
    # In the Three.js hierarchy, each part needs to be positioned relative to its joint pivot

    # Part → joint mapping:
    # AXIS1 → baseGroup (fixed, no joint — uses J1 as reference for placement)
    # ROTATE → j1Pivot (rotates around assembly Y axis at J1)
    # AXIS2 → j2Pivot (rotates around assembly Y axis at J2)
    # AXIS3 → liftCarriage (moves with J3 lift)
    # AXIS4 → j4Pivot (rotates around assembly Y axis at J4)

    # For Three.js, robots sit in the XZ plane (Y=up).
    # We need to map assembly coordinates to Three.js coordinates.
    # Three.js: X=right, Y=up, Z=forward (toward viewer)
    # We'll place the arm so it extends in Three.js XZ plane, with Y=up for rotation axes.
    #
    # Assembly → Three.js mapping:
    #   assy X → three X (along the arm, forward)
    #   assy Y → three Y (up, rotation axis)
    #   assy Z → three Z (lateral)
    #
    # We'll center the robot at its J1 base in the scene.

    offsets = {}

    # For each part, compute:
    # 1. Which joint it belongs to
    # 2. The offset from that joint to the part's assembly position
    # 3. The part's bbox extent (for reference)

    # Part assignments based on geometry analysis:
    # ROTATE: small disk at far end (X~604) — this is the J4 end-effector
    #   → belongs to j4Pivot
    # AXIS1: largest part, base column (X 0-176) — contains fixed base + lift column
    #   → belongs to baseGroup (fixed)
    # AXIS2: mid section (X 123-254) — rotates with J2
    #   → belongs to j2Pivot
    # AXIS3: mid-upper section (X 164-443) — this is the lift carriage/upper arm
    #   → belongs to liftCarriage (moves with J3, does NOT rotate with J1/J2)
    # AXIS4: far section (X 372-635) — the wrist/forearm
    #   → belongs to j4Pivot

    part_to_joint = {
        'M1_AXIS1_1': 'baseGroup',
        'M1_AXIS3_1': 'liftCarriage',
        'M1_AXIS2_1': 'j2Pivot',
        'M1_AXIS4_1': 'j4Pivot',
        'M1_ROTATE_1': 'j4Pivot',
    }

    joint_positions = {
        'J1_base': j1_assy,
        'J2_elbow': j2_assy,
        'J4_wrist': j4_assy,
    }

    for part in PARTS:
        center = bboxes[part]['center']
        joint_name = part_to_joint[part]

        # Determine which joint position to use
        if joint_name in ('baseGroup', 'liftCarriage'):
            joint_pos = j1_assy
        elif joint_name == 'j2Pivot':
            joint_pos = j2_assy
        elif joint_name == 'j4Pivot':
            joint_pos = j4_assy
        else:
            joint_pos = j1_assy

        # Offset from joint to part center in assembly coords
        offset = (
            center[0] - joint_pos[0],
            center[1] - joint_pos[1],
            center[2] - joint_pos[2],
        )

        offsets[part] = {
            'parent': joint_name,
            'joint_ref': joint_name,
            'assembly_center': [round(c, 1) for c in center],
            'offset_from_joint': [round(o, 1) for o in offset],
            'bbox_min': [round(v, 1) for v in bboxes[part]['min']],
            'bbox_max': [round(v, 1) for v in bboxes[part]['max']],
        }

    # ---- Compute centered offsets (subtract J1 to bring robot to origin) ----
    j1x, j1y, j1z = j1_assy
    centered_joints = {}
    for k, vs in joint_positions.items():
        centered_joints[k] = [round(vs[0] - j1x, 1), round(vs[1] - j1y, 1), round(vs[2] - j1z, 1)]

    centered_offsets = {}
    for part in PARTS:
        parent = part_to_joint[part]
        ref_joint = 'J1_base' if parent in ('baseGroup', 'liftCarriage') else \
                    'J2_elbow' if parent == 'j2Pivot' else 'J4_wrist'

        cj = centered_joints[ref_joint]
        ca = bboxes[part]['center']
        cx, cy, cz = ca[0] - j1x, ca[1] - j1y, ca[2] - j1z
        ox, oy, oz = cx - cj[0], cy - cj[1], cz - cj[2]

        centered_offsets[part] = {
            'parent': parent,
            'center_from_j1': [round(cx, 1), round(cy, 1), round(cz, 1)],
            'offset_from_parent_joint': [round(ox, 1), round(oy, 1), round(oz, 1)],
            'bbox_size': [round(bboxes[part]['max'][i] - bboxes[part]['min'][i], 1) for i in range(3)],
        }

    # Build output
    output = {
        'description': 'Dobot M1Pro joint offsets from STL assembly geometry',
        'coordinate_system': 'Assembly: X=forward along arm, Y=up/vertical, Z=lateral',
        'joint_positions_assembly': {
            k: [round(v, 1) for v in vs] for k, vs in joint_positions.items()
        },
        'joint_positions_centered': centered_joints,
        'part_offsets': centered_offsets,
    }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dobot_joint_offsets.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f'\nSaved to {out_path}')
    return output


if __name__ == '__main__':
    main()
