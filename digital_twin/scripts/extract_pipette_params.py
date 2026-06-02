"""
Extract XYZZ+dual ADP pipette arm kinematic parameters from STL assembly geometry.

Analyzes the 6 simplified STL parts to determine:
  - Axis travel ranges (X, Z, Y1, Y2) with mechanical constraints
  - ADP spacing
  - Reference/home positions
  - Coordinate system mappings

Outputs pipette_kinematic_params.json
"""

import struct
import os
import sys
import json

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'templates', 'models', 'pipette_arm', 'simplified')

PARTS = {
    'pipette_group1': 'frame',      # main frame/column
    'pipette_group2': 'cross_beam', # cross beam, slides along frame Y
    'pipette_group3': 'mid_beam',   # middle beam, slides along cross beam X
    'pipette_group4': 'adp1',       # ADP module 1, moves vertically (CAD Z)
    'pipette_group5': 'adp2',       # ADP module 2, moves vertically (CAD Z)
    'pipette_tip':    'tip',        # pipette tip, attached to ADP1
}


def read_stl_bbox(filepath):
    """Return (center, bbox_min, bbox_max, size) for an STL file."""
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
    center = ((bmin[0] + bmax[0]) / 2,
              (bmin[1] + bmax[1]) / 2,
              (bmin[2] + bmax[2]) / 2)
    size = (bmax[0] - bmin[0], bmax[1] - bmin[1], bmax[2] - bmin[2])
    return center, bmin, bmax, size


def main():
    bboxes = {}
    for fname, label in PARTS.items():
        fpath = os.path.join(MODELS_DIR, fname + '.stl')
        center, bmin, bmax, size = read_stl_bbox(fpath)
        bboxes[label] = {
            'center': [round(v, 1) for v in center],
            'min': [round(v, 1) for v in bmin],
            'max': [round(v, 1) for v in bmax],
            'size': [round(v, 1) for v in size],
        }

    frm  = bboxes['frame']
    cb   = bboxes['cross_beam']
    mb   = bboxes['mid_beam']
    adp1 = bboxes['adp1']
    adp2 = bboxes['adp2']
    tip  = bboxes['tip']

    print("=" * 70)
    print("XYZZ+双ADP 移液机械臂 — 运动学参数提取")
    print("=" * 70)

    # ---- 1. Assembly overview ----
    print("\n--- 装配体包围盒 (CAD 坐标: Z=上, X=前, Y=左) ---")
    all_x = [bb['min'][0] for bb in bboxes.values()] + [bb['max'][0] for bb in bboxes.values()]
    all_y = [bb['min'][1] for bb in bboxes.values()] + [bb['max'][1] for bb in bboxes.values()]
    all_z = [bb['min'][2] for bb in bboxes.values()] + [bb['max'][2] for bb in bboxes.values()]
    print(f"  总成: X[{min(all_x):.1f}, {max(all_x):.1f}] "
          f"Y[{min(all_y):.1f}, {max(all_y):.1f}] "
          f"Z[{min(all_z):.1f}, {max(all_z):.1f}]")
    print(f"  尺寸: {max(all_x)-min(all_x):.1f} × {max(all_y)-min(all_y):.1f} × {max(all_z)-min(all_z):.1f} mm")
    print(f"  底座占地面积: ~{(max(all_x)-min(all_x)):.0f} × {(max(all_y)-min(all_y)):.0f} mm")

    for label, bb in bboxes.items():
        print(f"  {label:12s}: X[{bb['min'][0]:6.1f}, {bb['max'][0]:6.1f}] "
              f"Y[{bb['min'][1]:6.1f}, {bb['max'][1]:6.1f}] "
              f"Z[{bb['min'][2]:6.1f}, {bb['max'][2]:6.1f}]  "
              f"size=({bb['size'][0]:.0f}, {bb['size'][1]:.0f}, {bb['size'][2]:.0f})")

    # ---- 2. Reference positions (part centers in CAD) ----
    # These correspond to the "home" position where parts are centered in their guides
    x_ref = mb['center'][0]                      # mid beam center X = 300.3
    z_ref = cb['center'][1]                      # cross beam center Y (CAD) = 416.1
    y1_ref = adp1['center'][2]                   # ADP1 center Z (CAD) = 123.1
    y2_ref = adp2['center'][2]                   # ADP2 center Z (CAD) = 123.1

    print(f"\n--- 参考位置 (零件几何中心) ---")
    print(f"  X_ref  = {x_ref:.1f} mm  (中梁 center X in CAD)")
    print(f"  Z_ref  = {z_ref:.1f} mm  (横梁 center Y in CAD)")
    print(f"  Y1_ref = {y1_ref:.1f} mm  (ADP1 center Z in CAD)")
    print(f"  Y2_ref = {y2_ref:.1f} mm  (ADP2 center Z in CAD)")

    # ---- 3. X axis: mid beam (group3) slides on cross beam (group2) in CAD X ----
    mb_half_x = mb['size'][0] / 2                # half-width of mid beam in X
    cb_x_min = cb['min'][0]                       # cross beam X range
    cb_x_max = cb['max'][0]
    x_mech_min = cb_x_min + mb_half_x            # mid beam center must clear cross beam ends
    x_mech_max = cb_x_max - mb_half_x

    print(f"\n--- X 轴: 中梁在横梁上滑动 (CAD X 方向) ---")
    print(f"  横梁 X 范围: [{cb_x_min:.1f}, {cb_x_max:.1f}], 跨度={cb_x_max-cb_x_min:.1f}mm")
    print(f"  中梁 X 宽度: {mb['size'][0]:.1f}mm, 半宽={mb_half_x:.1f}mm")
    print(f"  机械限位: X ∈ [{x_mech_min:.1f}, {x_mech_max:.1f}]")
    print(f"  机械行程: {x_mech_max - x_mech_min:.1f}mm")
    print(f"  参考位置: {x_ref:.1f}")

    # ---- 4. Z axis: cross beam (group2) slides along frame (group1) in CAD Y ----
    cb_half_y = cb['size'][1] / 2
    frm_y_min = frm['min'][1]
    frm_y_max = frm['max'][1]
    z_mech_min = frm_y_min + cb_half_y
    z_mech_max = frm_y_max - cb_half_y

    print(f"\n--- Z 轴: 横梁沿框架移动 (CAD Y 方向) ---")
    print(f"  框架 Y 范围: [{frm_y_min:.1f}, {frm_y_max:.1f}], 跨度={frm_y_max-frm_y_min:.1f}mm")
    print(f"  横梁 Y 深度: {cb['size'][1]:.1f}mm, 半深={cb_half_y:.1f}mm")
    print(f"  机械限位: Z ∈ [{z_mech_min:.1f}, {z_mech_max:.1f}]")
    print(f"  机械行程: {z_mech_max - z_mech_min:.1f}mm")
    print(f"  参考位置: {z_ref:.1f}")
    if z_ref > z_mech_max:
        print(f"  [WARN] Z_ref={z_ref:.1f} exceeds mechanical max={z_mech_max:.1f} by {z_ref-z_mech_max:.1f}mm")

    # ---- 5. Y1/Y2 axis: ADPs move in CAD Z (vertical) ----
    mb_half_z = mb['size'][2] / 2               # half-height of mid beam in Z
    adp_half_z = adp1['size'][2] / 2
    mb_z_min = mb['min'][2]
    mb_z_max = mb['max'][2]

    # ADPs must stay within mid beam's Z range
    y_mech_min = mb_z_min + adp_half_z
    y_mech_max = mb_z_max - adp_half_z

    print(f"\n--- Y1/Y2 轴: ADP 模块升降 (CAD Z 方向, 垂直) ---")
    print(f"  中梁 Z 范围: [{mb_z_min:.1f}, {mb_z_max:.1f}], 高度={mb_z_max-mb_z_min:.1f}mm")
    print(f"  ADP Z 高度: {adp1['size'][2]:.1f}mm, 半高={adp_half_z:.1f}mm")
    print(f"  机械限位: Y ∈ [{y_mech_min:.1f}, {y_mech_max:.1f}]")
    print(f"  机械行程: {y_mech_max - y_mech_min:.1f}mm")
    print(f"  Y1/Y2 参考位置: {y1_ref:.1f}")

    # ---- 6. ADP spacing ----
    adp1_cx = adp1['center'][0]
    adp2_cx = adp2['center'][0]
    adp_spacing = abs(adp2_cx - adp1_cx)

    print(f"\n--- ADP 模块间距 ---")
    print(f"  ADP1 center X: {adp1_cx:.1f}")
    print(f"  ADP2 center X: {adp2_cx:.1f}")
    print(f"  间距: {adp_spacing:.1f}mm (X方向)")

    # ---- 7. Summarize: recommended safe operating ranges ----
    # The current slider ranges from index.html vs. mechanical limits
    print(f"\n--- 推荐工作范围 vs 当前滑块设置 ---")
    print(f"  轴    机械限位                     当前滑块        推荐范围")
    print(f"  X     [{x_mech_min:.0f}, {x_mech_max:.0f}]              [200, 420]       [{max(x_mech_min,0):.0f}, {x_mech_max:.0f}]")
    print(f"  Z     [{z_mech_min:.0f}, {z_mech_max:.0f}]              [300, 500]       [{z_mech_min:.0f}, {z_mech_max:.0f}]")
    print(f"  Y1/Y2 [{y_mech_min:.0f}, {y_mech_max:.0f}]              [0, 200]         [{y_mech_min:.0f}, {y_mech_max:.0f}]")

    # ---- 8. Build output ----
    output = {
        "description": "XYZZ+双ADP 移液机械臂 — 运动学参数 (从STL装配体几何提取)",
        "model": "XYZZ+双ADP 移液机械臂 (2025.03.13)",
        "coordinate_system": {
            "cad": "Z=上, X=前, Y=左",
            "after_rx_neg90": "CAD_X→World_X, CAD_Y→World_Z, CAD_Z→World_-Y",
            "logical_axes": {
                "X":  "CAD X, World X — 中梁在横梁上水平滑动",
                "Z":  "CAD Y, World Z — 横梁沿框架前后移动",
                "Y1": "CAD Z, World Y — ADP1 垂直升降",
                "Y2": "CAD Z, World Y — ADP2 垂直升降",
            }
        },
        "assembly_bounds": {
            "x": [round(min(all_x), 1), round(max(all_x), 1)],
            "y": [round(min(all_y), 1), round(max(all_y), 1)],
            "z": [round(min(all_z), 1), round(max(all_z), 1)],
            "size_mm": [round(max(all_x)-min(all_x), 1),
                        round(max(all_y)-min(all_y), 1),
                        round(max(all_z)-min(all_z), 1)],
            "footprint_mm": f"{(max(all_x)-min(all_x)):.0f}×{(max(all_y)-min(all_y)):.0f}",
            "height_mm": round(max(all_z)-min(all_z), 1),
        },
        "joints": {
            "X": {
                "type": "prismatic",
                "description": "中梁在横梁上沿X方向滑动",
                "mechanical_range": [round(x_mech_min, 1), round(x_mech_max, 1)],
                "stroke_mm": round(x_mech_max - x_mech_min, 1),
                "reference_mm": round(x_ref, 1),
            },
            "Z": {
                "type": "prismatic",
                "description": "横梁沿框架Y方向移动",
                "mechanical_range": [round(z_mech_min, 1), round(z_mech_max, 1)],
                "stroke_mm": round(z_mech_max - z_mech_min, 1),
                "reference_mm": round(z_ref, 1),
            },
            "Y1": {
                "type": "prismatic",
                "description": "ADP1模块垂直升降",
                "mechanical_range": [round(y_mech_min, 1), round(y_mech_max, 1)],
                "stroke_mm": round(y_mech_max - y_mech_min, 1),
                "reference_mm": round(y1_ref, 1),
            },
            "Y2": {
                "type": "prismatic",
                "description": "ADP2模块垂直升降",
                "mechanical_range": [round(y_mech_min, 1), round(y_mech_max, 1)],
                "stroke_mm": round(y_mech_max - y_mech_min, 1),
                "reference_mm": round(y2_ref, 1),
            },
        },
        "adp_spacing_mm": round(adp_spacing, 1),
        "parts": {},
    }

    for label, bb in bboxes.items():
        output["parts"][label] = {
            "center": bb['center'],
            "size": bb['size'],
        }

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'pipette_kinematic_params.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")
    return output


if __name__ == '__main__':
    main()
