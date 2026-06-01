"""
Dobot M1Pro SCARA Robot Digital Twin — Flask Server
=====================================================
Serves the 3D interactive visualization and provides REST API
for kinematics computation.
"""

import json
import sys
import os

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from flask import Flask, request, jsonify, render_template

# Add project root for kinematics import
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from kinematics_M1Pro import (
    JointState, CartesianPose, IKSolution,
    forward_kinematics, inverse_kinematics, fk_compact,
    compute_jacobian, j3_deg_to_d3, d3_to_j3_deg,
    J1_MIN, J1_MAX, J2_MIN, J2_MAX, Z_MIN, Z_MAX, J4_MIN, J4_MAX,
    A1, A2, D1, D4, SCREW_LEAD, D3_BASE, D3_MIN, D3_MAX,
    deg2rad, rad2deg, clamp, wrap_180,
    compute_workspace_boundary, compute_workspace_inner,
)

from kinematics_pipette import (
    PipetteState, PipettePose, PipetteSolution,
    forward_kinematics as pipette_fk, fk_compact as pipette_fk_compact,
    inverse_kinematics as pipette_ik, get_joint_limits as pipette_get_limits,
    X_MIN as PX_MIN, X_MAX as PX_MAX, X_REF as PX_REF,
    Y_MIN as PY_MIN, Y_MAX as PY_MAX, Y_REF as PY_REF,
    Z_MIN as PZ_MIN, Z_MAX as PZ_MAX, Z_REF as PZ_REF,
    ADP_SPACING_X,
)

app = Flask(__name__, template_folder='templates')

# -------------------------------------------------------
# Serve UI
# -------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')


# -------------------------------------------------------
# Static file serving for 3D models (.stl, .step, .glb, .gltf)
# -------------------------------------------------------

@app.route('/models/<path:filename>')
def serve_model(filename):
    """Serve 3D model files from templates/models/ directory."""
    from flask import send_from_directory
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'models')
    return send_from_directory(models_dir, filename)


@app.route('/viewers/<path:filename>')
def serve_viewer(filename):
    """Serve generated HTML viewers from templates/viewers/ or templates/ directory."""
    from flask import send_from_directory
    # Check viewers subdirectory first, then templates root
    viewers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'viewers')
    templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
    # Try viewers dir first
    try:
        return send_from_directory(viewers_dir, filename)
    except:
        # Fall back to templates root
        return send_from_directory(templates_dir, filename)


# -------------------------------------------------------
# Kinematics API
# -------------------------------------------------------

@app.route('/api/fk', methods=['POST'])
def api_fk():
    """
    Forward Kinematics: joint → Cartesian.
    Body: {j1, j2, j3_deg, j4}  — J3 in protocol degrees
    Returns: {x, y, z, r, valid, joint_limits_ok}
    """
    data = request.get_json(force=True)
    try:
        j1 = float(data['j1'])
        j2 = float(data['j2'])
        j3_deg = float(data['j3_deg'])
        j4 = float(data['j4'])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    d3 = j3_deg_to_d3(j3_deg)
    q = JointState(j1_deg=j1, j2_deg=j2, d3_mm=d3, j4_deg=j4)
    pose, transforms = forward_kinematics(q)

    # Check joint limits
    limits_ok = (J1_MIN <= j1 <= J1_MAX and J2_MIN <= j2 <= J2_MAX and
                 D3_MIN <= d3 <= D3_MAX and J4_MIN <= j4 <= J4_MAX)

    # Persist to dobot_state.json
    from datetime import datetime
    _save_dobot_state({
        "joint": {"j1": j1, "j2": j2, "j3_deg": j3_deg, "j4": j4, "d3_mm": round(d3, 3)},
        "tcp": {"x": round(pose.x, 3), "y": round(pose.y, 3), "z": round(pose.z, 3), "r": round(pose.r, 3)},
        "updated_at": datetime.now().isoformat(timespec='seconds'),
    })

    return jsonify({
        'x': round(pose.x, 3),
        'y': round(pose.y, 3),
        'z': round(pose.z, 3),
        'r': round(pose.r, 3),
        'valid': True,
        'joint_limits_ok': limits_ok,
        'd3_mm': round(d3, 3),
    })


@app.route('/api/ik', methods=['POST'])
def api_ik():
    """
    Inverse Kinematics: Cartesian → joint.
    Body: {x, y, z, r, elbow_up(opt), near_joints(opt)}
    Returns both elbow-up and elbow-down solutions if available.
    """
    data = request.get_json(force=True)
    try:
        x = float(data['x'])
        y = float(data['y'])
        z = float(data['z'])
        r = float(data.get('r', 0))
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    solutions = []
    for elbow_up in [True, False]:
        sol = inverse_kinematics(x, y, z, r, elbow_up=elbow_up)
        info = {
            'elbow_up': elbow_up,
            'valid': sol.valid,
            'reason': sol.reason,
            'j1': round(sol.j1_deg, 3),
            'j2': round(sol.j2_deg, 3),
            'd3_mm': round(sol.d3_mm, 3),
            'j3_deg': round(d3_to_j3_deg(sol.d3_mm), 3),
            'j4': round(sol.j4_deg, 3),
        }
        if sol.valid:
            # Verify by FK
            qv = sol.to_joint_state()
            pv, _ = forward_kinematics(qv)
            from math import sqrt
            err = sqrt((pv.x-x)**2 + (pv.y-y)**2 + (pv.z-z)**2)
            info['verify_error_mm'] = round(err, 6)
        solutions.append(info)

    return jsonify({'solutions': solutions})


@app.route('/api/workspace', methods=['GET'])
def api_workspace():
    """Return workspace boundary (outer + inner) for visualization."""
    outer_x, outer_y = compute_workspace_boundary(360)
    inner_x, inner_y = compute_workspace_inner(360)
    return jsonify({
        'outer': {'x': [round(v, 2) for v in outer_x],
                  'y': [round(v, 2) for v in outer_y]},
        'inner': {'x': [round(v, 2) for v in inner_x],
                  'y': [round(v, 2) for v in inner_y]},
        'params': {
            'a1': A1, 'a2': A2, 'd1': D1, 'd4': D4,
            'reach_max': A1 + A2, 'reach_min': abs(A1 - A2),
        }
    })


@app.route('/api/joint_limits', methods=['GET'])
def api_joint_limits():
    """Return joint limits and DH parameters."""
    return jsonify({
        'joints': {
            'j1': {'min': J1_MIN, 'max': J1_MAX, 'unit': 'deg', 'type': 'revolute'},
            'j2': {'min': J2_MIN, 'max': J2_MAX, 'unit': 'deg', 'type': 'revolute'},
            'j3': {'min_deg': round(d3_to_j3_deg(Z_MAX), 1),
                   'max_deg': round(d3_to_j3_deg(Z_MIN), 1),
                   'z_min_mm': Z_MIN, 'z_max_mm': Z_MAX,
                   'unit': 'deg (motor) / mm (linear)', 'type': 'prismatic'},
            'j4': {'min': J4_MIN, 'max': J4_MAX, 'unit': 'deg', 'type': 'revolute'},
        },
        'dh_params': {
            'a1': A1, 'a2': A2, 'd1': D1, 'd4': D4,
        },
        'screw': {
            'lead_mm_per_rev': SCREW_LEAD,
            'd3_base_mm': D3_BASE,
        }
    })


@app.route('/api/jacobian', methods=['POST'])
def api_jacobian():
    """Compute Jacobian at given joint state."""
    data = request.get_json(force=True)
    try:
        j1 = float(data['j1'])
        j2 = float(data['j2'])
        j3_deg = float(data['j3_deg'])
        j4 = float(data['j4'])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    d3 = j3_deg_to_d3(j3_deg)
    q = JointState(j1_deg=j1, j2_deg=j2, d3_mm=d3, j4_deg=j4)
    J = compute_jacobian(q)
    return jsonify({
        'jacobian': J.tolist(),
        'determinant': round(float(
            J[0,0]*J[1,1]*(-1)*J[3,3] + J[0,0]*(-1)*J[1,0]*(-1) -
            J[0,1]*J[1,0]*(-1)*J[3,3] + 0
        ), 6) if J.shape == (4,4) else None,
    })


@app.route('/api/pose', methods=['GET'])
def api_pose():
    """Get current robot state (stored in server memory)."""
    return jsonify(app.config.get('robot_state', {
        'j1': 0, 'j2': 0, 'j3_deg': 0, 'j4': 0,
        'x': A1+A2, 'y': 0, 'z': D1 + D3_BASE - D4, 'r': 0,
    }))


# -------------------------------------------------------
# Platform Config API
# -------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CONFIG_PATH = os.path.join(DATA_DIR, 'config', 'platform_config.json')
DOBOT_STATE_PATH = os.path.join(DATA_DIR, 'runtime', 'dobot_state.json')
PIPETTE_STATE_PATH = os.path.join(DATA_DIR, 'runtime', 'pipette_state.json')


def _load_platform_config():
    """Read platform_config.json, return dict. Returns empty dict on failure."""
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[Config] Failed to load platform_config.json: {e}")
        return {}


def _save_platform_config(data):
    """Write platform_config.json. Returns True on success."""
    try:
        with open(CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[Config] Failed to save platform_config.json: {e}")
        return False


@app.route('/api/platform_config', methods=['GET'])
def api_get_platform_config():
    """Return platform module configuration."""
    cfg = _load_platform_config()
    return jsonify({'success': True, 'data': cfg})


@app.route('/api/platform_config', methods=['POST', 'PUT'])
def api_save_platform_config():
    """Save platform module configuration (JSON body)."""
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'Body must be a JSON object'}), 400
    ok = _save_platform_config(data)
    return jsonify({'success': ok, 'error': '' if ok else 'Failed to write file'})


# -------------------------------------------------------
# Pipette Arm API
# -------------------------------------------------------

@app.route('/api/pipette/limits', methods=['GET'])
def api_pipette_limits():
    """Return XYZZ+dual ADP pipette arm joint limits."""
    return jsonify({'success': True, 'data': pipette_get_limits()})


@app.route('/api/pipette/fk', methods=['POST'])
def api_pipette_fk():
    """
    Forward kinematics: axis values → tip positions.
    Body: {x, y, z1, z2}  — all in mm (logical axis positions)
    Returns tip1/tip2 positions in CAD and world coordinates.
    """
    data = request.get_json(force=True)
    try:
        x = float(data['x'])
        y = float(data['y'])
        z1 = float(data['z1'])
        z2 = float(data['z2'])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    state = PipetteState(x=x, y=y, z1=z1, z2=z2)
    pose = pipette_fk(state)
    # Persist to pipette_state.json
    from datetime import datetime
    _save_pipette_state({
        "axis": {"x": x, "y": y, "z1": z1, "z2": z2},
        "tip1": pose.tip1.to_dict(),
        "tip2": pose.tip2.to_dict(),
        "updated_at": datetime.now().isoformat(timespec='seconds'),
    })
    return jsonify({'success': True, 'data': pose.to_dict()})

@app.route('/api/pipette/ik', methods=['POST'])
def api_pipette_ik():
    """
    Inverse kinematics: target axis values → clamped values.
    Body: {x, y, z1, z2}  — target positions in mm
    Returns clamped values + limit violation warnings.
    """
    data = request.get_json(force=True)
    try:
        x = float(data['x'])
        y = float(data['y'])
        z1 = float(data['z1'])
        z2 = float(data['z2'])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({'error': f'Invalid parameters: {e}'}), 400

    sol = pipette_ik(x, y, z1, z2)
    return jsonify({'success': True, 'data': sol.to_dict()})


@app.route('/api/pipette/pose', methods=['GET'])
def api_pipette_pose():
    """Get current pipette state (stored in server memory)."""
    return jsonify(app.config.get('pipette_state', {
        'x': PX_REF, 'y': PY_REF, 'z1': PZ_REF, 'z2': PZ_REF,
    }))


# -------------------------------------------------------
# Runtime state — two JSON files: dobot_state.json + pipette_state.json
# -------------------------------------------------------

def _load_dobot_state():
    default = {
        "joint": {"j1": 0.0, "j2": 0.0, "j3_deg": 0.0, "j4": 0.0, "d3_mm": D3_BASE},
        "tcp": {"x": A1 + A2, "y": 0.0, "z": D1 + D3_BASE - D4, "r": 0.0},
        "updated_at": "",
    }
    try:
        with open(DOBOT_STATE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_dobot_state(state):
    try:
        with open(DOBOT_STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[DobotState] Failed to save: {e}")
        return False


def _load_pipette_state():
    default = {
        "axis": {"x": PX_REF, "y": PY_REF, "z1": PZ_REF, "z2": PZ_REF},
        "tip1": {"x": 0.0, "y": 0.0, "z": 0.0},
        "tip2": {"x": 0.0, "y": 0.0, "z": 0.0},
        "updated_at": "",
    }
    try:
        with open(PIPETTE_STATE_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _save_pipette_state(state):
    try:
        with open(PIPETTE_STATE_PATH, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[PipetteState] Failed to save: {e}")
        return False


@app.route('/api/runtime/dobot_state', methods=['GET'])
def api_get_dobot_state():
    return jsonify({'success': True, 'data': _load_dobot_state()})


@app.route('/api/runtime/dobot_state', methods=['POST', 'PUT'])
def api_set_dobot_state():
    data = request.get_json(force=True)
    state = _load_dobot_state()
    try:
        if 'joint' in data:
            for k in ('j1', 'j2', 'j3_deg', 'j4'):
                if k in data['joint']:
                    state['joint'][k] = float(data['joint'][k])
                    if k == 'j3_deg':
                        state['joint']['d3_mm'] = j3_deg_to_d3(state['joint'][k])
        if 'tcp' in data:
            for k in ('x', 'y', 'z', 'r'):
                if k in data['tcp']:
                    state['tcp'][k] = float(data['tcp'][k])
    except (TypeError, ValueError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    from datetime import datetime
    state['updated_at'] = datetime.now().isoformat(timespec='seconds')
    _save_dobot_state(state)
    return jsonify({'success': True, 'data': state})


@app.route('/api/runtime/pipette_state', methods=['GET'])
def api_get_pipette_state():
    return jsonify({'success': True, 'data': _load_pipette_state()})


@app.route('/api/runtime/pipette_state', methods=['POST', 'PUT'])
def api_set_pipette_state():
    data = request.get_json(force=True)
    state = _load_pipette_state()
    try:
        if 'axis' in data:
            for k in ('x', 'y', 'z1', 'z2'):
                if k in data['axis']:
                    state['axis'][k] = float(data['axis'][k])
    except (TypeError, ValueError) as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    # Recompute tips from axes
    pose = pipette_fk(PipetteState(**state['axis']))
    state['tip1'] = pose.tip1.to_dict()
    state['tip2'] = pose.tip2.to_dict()
    from datetime import datetime
    state['updated_at'] = datetime.now().isoformat(timespec='seconds')
    _save_pipette_state(state)
    return jsonify({'success': True, 'data': state})


# -------------------------------------------------------
# Start
# -------------------------------------------------------

if __name__ == '__main__':
    print(f"[DigitalTwin] Dobot M1Pro SCARA Robot")
    print(f"  DH: a1={A1}mm, a2={A2}mm, d1={D1}mm, d4={D4}mm")
    print(f"  Reach: {abs(A1-A2):.0f}–{A1+A2:.0f}mm")
    print(f"  Z stroke: {Z_MIN}–{Z_MAX}mm | d3=[{D3_MIN}–{D3_MAX}]mm | D3_BASE={D3_BASE}mm (screw lead={SCREW_LEAD}mm/rev)")
    print(f"  J1=[{J1_MIN},{J1_MAX}]°, J2=[{J2_MIN},{J2_MAX}]°, J4=[{J4_MIN},{J4_MAX}]°")
    print(f"")
    print(f"[DigitalTwin] XYZZ+dual ADP Pipette Arm")
    print(f"  X: [{PX_MIN:.0f}, {PX_MAX:.0f}]mm, stroke={PX_MAX-PX_MIN:.0f}mm, ref={PX_REF:.1f}")
    print(f"  Y: [{PY_MIN:.0f}, {PY_MAX:.0f}]mm, stroke={PY_MAX-PY_MIN:.0f}mm, ref={PY_REF:.1f}")
    print(f"  Z1/Z2: [{PZ_MIN:.0f}, {PZ_MAX:.0f}]mm, stroke={PZ_MAX-PZ_MIN:.0f}mm, ref={PZ_REF:.1f}")
    print(f"  ADP spacing: {ADP_SPACING_X:.1f}mm")
    print(f"  Open http://127.0.0.1:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
