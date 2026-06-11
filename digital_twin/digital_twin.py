"""
Dobot M1Pro SCARA Robot Digital Twin — Flask Server
=====================================================
Serves the 3D interactive visualization and provides REST API
for kinematics computation.
"""

import json
import sys
import os
import time
import queue
import re
import threading
from typing import Optional, Tuple

from flask import Flask, request, jsonify, render_template, Response

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from flask import Flask, request, jsonify, render_template

# Add project root for kinematics import + hardware.utils/tools import
_DT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DT_DIR)                        # digital_twin/
sys.path.insert(0, os.path.dirname(_DT_DIR))       # project root (for hardware/)

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


@app.route('/api/kinematic_params', methods=['GET'])
def api_kinematic_params():
    """Return kinematic limits and reference positions from JSON config files."""
    import kinematics_M1Pro as km
    import kinematics_pipette as kp

    # Dobot limits (from kinematics_M1Pro which already reads from JSON)
    dob_limits = {
        'j1': {'min': km.J1_MIN, 'max': km.J1_MAX, 'unit': 'deg'},
        'j2': {'min': km.J2_MIN, 'max': km.J2_MAX, 'unit': 'deg'},
        'j3_deg': {
            'min': km.J3_DEG_MIN,
            'max': km.J3_DEG_MAX,
        },
        'j4': {'min': km.J4_MIN, 'max': km.J4_MAX, 'unit': 'deg'},
        'd3_mm': {'min': km.Z_MIN, 'max': km.Z_MAX, 'unit': 'mm'},
    }

    # Pipette limits (from kinematics_pipette which already reads from JSON)
    pip_limits = {
        'x':  {'min': kp.X_MIN, 'max': kp.X_MAX, 'ref': kp.X_REF, 'unit': 'mm'},
        'y':  {'min': kp.Y_MIN, 'max': kp.Y_MAX, 'ref': kp.Y_REF, 'ref_stl': kp.Y_REF_STL, 'unit': 'mm'},
        'z1': {'min': kp.Z_MIN, 'max': kp.Z_MAX, 'ref': kp.Z_REF, 'unit': 'mm'},
        'z2': {'min': kp.Z_MIN, 'max': kp.Z_MAX, 'ref': kp.Z_REF, 'unit': 'mm'},
    }

    return jsonify({
        'dobot': dob_limits,
        'pipette': pip_limits,
    })


@app.route('/api/joint_limits', methods=['GET'])
def api_joint_limits():
    """Return joint limits and DH parameters."""
    return jsonify({
        'joints': {
            'j1': {'min': J1_MIN, 'max': J1_MAX, 'unit': 'deg', 'type': 'revolute'},
            'j2': {'min': J2_MIN, 'max': J2_MAX, 'unit': 'deg', 'type': 'revolute'},
            'j3': {'min_deg': J3_DEG_MIN,
                   'max_deg': J3_DEG_MAX,
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


# ============================================================
# 统一坐标转换: Z=垂直(J3方向), X/Y=水平面
# move_robot_arm(x,y,z,r) 与 IK(x,y,z) 一致
# placement.yaw 绕 Z (垂直)转, Z 高度不变
# ============================================================
import math as _m


def _get_placement(robot: str) -> dict:
    cfg = _load_platform_config()
    return cfg.get('placement', {}).get(robot, {'x': 0, 'z': 0, 'yaw': 0})


def _world_to_local(x: float, y: float, z: float, p: dict) -> tuple[float, float, float]:
    """世界 (X,Y,Z=高度) → 机器人本地。只在 X-Y 水平面反向旋转,Z 不变。
    用户 Y 方向与 SCARA 本地 Y 一致(机器人伸臂方向)。
    """
    dx, dy = x - p.get('x', 0), y - p.get('z', 0)
    rad = _m.radians(p.get('yaw', 0))
    c, s = _m.cos(rad), _m.sin(rad)
    return c * dx + s * dy, -s * dx + c * dy, z


def _local_to_world(x: float, y: float, z: float, p: dict) -> tuple[float, float, float]:
    """机器人本地 → 世界。只在 X-Y 水平面正向旋转,Z 不变。"""
    px, pz = p.get('x', 0), p.get('z', 0)
    rad = _m.radians(p.get('yaw', 0))
    c, s = _m.cos(rad), _m.sin(rad)
    return c * x - s * y + px, s * x + c * y + pz, z


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


@app.route('/api/layout/placement', methods=['GET'])
def api_get_placement():
    """Return only the dobot/pipette base placement (position + Z-yaw)."""
    cfg = _load_platform_config()
    return jsonify({
        'success': True,
        'data': cfg.get('placement', {
            'dobot':   {'x': 150, 'z': -600, 'yaw': 0},
            'pipette': {'x': 950, 'z': -250, 'yaw': 0},
        })
    })


@app.route('/api/layout/placement', methods=['POST', 'PUT'])
def api_save_placement():
    """Update only the dobot/pipette base placement, preserving other config."""
    data = request.get_json(force=True)
    if not isinstance(data, dict):
        return jsonify({'success': False, 'error': 'Body must be a JSON object'}), 400
    cfg = _load_platform_config()
    cfg['placement'] = {
        'dobot': {
            'x':   float(data.get('dobot',   {}).get('x',   cfg.get('placement', {}).get('dobot',   {}).get('x',   150))),
            'z':   float(data.get('dobot',   {}).get('z',   cfg.get('placement', {}).get('dobot',   {}).get('z',  -600))),
            'yaw': float(data.get('dobot',   {}).get('yaw', cfg.get('placement', {}).get('dobot',   {}).get('yaw',   0))),
        },
        'pipette': {
            'x':   float(data.get('pipette', {}).get('x',   cfg.get('placement', {}).get('pipette', {}).get('x',   950))),
            'z':   float(data.get('pipette', {}).get('z',   cfg.get('placement', {}).get('pipette', {}).get('z',  -250))),
            'yaw': float(data.get('pipette', {}).get('yaw', cfg.get('placement', {}).get('pipette', {}).get('yaw',   0))),
        },
    }
    ok = _save_platform_config(cfg)
    return jsonify({'success': ok, 'data': cfg['placement'],
                    'error': '' if ok else 'Failed to write file'})


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
# 离线规划器 — 数字孪生执行端点
# 背后是 index.html 的 Three.js 3D 动画
# -------------------------------------------------------

_twin_state = {"status": "idle", "current_task": "", "reason": ""}

# ============================================================
# SSE (Server-Sent Events) — 服务端主动推送到浏览器
# ============================================================
# 每个浏览器连上 /api/twin/stream 就分配一个 Queue,
# 后端动作完成后 notify() 把事件塞到所有 Queue 里,
# 浏览器 EventSource 收到事件触发 animateTo / 更新状态条。
# ============================================================
_sse_subscribers = []

# ============================================================
# 主题 pub/sub — 内存字典实现的"伪 MQTT"
# 服务端可以 publish,前端通过 /api/twin/publish/<topic> 也能 publish
# 内部模块可订阅等待事件
# ============================================================
_topics = {
    "joint":   [],   # 服务端发布 → SSE 推给所有浏览器
    "done":    [],   # 前端动画完成后发布 → call_tool 等这个
    "anomaly": [],
}

# 每个主题保留最后一次事件(兜底:避免 publish 在 subscribe 之前到达丢失)
_topic_last = {}


def publish(topic: str, data: dict):
    """向一个主题的所有订阅者派发事件(同步),同时保留最后一次事件作为兜底。"""
    _topic_last[topic] = data
    for cb in list(_topics.get(topic, [])):
        try:
            cb(data)
        except Exception as e:
            print(f"[topic:{topic}] subscriber error: {e}")


def subscribe(topic: str, callback):
    """订阅一个主题,返回 unsubscribe 函数。"""
    if topic not in _topics:
        _topics[topic] = []
    _topics[topic].append(callback)
    def _unsub():
        try:
            _topics[topic].remove(callback)
        except ValueError:
            pass
    return _unsub


def wait_event(topic: str, timeout: float = 10.0):
    """阻塞等待主题的第一个事件。返回事件数据,超时返回 None。
    兜底:如果 publish 在 subscribe 之前到达,事件已存到 _topic_last,直接取走。"""
    # 先看 buffer 里有"积压事件"没
    if topic in _topic_last:
        return _topic_last.pop(topic)

    import threading
    result = [None]
    done = threading.Event()

    def _listener(data):
        if not done.is_set():
            result[0] = data
            done.set()

    unsubscribe = subscribe(topic, _listener)
    try:
        done.wait(timeout=timeout)
    finally:
        unsubscribe()
    return result[0]


def _notify(event_type: str, data: dict):
    """向所有 SSE 订阅者推送一个事件。"""
    payload = json.dumps(data, ensure_ascii=False)
    dead = []
    for q in _sse_subscribers:
        try:
            q.put_nowait((event_type, payload))
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            _sse_subscribers.remove(q)
        except ValueError:
            pass


def _set_twin_state(status, current_task="", reason=""):
    """更新 _twin_state 并通过 SSE 推送。"""
    _twin_state["status"] = status
    _twin_state["current_task"] = current_task
    _twin_state["reason"] = reason
    _notify("status", _twin_state)


@app.route('/api/twin/stream')
def twin_stream():
    """SSE 推送端点。客户端 EventSource 连接后,后端主动推送 status / joint 事件。"""
    def gen():
        q = queue.Queue()
        _sse_subscribers.append(q)
        yield "event: hello\ndata: {\"subscribers\": %d}\n\n" % len(_sse_subscribers)
        try:
            while True:
                try:
                    evt, payload = q.get(timeout=15)
                    yield f"event: {evt}\ndata: {payload}\n\n"
                except queue.Empty:
                    yield ": ping\n\n"  # 防止代理超时
        except GeneratorExit:
            pass
        finally:
            try:
                _sse_subscribers.remove(q)
            except ValueError:
                pass
    return Response(gen(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    })


@app.route('/api/twin/health', methods=['GET'])
def twin_health():
    return jsonify({"status": "ok"})


@app.route('/api/twin/status', methods=['GET'])
def twin_status():
    return jsonify(_twin_state)


@app.route('/api/twin/listener/start', methods=['POST'])
def twin_listener_start():
    """启动 MQTT 'twin' 主题实时监听(后台线程,持续推送关节角到前端)。"""
    start_twin_listener()
    return jsonify({"ok": True, "running": is_twin_listener_running()})


@app.route('/api/twin/listener/stop', methods=['POST'])
def twin_listener_stop():
    """停止实时监听线程。"""
    stop_twin_listener()
    return jsonify({"ok": True, "running": is_twin_listener_running()})


@app.route('/api/twin/listener/status', methods=['GET'])
def twin_listener_status():
    return jsonify({
        "running": is_twin_listener_running(),
        "poll_interval_s": _TWIN_POLL_INTERVAL,
        "supported_prefixes": {"arm": "a", "pipette": "d"},
        "formats": {
            "arm":     "a{x},{y},{z},{r}",
            "pipette": "d{x},{y},{lz},{rz}",
        },
    })


@app.route('/api/twin/execute', methods=['POST'])
def twin_execute():
    """
    接收 move_robot_arm 指令,碰撞检测 → IK → 状态保存 → 返 done。
    状态变更通过 SSE 主动推给浏览器 (无需轮询)。
    """
    data = request.get_json(force=True)
    msg = data.get('msg', '')

    _set_twin_state("busy", current_task=msg)

    import re
    m = re.match(r'^a([-\d.]+),([-\d.]+),([-\d.]+),([-\d.]+),(\d+)$', msg)
    if not m:
        _set_twin_state("error", reason=f"格式无效: {msg}")
        return jsonify({"result": "error", "reason": f"格式无效: {msg}"}), 400

    x, y, z, r = (float(m.group(i)) for i in (1, 2, 3, 4))

    # 世界坐标 → 本地 (Z=高度不变,只转 X-Y)
    p = _get_placement('dobot')
    lx, ly, lz = _world_to_local(x, y, z, p)

    # 碰撞检测 (本地坐标)
    from hardware.utils.collision import check_collision
    code, reason = check_collision({"x": lx, "y": ly, "z": lz, "r": r})
    if code != 200:
        _set_twin_state("error", reason=reason)
        return jsonify({"result": "error", "reason": reason}), 400

    # IK 求解 (本地坐标)
    sol = inverse_kinematics(lx, ly, lz, r, elbow_up=True)
    if not sol.valid:
        _set_twin_state("error", reason=sol.reason)
        return jsonify({"result": "error", "reason": sol.reason}), 400

    # FK→世界 TCP
    lp = fk_compact(sol.j1_deg, sol.j2_deg, sol.d3_mm, sol.j4_deg)
    wx, wy, wz = _local_to_world(lp.x, lp.y, lp.z, p)

    # 保存到 dobot_state.json
    from datetime import datetime
    joint = {"j1": sol.j1_deg, "j2": sol.j2_deg,
             "j3_deg": d3_to_j3_deg(sol.d3_mm),
             "j4": sol.j4_deg, "d3_mm": sol.d3_mm}
    _save_dobot_state({
        "joint": joint,
        "tcp": {"x": round(wx, 3), "y": round(wy, 3), "z": round(wz, 3), "r": r},
        "updated_at": datetime.now().isoformat(timespec='seconds'),
    })

    # 主动推 joint 事件 — 浏览器 SSE 立刻收到,触发动画
    _notify("joint", joint)
    # 同时推 tcp 事件 (世界坐标) — 前端 TCP 显示用
    _notify("tcp", {"x": round(wx, 3), "y": round(wy, 3), "z": round(wz, 3), "r": r})

    # 阻塞等前端动画完成后发 "done" 事件 (事件驱动,无 sleep)
    event = wait_event("done", timeout=10.0)
    _set_twin_state("idle")

    if event is None:
        return jsonify({"result": "done", "tcp": {"x": x, "y": y, "z": z, "r": r}, "anim": "timeout"})
    return jsonify({
        "result": "done",
        "tcp": {"x": x, "y": y, "z": z, "r": r},
        "anim_ms": int(event.get("duration", 0)),
    })


@app.route('/api/twin/publish/<topic>', methods=['POST'])
def twin_publish(topic):
    """浏览器发布消息到指定主题(模拟 MQTT publish)。"""
    if topic not in _topics:
        return jsonify({"ok": False, "error": f"unknown topic: {topic}"}), 400
    data = request.get_json(force=True, silent=True) or {}
    publish(topic, data)
    print(f"[publish:{topic}] ← {data}")
    return jsonify({"ok": True, "topic": topic, "subscribers": len(_topics[topic])})


@app.route('/api/twin/call_tool', methods=['POST'])
def twin_call_tool():
    """
    浏览器调用 hardware/tools 中的 Python 函数。
    前端输入 (220,-220,200,0) → 解析 → 调用 → 返回结果。
    直接执行碰撞检测 + IK + 状态保存,不走 HTTP 自引用。
    """
    data = request.get_json(force=True)
    name = data.get('name')
    args = data.get('args', [])

    if name != 'move_robot_arm':
        return jsonify({"result": f"不支持的工具: {name}", "status": "error"}), 400

    if len(args) < 4:
        return jsonify({"result": "参数不足,需 x,y,z,r", "status": "error"}), 400

    try:
        x, y, z, r = (float(v) for v in args[:4])
    except (ValueError, TypeError) as e:
        return jsonify({"result": f"参数转换失败: {e}", "status": "error"}), 400

    task_name = f"move:({x:.0f},{y:.0f},{z:.0f},{r:.0f})"

    # 世界坐标 → 本地 (Z=高度不变,只转 X-Y)
    p = _get_placement('dobot')
    lx, ly, lz = _world_to_local(x, y, z, p)

    # 碰撞检测 (本地坐标)
    from hardware.utils.collision import check_collision
    code, reason = check_collision({"x": lx, "y": ly, "z": lz, "r": r})
    if code != 200:
        _set_twin_state("error", reason=reason)
        return jsonify({"result": f"机械臂移动拒绝 [400]: {reason}", "status": "rejected"})

    # IK 求解 (本地坐标)
    sol = inverse_kinematics(lx, ly, lz, r, elbow_up=True)
    if not sol.valid:
        _set_twin_state("error", reason=sol.reason)
        return jsonify({"result": f"机械臂移动拒绝: {sol.reason}", "status": "rejected"})

    # FK→世界 TCP
    lp = fk_compact(sol.j1_deg, sol.j2_deg, sol.d3_mm, sol.j4_deg)
    wx, wy, wz = _local_to_world(lp.x, lp.y, lp.z, p)

    # 保存状态 (世界坐标)
    from datetime import datetime
    joint = {"j1": sol.j1_deg, "j2": sol.j2_deg,
             "j3_deg": d3_to_j3_deg(sol.d3_mm),
             "j4": sol.j4_deg, "d3_mm": sol.d3_mm}
    _save_dobot_state({
        "joint": joint,
        "tcp": {"x": round(wx, 3), "y": round(wy, 3), "z": round(wz, 3), "r": r},
        "updated_at": datetime.now().isoformat(timespec='seconds'),
    })

    # SSE 推送: status=busy + joint 事件
    _set_twin_state("busy", current_task=task_name)
    _notify("joint", joint)
    _notify("tcp", {"x": round(wx, 3), "y": round(wy, 3), "z": round(wz, 3), "r": r})

    # 阻塞等待前端动画完成后发 "done" 事件
    # 真实的握手,响应时间 = 动画实际时长
    event = wait_event("done", timeout=10.0)
    _set_twin_state("idle")

    if event is None:
        return jsonify({
            "result": f"机械臂已移动至坐标 ({x}, {y}, {z}, {r}, 0) [孪生] (动画超时 10s)",
            "status": "warn"
        })

    duration = event.get("duration", 0)
    return jsonify({
        "result": f"机械臂已移动至坐标 ({x}, {y}, {z}, {r}, 0) [孪生] (动画 {duration:.0f}ms)",
        "status": "ok"
    })


# -------------------------------------------------------
# 进程管理 — 启动杀旧 / 退出杀自己
# -------------------------------------------------------
import atexit
import subprocess

PORT = 5001


def kill_port(port):
    """杀掉监听指定端口的所有进程(Windows)。"""
    try:
        out = subprocess.run(
            ['netstat', '-ano', '-p', 'TCP'],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return 0
    killed = set()
    for line in out.splitlines():
        if f':{port}' not in line or 'LISTENING' not in line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        pid = parts[-1]
        if pid == str(os.getpid()):
            continue
        if pid in killed:
            continue
        killed.add(pid)
        try:
            subprocess.run(
                ['taskkill', '/F', '/PID', pid, '/T'],
                capture_output=True, timeout=5,
            )
        except Exception:
            pass
    return len(killed)


def kill_self_and_children():
    """退出时:杀当前进程 + 所有子进程(Flask reloader 也会被带走)。"""
    pid = os.getpid()
    try:
        subprocess.run(
            ['taskkill', '/F', '/PID', str(pid), '/T'],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass


atexit.register(kill_self_and_children)


# ============================================================
# 实时监听 — 持续读取 agent_client.twin_message_received
# 两条消息流(通过首字母区分,避免格式冲突):
#   - 机械臂:  "a{x},{y},{z},{r}"          (Cartesian 位置)
#   - 移液臂:  "d{x},{y},{lz},{rz}"        (x,y 平面 + 左右滴头 Z)
# ============================================================
_TWIN_POLL_INTERVAL = 0.05      # 20Hz

_twin_listener_thread: Optional[threading.Thread] = None
_twin_listener_stop = threading.Event()


def _parse_arm_message(payload: str) -> Optional[Tuple[float, float, float, float]]:
    """
    解析机械臂消息 "a{x},{y},{z},{r}" (Cartesian 位置)。
    返回 (x, y, z, r) 或 None(格式/数值不合法)。
    """
    if not payload or not payload.startswith('a'):
        return None
    rest = payload[1:]  # 去掉 'a' 前缀
    parts = re.split(r'\s*[,，]\s*', rest.strip())
    if len(parts) != 4:
        print(f"[TwinListener][arm] 格式错误,期望 4 段,实得 {len(parts)}: {payload!r}")
        return None
    try:
        x, y, z, r = (float(p) for p in parts)
    except ValueError as e:
        print(f"[TwinListener][arm] 数值解析失败: {payload!r} ({e})")
        return None
    return x, y, z, r


def _parse_pipette_message(payload: str) -> Optional[Tuple[float, float, float, float]]:
    """
    解析移液臂消息 "d{x},{y},{lz},{rz}" (x, y 平面 + 左右滴头 Z)。
    返回 (x, y, lz, rz) 或 None。
    """
    if not payload or not payload.startswith('d'):
        return None
    rest = payload[1:]  # 去掉 'd' 前缀
    parts = re.split(r'\s*[,，]\s*', rest.strip())
    if len(parts) != 4:
        print(f"[TwinListener][pipette] 格式错误,期望 4 段,实得 {len(parts)}: {payload!r}")
        return None
    try:
        x, y, lz, rz = (float(p) for p in parts)
    except ValueError as e:
        print(f"[TwinListener][pipette] 数值解析失败: {payload!r} ({e})")
        return None
    return x, y, lz, rz


def _handle_arm_message(payload: str):
    """处理机械臂消息:解析 → IK 求关节角 → 保存 → SSE 推送 joint/tcp。"""
    parsed = _parse_arm_message(payload)
    if parsed is None:
        return
    x, y, z, r = parsed

    # IK 求解关节角(实时映射是镜像,失败仅打印,不阻塞)
    sol = inverse_kinematics(x, y, z, r, elbow_up=True)
    if not sol.valid:
        print(f"[TwinListener][arm] IK 无效: {sol.reason}  payload={payload!r}")
        return

    # 持久化(用 IK 解得的关节角 + 原始 TCP)
    from datetime import datetime
    joint = {"j1": sol.j1_deg, "j2": sol.j2_deg,
             "j3_deg": d3_to_j3_deg(sol.d3_mm),
             "j4": sol.j4_deg, "d3_mm": sol.d3_mm}
    _save_dobot_state({
        "joint": joint,
        "tcp": {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3), "r": round(r, 3)},
        "updated_at": datetime.now().isoformat(timespec='seconds'),
    })

    # SSE 推送:
    #   rt_arm   — 实时(MQTT),无动画,适配 20Hz 高频流;前端 updateRobot() 直接跳
    #   tcp      — 文本标签(世界坐标),实时和工具调用通用
    #   joint    — 留给 /api/twin/execute / /api/twin/call_tool 的"动画"路径(自带 done 副作用,不适合实时流)
    _notify("rt_arm", {k: round(joint[k], 3) for k in ("j1", "j2", "j3_deg", "j4", "d3_mm")})
    _notify("tcp",    {"x": round(x, 3), "y": round(y, 3), "z": round(z, 3), "r": round(r, 3)})

    print(f"[TwinListener][arm] tcp=({x:.1f}, {y:.1f}, {z:.1f}, {r:.1f})  "
          f"joint=({joint['j1']:.1f}, {joint['j2']:.1f}, d3={joint['d3_mm']:.1f}, {joint['j4']:.1f})")


def _handle_pipette_message(payload: str):
    """处理移液臂消息:解析 → FK 算 tip 位置 → 保存 → SSE 推送 pipette 事件。"""
    parsed = _parse_pipette_message(payload)
    if parsed is None:
        return
    x, y, lz, rz = parsed

    # 构建 PipetteState 并 FK(纯平移关节,FK 即坐标变换)
    state = PipetteState(x=x, y=y, z1=lz, z2=rz)
    pose = pipette_fk(state)

    # 持久化
    from datetime import datetime
    _save_pipette_state({
        "axis": {"x": x, "y": y, "z1": lz, "z2": rz},
        "tip1": pose.tip1.to_dict(),
        "tip2": pose.tip2.to_dict(),
        "updated_at": datetime.now().isoformat(timespec='seconds'),
    })

    # SSE 推送 — rt_pipette(实时,无动画,适配 20Hz 高频流;前端 updatePipette() 直接跳)
    _notify("rt_pipette", {
        "axis": {"x": x, "y": y, "z1": lz, "z2": rz},
        "tip1": pose.tip1.to_dict(),
        "tip2": pose.tip2.to_dict(),
    })

    print(f"[TwinListener][pipette] axis=(x={x:.1f}, y={y:.1f}, z1={lz:.1f}, z2={rz:.1f})  "
          f"tip1=({pose.tip1.x:.1f}, {pose.tip1.y:.1f}, {pose.tip1.z:.1f})  "
          f"tip2=({pose.tip2.x:.1f}, {pose.tip2.y:.1f}, {pose.tip2.z:.1f})")


def _twin_listener_loop():
    """
    后台线程:订阅 'twin' 主题,根据首字母 'a'/'d' 分派到机械臂或移液臂处理。
    各自用 last_seen 跟踪避免重复处理(同一个值被 MQTT 重复推送时不会触发多次)。
    """
    from hardware.agent_client import MQTTConnector

    mqtt_client = MQTTConnector(client_id="digital_twin_listener")
    print(f"[TwinListener] 正在连接 MQTT broker ({mqtt_client.client_config.ip}:{mqtt_client.client_config.port})...")
    if not mqtt_client.connect(timeout=5):
        print(f"[TwinListener] MQTT 连接失败,线程退出")
        return
    print(f"[TwinListener] 已连接,开始监听 'twin' 主题 (poll={1.0/_TWIN_POLL_INTERVAL:.0f}Hz)")
    print(f"[TwinListener]   机械臂 → 'a{{x}},{{y}},{{z}},{{r}}'")
    print(f"[TwinListener]   移液臂 → 'd{{x}},{{y}},{{lz}},{{rz}}'")

    last_seen = {"arm": None, "pipette": None}

    while not _twin_listener_stop.is_set():
        try:
            payload = mqtt_client.get_twin_message()
            if payload is None:
                _twin_listener_stop.wait(timeout=_TWIN_POLL_INTERVAL)
                continue

            # [测试] 收到 twin 主题消息的原始 payload,确认订阅 + 解析是否正常
            print(f"[TwinListener] 收到 twin 消息 raw payload = {payload!r}")

            # 根据首字母分派(避免两路消息格式冲突)
            if payload.startswith('a'):
                if payload != last_seen["arm"]:
                    _handle_arm_message(payload)
                    last_seen["arm"] = payload
            elif payload.startswith('d'):
                if payload != last_seen["pipette"]:
                    _handle_pipette_message(payload)
                    last_seen["pipette"] = payload
            else:
                print(f"[TwinListener] 忽略未知前缀消息: {payload!r}")
        except Exception as e:
            print(f"[TwinListener] 异常: {e}")
            import traceback
            traceback.print_exc()
            _twin_listener_stop.wait(timeout=_TWIN_POLL_INTERVAL)

    try:
        mqtt_client.disconnect()
    except Exception:
        pass
    print(f"[TwinListener] 已停止")


def start_twin_listener():
    """启动后台监听线程(幂等)。"""
    global _twin_listener_thread
    if _twin_listener_thread and _twin_listener_thread.is_alive():
        return
    _twin_listener_stop.clear()
    _twin_listener_thread = threading.Thread(
        target=_twin_listener_loop, daemon=True, name="twin-listener"
    )
    _twin_listener_thread.start()


def stop_twin_listener():
    _twin_listener_stop.set()


# 暴露给外部按需启停(已通过 /api/twin/listener/* 端点暴露,__main__ 也会默认启动)
def is_twin_listener_running() -> bool:
    return _twin_listener_thread is not None and _twin_listener_thread.is_alive()


# -------------------------------------------------------
# Start
# -------------------------------------------------------

if __name__ == '__main__':
    # 启动时清掉所有 5001 上的旧进程
    n = kill_port(PORT)
    if n:
        print(f"[CleanUp] Killed {n} stale process(es) on :{PORT}")
    time.sleep(0.5)

    # 默认启动实时关节角监听
    start_twin_listener()

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
    print(f"  Ctrl+C to stop (auto-cleanup)")

    try:
        app.run(host='0.0.0.0', port=PORT, debug=False)
    except KeyboardInterrupt:
        print(f"\n[CleanUp] KeyboardInterrupt, killing self + children")
        kill_self_and_children()
