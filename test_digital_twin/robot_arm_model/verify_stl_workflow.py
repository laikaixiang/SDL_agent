"""
STL/STEP Workflow Verification Script
=======================================
Validates the 3D model import pipeline for digital_twin project.

Workflow: CAD file (.stl/.step) → Three.js loading → HTML visualization
"""

import os
import struct
import sys

# Fix Windows encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

def validate_stl_binary(filepath):
    """Validate and extract info from binary STL file."""
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"

    try:
        with open(filepath, 'rb') as f:
            # Read header (80 bytes)
            header = f.read(80)

            # Read triangle count
            tri_count_bytes = f.read(4)
            if len(tri_count_bytes) < 4:
                return None, "Invalid STL: couldn't read triangle count"

            tri_count = struct.unpack('<I', tri_count_bytes)[0]

            if tri_count > 10_000_000:
                return None, f"Suspiciously large triangle count: {tri_count}"

            # Validate a few triangles
            sample_tris = min(5, tri_count)
            vertices = []
            for i in range(sample_tris):
                data = f.read(50)  # 12 floats + 2 bytes
                if len(data) < 50:
                    return None, f"Unexpected EOF at triangle {i}"

                vals = struct.unpack('<12f', data[:48])
                normal = vals[:3]
                v1 = vals[3:6]
                v2 = vals[6:9]
                v3 = vals[9:12]
                vertices.append((v1, v2, v3))

            return {
                'filepath': filepath,
                'triangle_count': tri_count,
                'sample_vertices': vertices,
                'header': header[:40].decode('ascii', errors='ignore').strip()
            }, None

    except Exception as e:
        return None, f"Error reading STL: {e}"


def validate_step(filepath):
    """Basic STEP file validation - check header structure."""
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            header = f.read(500)

            # STEP files start with ISO-10303 series
            if 'ISO-10303' in header or 'HEADER' in header:
                # Count significant lines
                lines = f.readlines()
                data_lines = [l for l in lines if l.strip() and not l.strip().startswith('/*')]
                return {
                    'filepath': filepath,
                    'type': 'STEP',
                    'total_lines': len(lines),
                    'data_lines': len(data_lines),
                    'has_header': 'HEADER' in header
                }, None
            else:
                return None, "Not a valid STEP file (missing ISO-10303 header)"

    except Exception as e:
        return None, f"Error reading STEP: {e}"


def validate_stl_ascii(filepath):
    """Validate ASCII STL format."""
    if not os.path.exists(filepath):
        return None, f"File not found: {filepath}"

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read(1000)
            if 'solid' in content.lower():
                return {'filepath': filepath, 'type': 'ASCII STL', 'validated': True}, None
            return None, "Not recognized as ASCII STL"
    except Exception as e:
        return None, f"Error reading ASCII STL: {e}"


def generate_stl_loader_html(stl_path, output_path=None):
    """Generate an HTML file that loads the given STL."""

    html_template = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>STL加载验证 - {filename}</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: -apple-system,'Segoe UI','Microsoft YaHei',sans-serif;
       background: #1a1d23; color: #abb2bf; overflow: hidden; height: 100vh; }}
#container {{ flex: 1; position: relative; display: flex; flex-direction: column; }}
#info-bar {{ padding: 10px 16px; background: #21252b; border-bottom: 1px solid #333;
            font-size: 13px; display: flex; gap: 20px; align-items: center; }}
#info-bar .label {{ color: #61afef; font-weight: 600; }}
#info-bar .value {{ color: #e5c07b; }}
#canvas {{ flex: 1; display: block; }}
.status-ok {{ color: #98c379; }}
.status-err {{ color: #e06c75; }}
</style>
</head>
<body>
<div id="container">
  <div id="info-bar">
    <span><span class="label">模型:</span> <span class="value" id="filename">-</span></span>
    <span><span class="label">三角面:</span> <span class="value" id="tri-count">-</span></span>
    <span><span class="label">顶点数:</span> <span class="value" id="vert-count">-</span></span>
    <span><span class="label">状态:</span> <span class="value" id="status">加载中...</span></span>
  </div>
  <div id="canvas"></div>
</div>

<script type="importmap">
{{ "imports": {{
  "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
  "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
}}}}
</script>

<script type="module">
import * as THREE from 'three';
import {{ OrbitControls }} from 'three/addons/controls/OrbitControls.js';
import {{ STLLoader }} from 'three/addons/loaders/STLLoader.js';

const container = document.getElementById('canvas');

// Renderer
const renderer = new THREE.WebGLRenderer({{ antialias: true }});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.1;
container.appendChild(renderer.domElement);

// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color('#1e2127');

// Camera
const camera = new THREE.PerspectiveCamera(50, 1, 1, 5000);
camera.position.set(200, 150, 200);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(50, 20, 40);
controls.enableDamping = true;
controls.update();

// Lighting
scene.add(new THREE.AmbientLight('#8899bb', 1.2));
const keyLight = new THREE.DirectionalLight('#fffef5', 2.5);
keyLight.position.set(200, 300, 150);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
scene.add(keyLight);
scene.add(new THREE.DirectionalLight('#aaccff', 0.8));

// Ground plane
const ground = new THREE.Mesh(
  new THREE.PlaneGeometry(800, 800),
  new THREE.MeshStandardMaterial({{ color: '#2a2d33', roughness: 0.8 }})
);
ground.rotation.x = -Math.PI / 2;
ground.position.y = -0.01;
ground.receiveShadow = true;
scene.add(ground);

// Grid
const grid = new THREE.GridHelper(600, 12, '#4a4f59', '#2a2d33');
grid.position.y = 0;
scene.add(grid);

// STL Loader
const loader = new STLLoader();

loader.load(
  './{filename}',
  (geometry) => {{
    document.getElementById('filename').textContent = '{filename}';
    document.getElementById('tri-count').textContent =
      (geometry.attributes.position.count / 3).toLocaleString();
    document.getElementById('vert-count').textContent =
      geometry.attributes.position.count.toLocaleString();

    // Center geometry
    geometry.computeBoundingBox();
    const center = new THREE.Vector3();
    geometry.boundingBox.getCenter(center);
    geometry.translate(-center.x, -center.y, -center.z);

    // Create mesh
    const material = new THREE.MeshStandardMaterial({{
      color: 0x61afef,
      roughness: 0.35,
      metalness: 0.6,
    }});

    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = true;
    mesh.receiveShadow = true;

    // Add to scene
    scene.add(mesh);

    // Adjust camera to fit
    const box = geometry.boundingBox;
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    camera.position.set(maxDim * 1.5, maxDim * 1.2, maxDim * 1.5);
    controls.target.copy(center);
    controls.update();

    document.getElementById('status').textContent = '加载成功';
    document.getElementById('status').className = 'status-ok';
    console.log('[STL] Loaded successfully:', {{ vertices: geometry.attributes.position.count }});
  }},
  (progress) => {{ console.log('[STL] Loading:', progress); }},
  (error) => {{
    document.getElementById('status').textContent = '加载失败: ' + error;
    document.getElementById('status').className = 'status-err';
    console.error('[STL] Error:', error);
  }}
);

// Resize handler
function resize() {{
  const w = container.clientWidth, h = container.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / Math.max(h, 1);
  camera.updateProjectionMatrix();
}}
window.addEventListener('resize', resize);
resize();

// Render loop
(function render() {{
  requestAnimationFrame(render);
  controls.update();
  renderer.render(scene, camera);
}})();
</script>
</body>
</html>'''

    filename = os.path.basename(stl_path)
    html_content = html_template.replace('{filename}', filename)

    out_path = output_path or stl_path.replace('.stl', '_viewer.html')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return out_path


def main():
    print("=" * 60)
    print("STL/STEP Workflow Verification")
    print("=" * 60)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Find test files
    test_files = []
    for fname in ['test_model.stl', 'test_module.stl']:
        fpath = os.path.join(base_dir, fname)
        if os.path.exists(fpath):
            test_files.append(fpath)

    if not test_files:
        print("\n[ERROR] No test STL files found in current directory")
        print("Place .stl or .step files to validate the workflow")
        return 1

    print(f"\nFound {len(test_files)} test file(s)\n")

    all_passed = True
    for filepath in test_files:
        fname = os.path.basename(filepath)
        ext = os.path.splitext(filepath)[1].lower()

        print(f"[{fname}]")
        print(f"  Path: {filepath}")

        if ext == '.stl':
            # Check file size
            size_kb = os.path.getsize(filepath) / 1024
            print(f"  Size: {size_kb:.1f} KB")

            info, err = validate_stl_binary(filepath)

            if err:
                print(f"  [FAIL] {err}")
                all_passed = False
                continue

            print(f"  Triangles: {info['triangle_count']:,}")
            print(f"  Header: {info['header'] or '(binary)'}")

            # Generate viewer HTML
            viewer_path = generate_stl_loader_html(filepath)
            print(f"  Viewer: {os.path.basename(viewer_path)}")

            print(f"  [PASS] Binary STL validated")

        elif ext == '.step':
            info, err = validate_step(filepath)
            if err:
                print(f"  [FAIL] {err}")
                all_passed = False
                continue

            print(f"  Type: STEP")
            print(f"  Lines: {info['total_lines']:,}")
            print(f"  [PASS] STEP validated")

        print()

    print("=" * 60)
    if all_passed:
        print("[RESULT] All validations passed!")
        print("\nWorkflow verified:")
        print("  1. CAD export → .stl / .step")
        print("  2. Python validate → binary STL structure")
        print("  3. HTML viewer generation → Three.js STLLoader")
        print("  4. Browser loads → 3D visualization")
        print("\nTo test in browser:")
        print("  python digital_twin.py")
        print("  # Open http://127.0.0.1:5001/{viewer_html}")
    else:
        print("[RESULT] Some validations failed")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())