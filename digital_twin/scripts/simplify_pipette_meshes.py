"""
Simplify pipette arm STL meshes using vertex clustering.
Reduces ~1.4M total tris → ~26K for web display.
Pure numpy + struct, zero extra dependencies.
"""
import struct
import os
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates', 'models', 'pipette_arm')

FILES = [
    ('pipette_group1.stl', 8000),
    ('pipette_group2.stl', 6000),
    ('pipette_group3.stl', 5000),
    ('pipette_group4.stl', 4000),
    ('pipette_group5.stl', 2500),
    ('pipette_tip.stl',    500),
]


def read_stl_binary(filepath):
    """Read binary STL, return (vertices_array_Nx9, triangle_count)."""
    with open(filepath, 'rb') as f:
        f.read(80)  # header
        tri_count = struct.unpack('<I', f.read(4))[0]
        data = f.read(tri_count * 50)

    tris = np.zeros((tri_count, 9), dtype=np.float32)
    for i in range(tri_count):
        offset = i * 50
        vals = struct.unpack_from('<12f', data, offset)
        tris[i, 0:3] = vals[3:6]   # v1
        tris[i, 3:6] = vals[6:9]   # v2
        tris[i, 6:9] = vals[9:12]  # v3
    return tris, tri_count


def write_stl_binary(filepath, tris):
    """Write binary STL from triangles array Nx9."""
    tri_count = len(tris)
    with open(filepath, 'wb') as f:
        f.write(b'\x00' * 80)
        f.write(struct.pack('<I', tri_count))
        for i in range(tri_count):
            v1, v2, v3 = tris[i, 0:3], tris[i, 3:6], tris[i, 6:9]
            # Compute face normal
            e1 = v2 - v1
            e2 = v3 - v1
            normal = np.cross(e1, e2)
            nrm = np.linalg.norm(normal)
            if nrm > 1e-10:
                normal = normal / nrm
            f.write(struct.pack('<12f', *normal, *v1, *v2, *v3))
            f.write(b'\x00\x00')


def vertex_clustering_simplify(tris, target_tris):
    """Simplify mesh using vertex clustering on a uniform grid.

    1. Partition bounding box into 3D grid cells
    2. Merge all vertices in each cell to their centroid
    3. Remove degenerate triangles (≥2 vertices in same cell)
    4. Iterate with finer grid until target is reached
    """
    if len(tris) <= target_tris:
        return tris

    verts = tris.reshape(-1, 3)  # (N*3, 3) — all vertices
    bbox_min = verts.min(axis=0)
    bbox_max = verts.max(axis=0)
    size = bbox_max - bbox_min

    # Start coarse, refine if needed
    current = tris
    for iteration in range(5):
        if len(current) <= target_tris:
            break

        v = current.reshape(-1, 3)
        bmin = v.min(axis=0)
        bmax = v.max(axis=0)
        sz = bmax - bmin

        # Grid: target about target_tris cells
        cells_per_axis = max(3, int(np.cbrt(target_tris * 2)))
        csize = np.maximum(sz / cells_per_axis, 1e-4)

        # Cell index per vertex
        cidx = np.floor((v - bmin) / csize).astype(np.int32)
        for d in range(3):
            mx = max(0, int(np.ceil(sz[d] / csize[d])) - 1)
            cidx[:, d] = np.clip(cidx[:, d], 0, mx)

        # Hash cell indices
        maxc = np.max(cidx, axis=0).astype(np.int64) + 1
        keys = (cidx[:, 0].astype(np.int64) * maxc[1] * maxc[2] +
                cidx[:, 1].astype(np.int64) * maxc[2] +
                cidx[:, 2].astype(np.int64))

        _, inverse = np.unique(keys, return_inverse=True)
        n_cells = inverse.max() + 1

        # Centroid per cell
        new_v = np.zeros((n_cells, 3), dtype=np.float64)
        np.add.at(new_v, inverse, v.astype(np.float64))
        cnt = np.bincount(inverse, minlength=n_cells)
        new_v /= cnt[:, np.newaxis]
        new_v = new_v.astype(np.float32)

        # Map triangle vertices: each tri has 3 vertices at positions [i*3, i*3+1, i*3+2]
        v1_cell = inverse[0::3]  # cell index of 1st vertex of each triangle
        v2_cell = inverse[1::3]  # cell index of 2nd vertex
        v3_cell = inverse[2::3]  # cell index of 3rd vertex

        valid = (v1_cell != v2_cell) & (v1_cell != v3_cell) & (v2_cell != v3_cell)
        n_valid = valid.sum()

        new_tris = np.zeros((n_valid, 9), dtype=np.float32)
        new_tris[:, 0:3] = new_v[v1_cell[valid]]
        new_tris[:, 3:6] = new_v[v2_cell[valid]]
        new_tris[:, 6:9] = new_v[v3_cell[valid]]

        current = new_tris

    return current


def distribute_budget(file_targets, total_original):
    """Distribute simplification budget proportionally, with floor per file."""
    # Already specified in FILES list
    return


def simplify_all():
    """Main entry: simplify all pipette STL files."""
    output_dir = os.path.join(MODELS_DIR, 'simplified')
    os.makedirs(output_dir, exist_ok=True)

    total_orig = 0
    total_new = 0

    for filename, target in FILES:
        fpath = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(fpath):
            print(f'[SKIP] {filename} — not found')
            continue

        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        tris, count = read_stl_binary(fpath)
        total_orig += count
        print(f'[{filename}] {count:,} tris, {size_mb:.1f}MB → target {target:,} tris')

        simplified = vertex_clustering_simplify(tris, target)
        new_count = len(simplified)
        total_new += new_count

        out_path = os.path.join(output_dir, filename)
        write_stl_binary(out_path, simplified)
        out_size = os.path.getsize(out_path) / 1024
        ratio = (1 - new_count / count) * 100 if count > 0 else 0
        print(f'  → {new_count:,} tris ({ratio:.1f}% reduction), {out_size:.1f}KB')

    print(f'\nDone: {total_orig:,} → {total_new:,} tris ({total_new/total_orig*100:.1f}%)')
    return output_dir


if __name__ == '__main__':
    simplify_all()
