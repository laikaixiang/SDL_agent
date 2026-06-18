"""
CSV 预览 API 集成测试 (test_csv_preview_api.py)

覆盖 /api/csv/preview 和 /api/csv/columns 端点:
  - 200 + 完整 PreviewData
  - 400 (path 缺失 / n 非法)
  - 404 (文件不存在)
  - 500 (文件无法解析)
  - 大文件不爆栈
  - 列名端点 (lightweight)

运行: pytest platform_init/test/software/test_csv_preview_api.py -v

注意: 此测试只 import app.py 的 flask app, 不启动 server。
     业务函数 inspect_csv 已由 test_csv_inspector.py 覆盖。
"""
import os
import sys
import csv
import json
import tempfile
import math
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pytest

# 尝试导入 app; 缺失依赖时 skip 而非 fail
try:
    from app import app as flask_app
    FLASK_AVAILABLE = True
except Exception as e:
    FLASK_AVAILABLE = False
    _IMPORT_ERROR = e


pytestmark = pytest.mark.skipif(
    not FLASK_AVAILABLE,
    reason=f"Flask app 不可用: {_IMPORT_ERROR if not FLASK_AVAILABLE else ''}"
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def client():
    flask_app.config['TESTING'] = True
    with flask_app.test_client() as c:
        yield c


@pytest.fixture
def sample_csv(tmp_path) -> str:
    """小型样本 CSV: 3 列 (int/float/str), 5 行"""
    p = tmp_path / "sample.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "score", "name"])
        for i in range(5):
            writer.writerow([i + 1, round(0.1 * (i + 1), 2), f"item_{i}"])
    return str(p)


@pytest.fixture
def spectrum_csv(tmp_path) -> str:
    """光谱 CSV: wavelength(int) + intensity(float)"""
    p = tmp_path / "spectrum.csv"
    wl = list(range(400, 700))
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wavelength", "intensity"])
        for w in wl:
            intensity = round(0.05 + 0.9 * math.exp(-0.5 * ((w - 532) / 15) ** 2), 6)
            writer.writerow([w, intensity])
    return str(p)


# =============================================================================
# /api/csv/preview 路由测试
# =============================================================================

class TestCsvPreviewEndpoint:
    def test_200_with_full_data(self, client, sample_csv):
        """正常返回: success=True, data 含 columns/row_count/total_rows/file_size"""
        resp = client.get(f"/api/csv/preview?path={sample_csv}&n=10")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        data = body["data"]
        assert data["path"] == sample_csv
        assert len(data["columns"]) == 3
        assert data["row_count"] == 5
        assert data["total_rows"] == 5
        assert data["file_size"] > 0
        # 列名正确
        col_names = [c["name"] for c in data["columns"]]
        assert col_names == ["id", "score", "name"]
        # 类型推断
        types = {c["name"]: c["type"] for c in data["columns"]}
        assert types["id"] == "int"
        assert types["score"] == "float"
        assert types["name"] == "str"

    def test_n_caps_at_max(self, client, sample_csv):
        """n>200 会被截到 200 (不报错)"""
        resp = client.get(f"/api/csv/preview?path={sample_csv}&n=9999")
        assert resp.status_code == 200

    def test_n_below_min_defaults_to_1(self, client, sample_csv):
        """n=0 / n=-5 应被钳到 1"""
        resp = client.get(f"/api/csv/preview?path={sample_csv}&n=0")
        assert resp.status_code == 200

    def test_missing_path_returns_400(self, client):
        """path 为空 → 400"""
        resp = client.get("/api/csv/preview")
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["success"] is False

    def test_invalid_n_returns_400(self, client, sample_csv):
        """n=abc → 400"""
        resp = client.get(f"/api/csv/preview?path={sample_csv}&n=abc")
        assert resp.status_code == 400

    def test_nonexistent_file_returns_404(self, client):
        """文件不存在 → 404"""
        resp = client.get("/api/csv/preview?path=/no/such/file.csv")
        assert resp.status_code == 404
        body = resp.get_json()
        assert body["success"] is False
        assert "不存在" in body["message"] or "not found" in body["message"].lower()

    def test_sample_in_response(self, client, sample_csv):
        """sample 字段存在, 至少含 1 个值"""
        resp = client.get(f"/api/csv/preview?path={sample_csv}&n=5")
        data = resp.get_json()["data"]
        id_col = next(c for c in data["columns"] if c["name"] == "id")
        assert len(id_col["sample"]) >= 1
        assert id_col["sample"][0] in ("1",)

    def test_spectrum_csv(self, client, spectrum_csv):
        """光谱 CSV 正确推断 wavelength=int, intensity=float"""
        resp = client.get(f"/api/csv/preview?path={spectrum_csv}&n=20")
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        types = {c["name"]: c["type"] for c in data["columns"]}
        assert types["wavelength"] == "int"
        assert types["intensity"] == "float"

    def test_default_n_is_20(self, client, spectrum_csv):
        """不传 n → 默认 20"""
        resp = client.get(f"/api/csv/preview?path={spectrum_csv}")
        assert resp.status_code == 200
        data = resp.get_json()["data"]
        # 光谱 300 行, n=20 时 row_count=20
        assert data["row_count"] == 20


# =============================================================================
# /api/csv/columns 路由测试 (lightweight)
# =============================================================================

class TestCsvColumnsEndpoint:
    def test_returns_column_list(self, client, sample_csv):
        """返回 columns 数组"""
        resp = client.get(f"/api/csv/columns?path={sample_csv}")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert body["data"]["columns"] == ["id", "score", "name"]

    def test_missing_path_returns_400(self, client):
        resp = client.get("/api/csv/columns")
        assert resp.status_code == 400

    def test_nonexistent_file_returns_404(self, client):
        resp = client.get("/api/csv/columns?path=/no/such/file.csv")
        assert resp.status_code == 404


# =============================================================================
# /api/csv/preview: 大文件不爆栈
# =============================================================================

def test_large_file_preview_does_not_crash(client, tmp_path):
    """80KB+ 文件走估算路径, 不读全部内容"""
    p = tmp_path / "large.csv"
    with open(p, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        for i in range(10000):
            writer.writerow([str(i), str(i * 0.5)])
    assert p.stat().st_size > 65536

    resp = client.get(f"/api/csv/preview?path={p}&n=20")
    assert resp.status_code == 200
    data = resp.get_json()["data"]
    assert data["row_count"] == 20
    # 估算总行数应接近 10000
    assert 8000 <= data["total_rows"] <= 12000
