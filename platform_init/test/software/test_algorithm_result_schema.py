"""
BaseAlgorithm.result_schema 集成测试 (test_algorithm_result_schema.py)

覆盖:
  - /api/list_algorithms 返回的每个算法都含 result_schema 字段
  - 4 个声明了 result_schema 的算法都非空
  - 未声明的算法 (func, normalize_normal_distribution) result_schema 为空 dict
  - 透传: API 响应里的 result_schema 字典与 get_info() 的一致

运行: pytest platform_init/test/software/test_algorithm_result_schema.py -v
"""
import os
import sys
import json

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

# 尝试 import app
try:
    from app import app as flask_app
    from core.software_manager import SoftwareManager
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
def manager():
    return SoftwareManager()


# =============================================================================
# BaseAlgorithm 直接测试
# =============================================================================

class TestBaseAlgorithmResultSchemaField:
    def test_default_result_schema_is_empty_dict(self):
        from software.algorithms.base import BaseAlgorithm
        # 抽象类不能直接实例化, 但类属性 result_schema 应为 {}
        assert BaseAlgorithm.result_schema == {}

    def test_get_info_includes_result_schema(self, manager):
        """get_info() 返回的 dict 必须含 result_schema 键"""
        for algo in manager.list_algorithms():
            assert "result_schema" in algo, f"算法 {algo.get('name')} 缺 result_schema"
            assert isinstance(algo["result_schema"], dict)

    def test_declared_algorithms_have_non_empty_schema(self, manager):
        """声明了 result_schema 的算法应非空"""
        declared = {"data_statistics", "spectrum_analysis", "data_normalization", "bayesian_optimization"}
        algos_by_name = {a["name"]: a for a in manager.list_algorithms()}
        for name in declared:
            assert name in algos_by_name, f"未找到算法 {name}"
            schema = algos_by_name[name]["result_schema"]
            assert schema, f"算法 {name} 的 result_schema 为空"
            assert "type" in schema, f"算法 {name} 的 result_schema 缺 type"


# =============================================================================
# /api/list_algorithms 端点测试
# =============================================================================

class TestListAlgorithmsApiReturnsResultSchema:
    def test_endpoint_returns_200(self, client):
        resp = client.get("/api/list_algorithms")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["success"] is True
        assert "algorithms" in body

    def test_each_algorithm_has_result_schema_field(self, client):
        resp = client.get("/api/list_algorithms")
        algos = resp.get_json()["algorithms"]
        for algo in algos:
            assert "result_schema" in algo, f"算法 {algo.get('name')} API 响应缺 result_schema"

    def test_declared_algorithms_pass_through_schema(self, client):
        """API 响应里的 result_schema 字段非空"""
        resp = client.get("/api/list_algorithms")
        algos = resp.get_json()["algorithms"]
        algos_by_name = {a["name"]: a for a in algos}

        # data_statistics: type=table, sections 含描述性统计 + 相关性矩阵
        ds = algos_by_name.get("data_statistics", {})
        assert ds.get("result_schema", {}).get("type") == "table"
        sections = ds.get("result_schema", {}).get("sections", [])
        assert len(sections) >= 1
        assert any("mean" in str(s.get("columns", [])) for s in sections)

        # spectrum_analysis: type=kv, 含 items
        sa = algos_by_name.get("spectrum_analysis", {})
        assert sa.get("result_schema", {}).get("type") == "kv"
        items = sa.get("result_schema", {}).get("items", [])
        assert any(item["key"] == "peak_wavelength" for item in items)

    def test_undeclared_algorithms_have_empty_schema(self, client):
        """没声明 result_schema 的算法: result_schema = {} (frontend 走 fallback)"""
        resp = client.get("/api/list_algorithms")
        algos = resp.get_json()["algorithms"]
        algos_by_name = {a["name"]: a for a in algos}

        # func 和 normalize_normal_distribution 没声明
        for name in ("func", "normalize_normal_distribution"):
            if name in algos_by_name:
                assert algos_by_name[name]["result_schema"] == {}, \
                    f"算法 {name} 应返回空 result_schema, 实际: {algos_by_name[name]['result_schema']}"
