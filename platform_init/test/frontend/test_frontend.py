"""
前端重构集成测试

验证:
  1. Vite 构建产物存在且正确
  2. Flask 正确服务新旧两个前端
  3. 所有 SPA 路由返回 index.html
  4. API 路由不受影响
  5. 静态资源可访问
  6. Phase 4 语义搜索 API 正常

运行方法: python platform_init/test/frontend/test_frontend.py
"""
import sys
import io
import os
import json
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app import app as flask_app


def test_old_ui_available():
    """旧版 UI / 可访问"""
    print("\n=== test_old_ui_available ===")
    with flask_app.test_client() as client:
        resp = client.get('/')
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        html = resp.data.decode('utf-8', errors='replace')
        assert 'app-wrapper' in html or 'SDL' in html, "Old UI should contain familiar content"
    print("PASS")


def test_new_ui_v2_root():
    """新版 UI /v2 返回 Vue SPA"""
    print("\n=== test_new_ui_v2_root ===")
    with flask_app.test_client() as client:
        resp = client.get('/v2')
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
        html = resp.data.decode('utf-8')
        assert '<div id="app">' in html, "Must contain Vue mount point"
        assert '/v2-static/assets/' in html, "Must reference built assets"
    print("PASS")


def test_new_ui_v2_routes():
    """所有 /v2/* SPA 路由返回同一个 index.html"""
    print("\n=== test_new_ui_v2_routes ===")
    routes = ['/v2/search', '/v2/extraction', '/v2/hardware', '/v2/analysis', '/v2/experiment']
    with flask_app.test_client() as client:
        for route in routes:
            resp = client.get(route)
            assert resp.status_code == 200, f"{route}: Expected 200, got {resp.status_code}"
            assert b'<div id="app">' in resp.data, f"{route}: Must contain Vue mount point"
    print(f"  {len(routes)} routes all return 200 + Vue mount — PASS")


def test_api_unaffected():
    """API 路由不受 SPA 影响"""
    print("\n=== test_api_unaffected ===")
    with flask_app.test_client() as client:
        # Test chat endpoint
        resp = client.post('/api/chat', data=json.dumps({'message': 'hello'}),
                           content_type='application/json')
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

        # Test list algorithms
        resp = client.get('/api/list_algorithms')
        assert resp.status_code == 200

        # Test hardware tools
        resp = client.get('/api/hardware_tools')
        assert resp.status_code == 200

        # Test hardware status
        resp = client.get('/api/hardware_status')
        assert resp.status_code == 200
    print("  /api/chat, /api/list_algorithms, /api/hardware_tools, /api/hardware_status — PASS")


def test_static_assets():
    """Vite 构建的静态资源可访问"""
    print("\n=== test_static_assets ===")
    # Check dist/index.html exists (now in frontend/dist/)
    dist_index = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                              'frontend', 'dist', 'index.html')
    assert os.path.exists(dist_index), f"dist/index.html not found at {dist_index}"
    assert os.path.getsize(dist_index) > 100, "dist/index.html is too small"

    # Check dist/assets directory exists
    dist_assets = os.path.join(os.path.dirname(dist_index), 'assets')
    assert os.path.exists(dist_assets), "dist/assets directory not found"

    # Check at least one JS bundle exists
    js_files = [f for f in os.listdir(dist_assets) if f.endswith('.js')]
    assert len(js_files) > 0, "No JS files found in dist/assets"
    print(f"  dist/index.html ({os.path.getsize(dist_index)}B), {len(js_files)} JS bundles — PASS")


def test_semantic_search_api():
    """Phase 4 语义搜索 API"""
    print("\n=== test_semantic_search_api ===")
    with flask_app.test_client() as client:
        resp = client.post('/api/semantic_search',
                           data=json.dumps({'query': 'perovskite passivation', 'top_k': 3}),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] is True, f"Expected success=True, got {data}"
        assert 'results' in data, "Response must contain 'results'"
        assert 'total_pages_indexed' in data, "Response must contain 'total_pages_indexed'"
        print(f"  Query OK, {data['total_pages_indexed']} pages indexed, {len(data['results'])} results — PASS")


def test_page_image_api():
    """Phase 4 页面图片 API — 参数验证"""
    print("\n=== test_page_image_api ===")
    with flask_app.test_client() as client:
        # Missing pdf_path should return 400
        resp = client.post('/api/page_image',
                           data=json.dumps({'page_num': 1}),
                           content_type='application/json')
        assert resp.status_code == 400, f"Expected 400 for missing pdf_path, got {resp.status_code}"

        # Negative page_num should return 400
        resp = client.post('/api/page_image',
                           data=json.dumps({'pdf_path': 'test.pdf', 'page_num': -1}),
                           content_type='application/json')
        assert resp.status_code == 400, f"Expected 400 for invalid page_num, got {resp.status_code}"

        # Missing file should return 404
        resp = client.post('/api/page_image',
                           data=json.dumps({'pdf_path': 'nonexistent.pdf', 'page_num': 1}),
                           content_type='application/json')
        assert resp.status_code == 404, f"Expected 404 for missing file, got {resp.status_code}"
    print("  400 (bad params), 400 (invalid page), 404 (not found) — PASS")


def test_extraction_empty_input_defaults():
    """文献提取无输入时使用默认配置"""
    print("\n=== test_extraction_empty_input_defaults ===")
    with flask_app.test_client() as client:
        # Empty description should trigger default extraction
        resp = client.post('/api/chat',
                           data=json.dumps({'message': '帮我搜寻：', 'action': ''}),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        # Empty input → direct task_trigger with default FAPbI3 config
        assert data['type'] == 'task_trigger', \
            f"Expected task_trigger for empty input, got {data['type']}"
        assert 'FAPbI3' in data['reply'] or '已启动' in data['reply'], \
            f"Reply should mention defaults, got: {data['reply']}"
    print("  Empty input → default FAPbI3 extraction triggered — PASS")


def test_extraction_with_description_endpoint():
    """文献提取带描述时后端正确响应"""
    print("\n=== test_extraction_with_description_endpoint ===")
    with flask_app.test_client() as client:
        resp = client.post('/api/chat',
                           data=json.dumps({'message': '帮我搜寻：提取钙钛矿钝化剂参数'}),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        # Either field_confirm (LLM success) or system (LLM fallback)
        assert data['type'] in ('field_confirm', 'system'), \
            f"Expected field_confirm or system, got {data['type']}"
        if data['type'] == 'field_confirm':
            assert 'fields' in data, "field_confirm must contain fields"
            assert len(data['fields']) > 0, "Fields list must not be empty"
            print(f"  Description → field_confirm with {len(data['fields'])} fields — PASS")
        else:
            print(f"  Description → system (LLM fallback, expected in offline env) — PASS")


def test_build_integrity():
    """构建产物完整性"""
    print("\n=== test_build_integrity ===")
    dist_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                            'frontend', 'dist')
    assets_dir = os.path.join(dist_dir, 'assets')

    # Count all page JS chunks (lazy-loaded routes)
    js_files = os.listdir(assets_dir)
    page_chunks = [f for f in js_files if f.endswith('.js') and not f.startswith('index-')]
    css_files = [f for f in js_files if f.endswith('.css')]

    # Must have at least main bundle + 6 page chunks
    assert any(f.startswith('index-') and f.endswith('.js') for f in js_files), "Main JS bundle missing"
    assert len(page_chunks) >= 6, f"Expected >=6 page chunks, got {len(page_chunks)}"
    assert len(css_files) >= 2, f"Expected >=2 CSS files, got {len(css_files)}"

    print(f"  Main bundle + {len(page_chunks)} page chunks + {len(css_files)} CSS files — PASS")


if __name__ == "__main__":
    tests = [
        test_old_ui_available,
        test_new_ui_v2_root,
        test_new_ui_v2_routes,
        test_api_unaffected,
        test_static_assets,
        test_semantic_search_api,
        test_page_image_api,
        test_extraction_empty_input_defaults,
        test_extraction_with_description_endpoint,
        test_build_integrity,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*50}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*50}")
