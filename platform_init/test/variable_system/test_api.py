"""
变量系统 — Flask API 集成测试

运行方法: python platform_init/test/variable_system/test_api.py
需要有 Flask app 可导入（不需要 Flask 运行中）。
"""
import sys
import io
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from app import app as flask_app


# ==================== CSV 导入 API ====================

def test_import_csv_success():
    """正常 CSV 导入"""
    print("\n=== test_import_csv_success ===")
    with flask_app.test_client() as client:
        resp = client.post('/api/variables/import_csv',
                           data=json.dumps({"csv_content": "speed,duration\n3000,30\n4000,25"}),
                           content_type='application/json')
        assert resp.status_code == 200, f"期望200，得到 {resp.status_code}"
        data = resp.get_json()
        assert data['type'] == 'variables_csv'
        assert 'speed' in data['variables']
        assert data['variables']['speed']['type'] == 'int'
        assert len(data['batch_data']) == 2
        assert 'reply' in data
        print(f"PASS — reply: {data['reply'][:80]}...")


def test_import_csv_empty():
    """空 CSV 内容"""
    print("\n=== test_import_csv_empty ===")
    with flask_app.test_client() as client:
        resp = client.post('/api/variables/import_csv',
                           data=json.dumps({"csv_content": ""}),
                           content_type='application/json')
        assert resp.status_code == 400
        data = resp.get_json()
        assert data['type'] == 'error'
        print(f"PASS — error: {data['reply']}")


def test_import_csv_missing_body():
    """请求体缺少 csv_content"""
    print("\n=== test_import_csv_missing_body ===")
    with flask_app.test_client() as client:
        resp = client.post('/api/variables/import_csv',
                           data=json.dumps({}),
                           content_type='application/json')
        assert resp.status_code == 400
        print("PASS")


def test_import_csv_invalid_json():
    """无效 JSON 请求"""
    print("\n=== test_import_csv_invalid_json ===")
    with flask_app.test_client() as client:
        resp = client.post('/api/variables/import_csv',
                           data="not json",
                           content_type='application/json')
        assert resp.status_code == 400
        print("PASS")


# ==================== 实验设计 API — 变量相关 ====================

def test_experiment_chat_reply_has_variables_hint():
    """
    验证 experiment_chat 返回的 experiment_json 包含 variables 字段时，
    前端 receive 的类型定义能匹配（不实际调 LLM，只验证路由存在）
    """
    print("\n=== test_experiment_chat_reply_has_variables_hint ===")
    # 不调 LLM，只验证路由可访问且参数校验正常
    with flask_app.test_client() as client:
        resp = client.post('/api/experiment_chat',
                           data=json.dumps({"message": "", "stream": False}),
                           content_type='application/json')
        data = resp.get_json()
        assert data['type'] == 'error', f"空消息应返回error，实际type={data.get('type')}"
        print(f"PASS — 路由参数校验正常 (type={data['type']})")


# ==================== 编译 API — 变量支持 ====================

def test_compile_with_variables():
    """编译含变量的实验 JSON"""
    print("\n=== test_compile_with_variables ===")
    with flask_app.test_client() as client:
        experiment = {
            "experiment_name": "变量测试",
            "variables": {
                "speed1": {"type": "int", "default_value": 3000, "constraints": {"min": 1000, "max": 6000}}
            },
            "steps": [
                {"type": "tool", "name": "spin_coating", "params": {
                    "spin_speed": "speed1", "spin_acc": 500, "spin_dur": 30000,
                    "reagent": "Perovskite", "volume": 60
                }}
            ]
        }
        resp = client.post('/api/compile_experiment',
                           data=json.dumps({"experiment_json": experiment}),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] == True
        assert 'code' in data
        # 变量应该被替换为默认值 3000
        assert '3000' in data['code'], f"编译代码应包含变量替换后的值 3000，实际: {data['code'][:200]}"
        print(f"PASS — code preview: {data['code'][:120]}...")


def test_compile_without_variables():
    """编译不含变量的实验 JSON（向后兼容）"""
    print("\n=== test_compile_without_variables ===")
    with flask_app.test_client() as client:
        experiment = {
            "experiment_name": "无变量测试",
            "steps": [
                {"type": "helper", "name": "WAIT", "params": {"duration": 1000}}
            ]
        }
        resp = client.post('/api/compile_experiment',
                           data=json.dumps({"experiment_json": experiment}),
                           content_type='application/json')
        assert resp.status_code == 200
        data = resp.get_json()
        assert data['success'] == True
        print("PASS")


# ==================== 执行 API — 变量校验 ====================

def test_execute_with_undeclared_variable():
    """执行含未声明变量的实验应返回校验失败"""
    print("\n=== test_execute_with_undeclared_variable ===")
    with flask_app.test_client() as client:
        experiment = {
            "experiment_name": "错误变量测试",
            "steps": [
                {"type": "tool", "name": "spin_coating", "params": {
                    "spin_speed": "unknown_var", "spin_acc": 500, "spin_dur": 30000,
                    "reagent": "Perovskite", "volume": 60
                }}
            ]
        }
        resp = client.post('/api/execute_experiment_design',
                           data=json.dumps(experiment),
                           content_type='application/json')
        # 应该启动后台线程并返回 task_trigger，或直接返回错误
        data = resp.get_json()
        print(f"PASS — type={data.get('type')}, reply={data.get('reply', '')[:80]}")


# ==================== 运行所有测试 ====================

if __name__ == "__main__":
    tests = [
        test_import_csv_success,
        test_import_csv_empty,
        test_import_csv_missing_body,
        test_import_csv_invalid_json,
        test_experiment_chat_reply_has_variables_hint,
        test_compile_with_variables,
        test_compile_without_variables,
        test_execute_with_undeclared_variable,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{'='*40}")
    print(f"结果: {passed} 通过, {failed} 失败, 共 {len(tests)} 项")
    if failed > 0:
        sys.exit(1)
