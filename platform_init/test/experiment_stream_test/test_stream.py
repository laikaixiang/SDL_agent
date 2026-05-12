"""
实验设计流式端点测试

用法:
    cd D:/PycharmProjects/SDL_agent
    python test/experiment_stream_test/test_stream.py
"""
import sys
import io
import json
import traceback

# Windows: fix stdout encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, '.')


def parse_sse_stream(raw: str) -> list[dict]:
    """Properly parse SSE stream data, handling multi-line and large events."""
    events = []
    current_data = ''
    for line in raw.split('\n'):
        if line.startswith('data: '):
            current_data += line[6:]
        elif line == '' and current_data:
            try:
                events.append(json.loads(current_data))
            except json.JSONDecodeError as e:
                print(f"  [WARN] JSON parse error: {e}, data[:200]={current_data[:200]}")
            current_data = ''
    # Handle last event if no trailing newline
    if current_data:
        try:
            events.append(json.loads(current_data))
        except json.JSONDecodeError:
            pass
    return events


def test_sse_event():
    """测试 _sse_event 静态方法是否能正常访问模块级 json"""
    print("=" * 50)
    print("TEST 1: _sse_event static method")
    from core.field_inference import ExperimentDesignAgent
    try:
        result = ExperimentDesignAgent._sse_event("chunk", "hello 中文")
        assert result.startswith("data: "), f"Unexpected format: {result[:50]}"
        parsed = json.loads(result[6:].strip())
        assert parsed["type"] == "chunk"
        assert parsed["data"] == "hello 中文"
        print("PASS: _sse_event works with Chinese text")

        # Test with nested dict (like complete event)
        large_data = {
            "experiment_json": {"experiment_name": "测试实验", "steps": []},
            "visual_data": {"nodes": [], "edges": []},
            "reply": "中文回复测试" * 10,
        }
        result2 = ExperimentDesignAgent._sse_event("complete", large_data)
        assert result2.startswith("data: ")
        parsed2 = json.loads(result2[6:].strip())
        assert parsed2["type"] == "complete"
        assert parsed2["data"]["reply"].startswith("中文回复测试")
        print("PASS: _sse_event works with nested dict")
    except Exception as e:
        print(f"FAIL: {e}")
        traceback.print_exc()
        return False
    return True


def test_parse_experiment_design_stream():
    """测试流式生成方法（直接调用，不走Flask）"""
    print("\n" + "=" * 50)
    print("TEST 2: parse_experiment_design_stream (direct)")
    from core.field_inference import ExperimentDesignAgent

    agent = ExperimentDesignAgent()

    # Try up to 3 times since LLM output is non-deterministic
    for attempt in range(3):
        chunks = 0
        complete_data = None
        error_msg = None

        gen = agent.parse_experiment_design_stream("旋涂实验：转速3000rpm，时间30秒")

        try:
            for event_str in gen:
                assert event_str.startswith("data: "), f"Bad format: {event_str[:80]}"
                assert event_str.endswith("\n\n"), f"Missing newlines: {repr(event_str[-10:])}"
                msg = json.loads(event_str[6:].strip())
                if msg["type"] == "chunk":
                    chunks += 1
                elif msg["type"] == "complete":
                    complete_data = msg["data"]
                    break
                elif msg["type"] == "error":
                    error_msg = msg["data"]
                    break
        except Exception as e:
            print(f"FAIL (attempt {attempt+1}): Generator raised exception: {e}")
            traceback.print_exc()
            if attempt == 2:
                return False
            continue

        if error_msg:
            print(f"  Attempt {attempt+1}: LLM returned error: {error_msg[:80]}...")
            if attempt == 2:
                print(f"FAIL: All 3 attempts returned errors. Last: {error_msg[:200]}")
                return False
            continue

        if complete_data:
            print(f"PASS: Received {chunks} chunks, complete event present")
            print(f"  experiment_name: {complete_data['experiment_json'].get('experiment_name')}")
            print(f"  steps: {len(complete_data['experiment_json'].get('steps', []))}")
            print(f"  visual nodes: {len(complete_data['visual_data'].get('nodes', []))}")
            print(f"  reply: {complete_data['reply'][:50]}...")
            return True

    return False


def test_flask_endpoint():
    """通过 Flask test client 测试完整端点"""
    print("\n" + "=" * 50)
    print("TEST 3: Flask endpoint /api/experiment_chat_stream")

    from app import app as flask_app

    # Try up to 3 times due to LLM non-determinism
    for attempt in range(3):
        with flask_app.test_client() as client:
            resp = client.get('/api/experiment_chat_stream?message=旋涂实验：转速3000rpm，时间30秒')

            if resp.status_code != 200:
                print(f"FAIL: HTTP {resp.status_code}")
                print(f"  Body: {resp.data.decode('utf-8')}")
                return False

            if 'text/event-stream' not in (resp.content_type or ''):
                print(f"FAIL: Wrong content-type: {resp.content_type}")
                return False

            raw = resp.data.decode('utf-8')
            events = parse_sse_stream(raw)

            chunks = [e for e in events if e['type'] == 'chunk']
            complete = [e for e in events if e['type'] == 'complete']
            errors = [e for e in events if e['type'] == 'error']

            if errors:
                print(f"  Attempt {attempt+1}: Got error event: {errors[0]['data'][:80]}...")
                if attempt == 2:
                    print(f"FAIL: All 3 attempts returned errors. Last: {errors[0]['data'][:200]}")
                    return False
                continue

            if complete:
                c = complete[0]
                exp_json = c['data']['experiment_json']
                print(f"PASS: HTTP 200, {len(chunks)} chunks, {len(events)} events")
                print(f"  experiment: {exp_json.get('experiment_name')}, {len(exp_json.get('steps', []))} steps")
                print(f"  reply: {c['data']['reply'][:60]}...")
                return True

            print(f"  Attempt {attempt+1}: No complete or error event found ({len(events)} events)")
            if events:
                types = [e['type'] for e in events]
                print(f"  Event types: {types}")
            if attempt == 2:
                print("FAIL: No complete or error event after 3 attempts")
                return False

    return False


def test_non_streaming_endpoint():
    """测试非流式端点 /api/experiment_chat（与原templates方案一致）"""
    print("\n" + "=" * 50)
    print("TEST 4a: Non-streaming /api/experiment_chat (POST)")

    from app import app as flask_app

    for attempt in range(3):
        with flask_app.test_client() as client:
            resp = client.post('/api/experiment_chat',
                data=json.dumps({'message': '旋涂实验：转速3000rpm，时间30秒'}),
                content_type='application/json')

            if resp.status_code != 200:
                print(f"FAIL (attempt {attempt+1}): HTTP {resp.status_code}")
                print(f"  Body: {resp.data.decode('utf-8')[:300]}")
                return False

            data = resp.get_json()
            if data.get('type') == 'experiment_design':
                exp = data['experiment_json']
                print(f"PASS: type={data['type']}, experiment='{exp.get('experiment_name')}', "
                      f"{len(exp.get('steps', []))} steps, visual nodes={len(data.get('visual_data', {}).get('nodes', []))}")
                return True
            elif data.get('type') == 'error':
                print(f"  Attempt {attempt+1}: Got error: {data.get('reply', '')[:100]}")
                if attempt == 2:
                    print(f"FAIL: All 3 attempts returned errors")
                    return False
                continue
            else:
                print(f"FAIL: Unexpected type: {data.get('type')}")
                return False
    return False


def test_full_chat_flow():
    """测试从 /api/chat 到 /api/experiment_chat 的完整流程（匹配templates方案）"""
    print("\n" + "=" * 50)
    print("TEST 4b: Full flow /api/chat -> /api/experiment_chat (POST)")

    from app import app as flask_app

    with flask_app.test_client() as client:
        # Step 1: Send experiment chat message (matching templates)
        resp1 = client.post('/api/chat',
            data=json.dumps({
                'message': '实验设计：旋涂实验转速3000rpm',
                'action': 'chat',
                'history': []
            }),
            content_type='application/json')

        if resp1.status_code != 200:
            print(f"FAIL step1: HTTP {resp1.status_code}")
            return False

        data1 = resp1.get_json()
        if data1.get('type') != 'experiment_design_mode':
            print(f"FAIL step1: Wrong type: {data1.get('type')}")
            print(f"  reply: {data1.get('reply', '')[:100]}")
            return False

        cmd = data1.get('command', '')
        print(f"PASS step1: Got experiment_design_mode, command='{cmd}'")

        # Step 2: POST to /api/experiment_chat (matching templates startExperimentChat)
        for attempt in range(3):
            resp2 = client.post('/api/experiment_chat',
                data=json.dumps({'message': cmd}),
                content_type='application/json')

            if resp2.status_code != 200:
                print(f"FAIL step2: HTTP {resp2.status_code}")
                print(f"  Body: {resp2.data.decode('utf-8')[:300]}")
                return False

            data2 = resp2.get_json()
            if data2.get('type') == 'experiment_design':
                exp = data2['experiment_json']
                print(f"PASS step2: type={data2['type']}, experiment='{exp.get('experiment_name')}', "
                      f"{len(exp.get('steps', []))} steps")
                return True
            elif data2.get('type') == 'error':
                print(f"  Attempt {attempt+1}: Got error: {data2.get('reply', '')[:100]}")
                if attempt == 2:
                    print(f"FAIL step2: All 3 attempts returned errors")
                    return False
                continue
            else:
                print(f"FAIL step2: Unexpected type: {data2.get('type')}")
                return False

    return False


def test_unicode_edge_cases():
    """测试 Unicode 边界情况"""
    print("\n" + "=" * 50)
    print("TEST 5: Unicode edge cases")

    from core.field_inference import ExperimentDesignAgent

    agent = ExperimentDesignAgent()

    test_msgs = [
        "旋涂实验：使用溶液A（浓度5mg/mL）在3000rpm下旋涂",
        "测试特殊字符: α-β-γ 钙钛矿 CsPbI₃",
    ]

    for msg in test_msgs:
        for attempt in range(3):
            gen = agent.parse_experiment_design_stream(msg)
            complete = None
            error_msg = None
            for event_str in gen:
                msg_data = json.loads(event_str[6:].strip())
                if msg_data["type"] == "complete":
                    complete = msg_data["data"]
                    break
                elif msg_data["type"] == "error":
                    error_msg = msg_data["data"]
                    break

            if error_msg:
                if attempt == 2:
                    print(f"FAIL for '{msg[:30]}...': {error_msg[:100]}")
                    return False
                continue

            if complete:
                print(f"PASS: '{msg[:40]}...' -> {complete['experiment_json'].get('experiment_name', 'N/A')}")
                break
        else:
            print(f"FAIL: No complete for '{msg[:40]}...' after 3 attempts")
            return False
    return True


if __name__ == '__main__':
    tests = [
        test_sse_event,
        test_parse_experiment_design_stream,
        test_flask_endpoint,
        test_non_streaming_endpoint,
        test_full_chat_flow,
        test_unicode_edge_cases,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\nUNEXPECTED ERROR in {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
