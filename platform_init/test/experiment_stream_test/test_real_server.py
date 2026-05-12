"""Test the real Flask server for experiment_chat endpoint."""
import subprocess, sys, time, json, urllib.request, io, os, socket

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def kill_port_5000():
    """Kill any process listening on port 5000."""
    try:
        result = subprocess.run(
            ['netstat', '-ano'], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split('\n'):
            if ':5000' in line and 'LISTENING' in line:
                parts = line.split()
                pid = parts[-1]
                subprocess.run(['taskkill', '//F', '//PID', pid],
                             capture_output=True, timeout=5)
                print(f'Killed PID {pid} on port 5000')
                time.sleep(1)
    except Exception as e:
        print(f'kill_port_5000 error: {e}')


def wait_for_port(port, timeout=10):
    """Wait until port is accepting connections."""
    for i in range(timeout * 2):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            s.connect(('127.0.0.1', port))
            s.close()
            return True
        except:
            time.sleep(0.5)
    return False


# Kill any existing server
kill_port_5000()
time.sleep(2)

# Verify port is free
if wait_for_port(5000, timeout=2):
    print('ERROR: Port 5000 still occupied!')
    sys.exit(1)
print('Port 5000 is free')

# Start Flask
proc = subprocess.Popen(
    [sys.executable, '-u', 'app.py'],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    cwd='D:/PycharmProjects/SDL_agent',
)

# Wait for server to be ready
if not wait_for_port(5000, timeout=15):
    print('ERROR: Flask server did not start!')
    stdout = proc.stdout.read().decode('utf-8', errors='replace')
    stderr = proc.stderr.read().decode('utf-8', errors='replace')
    print(f'stdout: {stdout[:500]}')
    print(f'stderr: {stderr[:500]}')
    proc.terminate()
    sys.exit(1)
print('Flask server is ready')

# Test the endpoint
for attempt in range(1, 4):
    try:
        data = json.dumps({'message': f'test{attempt}'}).encode('utf-8')
        req = urllib.request.Request(
            'http://127.0.0.1:5000/api/experiment_chat',
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        resp = urllib.request.urlopen(req, timeout=120)
        body = json.loads(resp.read().decode('utf-8'))
        typ = body.get('type', '')
        if typ == 'experiment_design':
            exp = body['experiment_json']
            print(f'\nSUCCESS attempt{attempt}: {exp.get("experiment_name")}, '
                  f'{len(exp.get("steps", []))} steps')
            break
        elif typ == 'error':
            print(f'\nAttempt {attempt} error: {body.get("reply", "")[:200]}')
        else:
            print(f'\nAttempt {attempt}: unexpected type={typ}')
    except urllib.error.HTTPError as e:
        body = e.read().decode('utf-8')
        print(f'\nAttempt {attempt}: HTTP {e.code}')
        try:
            parsed = json.loads(body)
            print(f'  Body keys: {list(parsed.keys())}')
            print(f'  Content: {json.dumps(parsed, ensure_ascii=False)[:400]}')
        except:
            print(f'  Raw: {body[:400]}')
    except Exception as e:
        print(f'\nAttempt {attempt}: {e}')

# Read server output for any errors
time.sleep(0.5)
stdout = proc.stdout.read().decode('utf-8', errors='replace')
stderr = proc.stderr.read().decode('utf-8', errors='replace')
proc.terminate()
proc.wait()

print('\n=== Server STDOUT (last 15 lines) ===')
for l in stdout.split('\n')[-15:]:
    print(l)
print('\n=== Server STDERR ===')
for l in stderr.split('\n')[-10:]:
    print(l)
