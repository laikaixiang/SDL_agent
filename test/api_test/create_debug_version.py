"""
前端调试版本生成器
生成一个带有详细console.log的index.html副本，用于调试
"""

import re

# 读取原始文件
with open('../templates/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

# 在关键位置添加调试日志
debug_points = [
    # 在sendMessage函数开始处
    (r'(async function sendMessage\(\) {)',
     r'\1\n        console.log("[DEBUG] sendMessage called");'),

    # 在创建msgDiv后
    (r'(const msgDiv = createMessageDiv\(\'ai\'\);)',
     r'\1\n                console.log("[DEBUG] Created message div:", msgDiv);'),

    # 在流式读取循环中
    (r'(const \{ done, value \} = await reader\.read\(\);)',
     r'\1\n                        console.log("[DEBUG] Read chunk - done:", done, "value length:", value?.length);'),

    # 在解码后
    (r'(msgDiv\.textContent \+= decoder\.decode\(value, \{stream: true\}\);)',
     r'const decoded = decoder.decode(value, {stream: true});\n                        console.log("[DEBUG] Decoded text:", decoded);\n                        msgDiv.textContent += decoded;'),

    # 在fetch请求后
    (r'(const response = await fetch\(\'/api/chat\',)',
     r'console.log("[DEBUG] Sending request to /api/chat");\n            \1'),

    # 在检查content-type后
    (r'(const contentType = response\.headers\.get\("content-type"\);)',
     r'\1\n            console.log("[DEBUG] Content-Type:", contentType);'),
]

debug_content = content
for pattern, replacement in debug_points:
    debug_content = re.sub(pattern, replacement, debug_content)

# 保存调试版本
with open('../templates/index_debug.html', 'w', encoding='utf-8') as f:
    f.write(debug_content)

print("Debug version created: templates/index_debug.html")
print("\nTo use it:")
print("1. Rename templates/index.html to templates/index_backup.html")
print("2. Rename templates/index_debug.html to templates/index.html")
print("3. Restart app.py")
print("4. Open browser and check Console (F12)")
