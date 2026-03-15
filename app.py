from flask import Flask, request, jsonify, render_template
import time

app = Flask(__name__)


# 路由：渲染前端页面
@app.route('/')
def home():
    # Flask 会自动去 templates 文件夹下寻找 index.html
    return render_template('index.html')


# 路由：处理对话请求
@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    if not data or 'message' not in data:
        return jsonify({'error': '无效的请求'}), 400

    user_message = data['message']

    # 【核心提示】
    # 这里是你未来接入真实大模型 (如 Gemini, OpenAI, 本地 Ollama 等) 的地方。
    # 现在我们用 time.sleep 模拟网络请求延迟，并返回一个 Mock 数据。

    time.sleep(1)  # 模拟 AI 思考时间

    # 模拟 AI 回复
    ai_response = f"我是本地 AI 小助手。我收到了你的消息：\n「{user_message}」\n有什么我可以帮你的吗？"

    return jsonify({'reply': ai_response})


if __name__ == '__main__':
    print("🚀 后端服务已启动！请在浏览器访问: http://127.0.0.1:5000")
    app.run(debug=True, port=5000)