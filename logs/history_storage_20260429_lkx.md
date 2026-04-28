# 对话历史持久化 & LLM 多轮对话上下文

**日期**: 2026-04-29  
**作者**: laikaixiang  
**版本**: v1

---

## 概述

新增对话历史自动存储机制，每次会话生成 `chat_history.json` 记录完整对话；
同时普通对话模式将历史传入 LLM API，实现多轮上下文感知。

---

## 改动文件清单

| 文件 | 操作 | 说明 |
|---|---|---|
| `templates/static/js/state.js` | 修改 | 新增 `messageHistory`、`historyLastSavedIndex` 全局变量 |
| `templates/static/js/chat/history.js` | **新建** | Monkey-patch `appendMessage`/`appendMessageHtml`，逐条保存，页面关闭兜底 |
| `templates/static/js/chat/chat.js` | 修改 | `sendMessage()` 传入 `history`；流式结束后记录；`_dispatchJsonResponse` 传入 `response_type` |
| `templates/index.html` | 修改 | 在 `input_state.js` 后加载 `history.js` |
| `templates/static/js/notification.js` | 修改 | 新增 `warning` 通知类型 |
| `app.py` | 修改 | 新增 `POST /api/history/save_batch`、`GET /api/history/sessions`；标题生成；outputs 扫描；atexit 关闭钩子；普通对话传 history |
| `core/adaptive_stream.py` | 修改 | `_build_messages()` 将前端历史转为 LLM API 多轮消息格式；全部方法接受 `history` 参数 |

---

## 架构

```
浏览器
  ├─ appendMessage/appendMessageHtml (monkey-patched)
  │     → recordMessage() → messageHistory[]
  │     → scheduleHistorySave() → 500ms debounce → POST /api/history/save_batch
  ├─ beforeunload → sendBeacon → POST /api/history/save_batch (兜底)
  └─ sendMessage() → POST /api/chat { history: messageHistory }

Flask 后端
  ├─ /api/chat → handle_normal_chat(user_message, history)
  │     → adaptive_handler.generate_response(..., history=history)
  │           → _build_messages() → role ai→assistant → LLM API
  └─ /api/history/save_batch
        → 扫描 outputs (extract/results/temporal/experiment_designs)
        → 若无 title → _generate_title() → LLM 生成标题
        → 写 chat_history.json
        → 更新 sessions_index.json
```

---

## JSON 格式

### chat_history.json

```json
{
  "title": "钙钛矿钝化剂文献提取",
  "session": {
    "timestamp": "20260429_120000",
    "started_at": "2026-04-29T12:00:00",
    "saved_at": "2026-04-29T12:30:00",
    "message_count": 24
  },
  "outputs": {
    "extract": ["FAPbI3_passivators.csv"],
    "temporal": ["extraction.csv"],
    "results": ["analysis_result.json"],
    "experiment_designs": ["旋涂实验_v1.json"]
  },
  "messages": [
    {
      "role": "user",
      "content": "帮我搜寻：钙钛矿钝化剂",
      "timestamp": "2026-04-29T12:00:05.123000",
      "mode": "extract",
      "prefix": "帮我搜寻："
    },
    {
      "role": "ai",
      "content": "我分析了你的需求...",
      "timestamp": "2026-04-29T12:00:08.456000",
      "mode": "extract",
      "response_type": "field_confirm"
    }
  ]
}
```

### sessions_index.json

```json
{
  "sessions": [
    {
      "timestamp": "20260429_120000",
      "started_at": "2026-04-29T12:00:00",
      "last_saved_at": "2026-04-29T12:30:00",
      "message_count": 24,
      "title": "钙钛矿钝化剂文献提取",
      "path": "20260429_120000"
    }
  ]
}
```

---

## 消息字段说明

| 字段 | 类型 | 出现位置 | 说明 |
|---|---|---|---|
| `role` | `user`/`ai` | 全部 | 消息角色 |
| `content` | string | 全部 | 纯文本内容（AI HTML 消息已剥离标签） |
| `timestamp` | ISO 8601 | 全部 | 消息时间戳 |
| `mode` | string | 全部 | 当前模式：`normal`/`extract`/`hardware_single`/`experiment_design`/`analyze` |
| `prefix` | string | user | 模式前缀，如 `帮我搜寻：`、`硬件控制：` |
| `response_type` | string | ai | 后端返回类型：`streaming`/`task_trigger`/`field_confirm`/`hardware_confirm`/`experiment_design_mode`/`system` |

---

## 新 API 路由

| 路由 | 方法 | 说明 |
|---|---|---|
| `/api/history/save_batch` | POST | 接收 `{ messages: [...] }`，写 `chat_history.json` + 更新索引 |
| `/api/history/sessions` | GET | 返回 `sessions_index.json` 内容 |
