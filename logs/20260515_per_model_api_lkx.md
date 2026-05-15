# Per-Model API Configuration + DeepSeek Compatibility

**Date:** 2026-05-15  
**Author:** lkx  
**Version:** v2.3

## Summary

解耦多模型 API 配置，每个模型 (TALK / VL / EXPERIMENT / EMBEDDING) 可独立配置 API_KEY / API_URL / EXTRA_BODY。新增全局 MAX_TOKENS 控制。修复 DeepSeek thinking 模型兼容性。

## Changed Files

| File | Change |
|------|--------|
| `core/config.py` | 新增 TALK_API_KEY/URL, VL_API_KEY/URL, EXPERIMENT_API_KEY/URL + EXTRA_BODY 系列 + MAX_TOKENS + `get_extra_body()` |
| `core/llm_client.py` | `__init__` 接受 api_key/api_url/extra_body；`call_api`/`stream_raw` 支持 tools/extra_body/MAX_TOKENS；reasoning_content fallback；新增 `run_with_tools()` 循环 |
| `core/tool_executor.py` | **新文件** — ToolDef, TOOL_REGISTRY, ToolExecutor, build_openai_tools() |
| `core/adaptive_stream.py` | payload merge TALK_EXTRA_BODY + MAX_TOKENS；delta 处理 reasoning_content；消息历史保留 reasoning_content |
| `extract/extraction_engine.py` | VL/TALK 方法分别 merge EXTRA_BODY + MAX_TOKENS；delta 处理 reasoning_content |
| `core/field_inference.py` | FieldInference/ExperimentDesignAgent 注入对应 API 凭证 + extra_body；max_tokens → None |
| `core/experiment_agent.py` | EXPERIMENT_API_KEY/URL 替代全局 API_KEY/URL |
| `experiment/agent.py` | 同上 |
| `core/hardware_controller.py` | max_tokens → None |
| `software/auto_analyze.py` | merge TALK_EXTRA_BODY + MAX_TOKENS；reasoning_content fallback |
| `software/.../prompt_template.py` | 同上 |
| `prompts/optimizer.py` | merge TALK_EXTRA_BODY；max_tokens 条件化 |
| `prompts/api.py` | max_tokens → None |
| `utils/pdf_metadata_extractor.py` | LLMClient 注入 VL_API_KEY/URL/extra_body |
| `extract/literature_indexer.py` | 同上 |
| `utils/stream_adapter.py` | TODO: tool_calls 事件检测 |
| `app.py` | LLMClient 注入 TALK 凭证；消息历史保留 reasoning_content；TODO: /api/chat_with_tools |
| `config.example.json` | 新增 14 个配置项 |
| `config.json` | 同上 |
| `platform_init/test/api_test/api_test.py` | 重写为使用工程 LLMClient/AdaptiveStreamHandler，19 项测试 |
| `platform_init/check_stream_capability.py` | EXPERIMENT_API_KEY/URL |

## New Config Keys

```json
"TALK_API_KEY": "",        "TALK_API_URL": "",
"VL_API_KEY": "",          "VL_API_URL": "",
"EXPERIMENT_API_KEY": "",  "EXPERIMENT_API_URL": "",
"EXTRA_BODY": "",
"TALK_EXTRA_BODY": "",     "VL_EXTRA_BODY": "",     "EXPERIMENT_EXTRA_BODY": "",
"MAX_TOKENS": null
```

## DeepSeek Usage

```json
"TALK_API_KEY": "<key>",
"TALK_API_URL": "https://api.deepseek.com/v1/chat/completions",
"MODEL_NAME_TALK": "deepseek-v4-flash",
"TALK_EXTRA_BODY": "{\"thinking\": {\"type\": \"enabled\"}}",
"MAX_TOKENS": null
```

## TODO

- `core/tool_executor.py` — 从 hardware/tools/REGISTRY.json 导入硬件工具
- `app.py` — `/api/chat_with_tools` 路由
- `utils/stream_adapter.py` — tool_calls SSE 事件
- `llm_client.py:run_with_tools()` — 流式 tool calling 支持
