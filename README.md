# AO Gateway（Anthropic -> OpenAI Responses）

一个轻量的 FastAPI 网关：对外暴露 Anthropic `v1/messages` / `v1/messages/count_tokens` 形状的接口，内部转发到上游 OpenAI **Responses** API，并做必要的请求/响应与流式 SSE 映射，方便 Claude Code CLI 之类的工具走“LLM Gateway”。

## 文件结构

- `AO.py`：主服务（FastAPI app）
- `requirements.txt`：Python 依赖
- `.env.example`：环境变量模板（复制为 `.env`）
- `test_service.py`：本地测试脚本（请求内容固定为“你好”）

## 快速开始（Windows / PowerShell）

### 1) 创建并安装依赖

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

### 2) 配置 `.env`

```powershell
Copy-Item .env.example .env
```

编辑 `.env`，至少设置：

- `GATEWAY_TOKEN`：客户端访问本网关的口令
- `UPSTREAM_RESPONSES_URL`：上游 Responses 地址（例如 OpenAI 官方 `https://api.openai.com/v1/responses`）
- `OPENAI_API_KEY`（或 `UPSTREAM_API_KEY`）：上游鉴权 key（直连 OpenAI 时需要）

### 3) 启动服务

```powershell
.\.venv\Scripts\python -m uvicorn AO:app --host 127.0.0.1 --port 8000
```

## 接口说明

### `POST /v1/messages`

- 入参（常用字段）：
  - `system`：字符串或 text blocks（会映射到上游 `instructions`）
  - `messages`：Anthropic messages（支持 `content` 为 string 或 blocks）
  - `tools`：Anthropic tools（会映射到上游 Responses function tools）
  - `max_tokens` / `temperature` / `top_p`
  - `stream`：`true` 时返回 SSE（流式），`false` 时一次性 JSON
- 鉴权：
  - `x-api-key: <GATEWAY_TOKEN>` 或 `Authorization: Bearer <GATEWAY_TOKEN>`
- 行为：
  - 非流式：上游返回后，转成 Anthropic message 形状返回
  - 流式：把上游 SSE 的文本增量转为 Anthropic streaming 事件（`message_start` / `content_block_*` / `message_delta` / `message_stop`）

### `POST /v1/messages/count_tokens`

- 返回 `{"input_tokens": <approx>}`
- 仅为粗略估算（按字符数/4），不是精确 tokenizer。

## 配置项（环境变量）

本项目会在启动时自动读取同目录 `.env`（见 `AO.py`），并以 `.env` 为准覆盖同名环境变量，避免“两个终端会话变量不一致”导致的 401/配置不生效。

- **入站鉴权（客户端 -> 本网关）**
  - `GATEWAY_TOKEN`：必填（默认 `local-dev-token`）
- **上游配置（本网关 -> Responses）**
  - `UPSTREAM_RESPONSES_URL`：上游 Responses URL（默认 `http://127.0.0.1:9000/v1/responses`）
  - `OPENAI_API_KEY` / `UPSTREAM_API_KEY`：上游 key（会转为 `Authorization: Bearer ...`）
- **默认模型**
  - `ANTHROPIC_MODEL`：当客户端请求未传 `model` 时使用

`test_service.py` 额外支持：

- `AO_BASE_URL`：测试脚本访问的网关地址（默认 `http://127.0.0.1:8000`）

## 本地测试

先启动服务，然后执行：

```powershell
.\.venv\Scripts\python .\test_service.py
```

流式测试：

```powershell
.\.venv\Scripts\python .\test_service.py --stream
```

## Claude Code CLI（走网关）

把 Claude Code 的 Anthropic API 地址指向你的网关，并让它带上 token：

```powershell
$env:ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
$env:ANTHROPIC_AUTH_TOKEN="local-dev-token"
$env:ANTHROPIC_MODEL="上游支持的模型名"
claude
```

注意：本网关会把 `model` 直接转发到上游 Responses，所以 `ANTHROPIC_MODEL` 需要填写“上游可识别的模型名”。

## 常见问题

- `WinError 10061`：`127.0.0.1:8000` 没有服务在监听（先启动 `uvicorn` 或检查端口）。
- `401 Unauthorized`：请求头 token 与 `.env` 的 `GATEWAY_TOKEN` 不一致。
- 上游报 401/403：检查 `.env` 的 `OPENAI_API_KEY/UPSTREAM_API_KEY` 是否正确、`UPSTREAM_RESPONSES_URL` 是否指向正确的 Responses 服务。
