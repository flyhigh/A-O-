import os, json, time
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from codex_cli_config import CodexCliConfig

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_BASE_DIR, ".env"), override=True)

app = FastAPI()

_CODEX = CodexCliConfig.from_env()

UPSTREAM_RESPONSES_URL = (
    os.getenv("UPSTREAM_RESPONSES_URL")
    or _CODEX.responses_url
    or "http://127.0.0.1:9000/v1/responses"
)
UPSTREAM_API_KEY = (
    os.getenv("UPSTREAM_API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or _CODEX.api_key
)
GATEWAY_TOKEN = os.getenv("GATEWAY_TOKEN", "local-dev-token")

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL") or _CODEX.model or "my-model"
DEFAULT_REASONING_EFFORT = (
    os.getenv("UPSTREAM_REASONING_EFFORT")
    or os.getenv("MODEL_REASONING_EFFORT")
    or _CODEX.model_reasoning_effort
)
UPSTREAM_REASONING_FIELD = (os.getenv("UPSTREAM_REASONING_FIELD") or "reasoning").strip().lower()
DISABLE_RESPONSE_STORAGE = (
    (str(os.getenv("DISABLE_RESPONSE_STORAGE") or "").strip().lower() in ("1", "true", "yes", "y", "on"))
    if os.getenv("DISABLE_RESPONSE_STORAGE") is not None
    else bool(_CODEX.disable_response_storage)
)


def _truncate_text(s: str, limit: int = 2000) -> str:
    if not s:
        return ""
    if len(s) <= limit:
        return s
    return s[:limit] + f"...(truncated, {len(s)} chars)"


def _apply_reasoning_effort(oai_payload: Dict[str, Any], reasoning_effort: str) -> None:
    effort = (reasoning_effort or "").strip()
    if not effort:
        return

    # Some OpenAI-compatible proxies may not accept nested objects. Allow overriding the field name.
    # - reasoning (default): {"reasoning": {"effort": "xhigh"}}
    # - reasoning_effort: {"reasoning_effort": "xhigh"}
    field = (UPSTREAM_REASONING_FIELD or "reasoning").strip().lower()
    if field in ("reasoning", "reasoning.effort"):
        oai_payload["reasoning"] = {"effort": effort}
        return

    oai_payload[field] = effort


def _check_auth(req: Request):
    # Claude Code 网关文档：ANTHROPIC_AUTH_TOKEN 会作为 Authorization 发出 [<sup>1</sup>](https://docs.anthropic.com/en/docs/claude-code/llm-gateway)
    auth = req.headers.get("authorization", "") or ""
    x_api_key = req.headers.get("x-api-key")
    bearer = auth[7:].strip() if auth.lower().startswith("bearer ") else None
    raw_auth = auth.strip()
    raw_token = raw_auth if raw_auth and " " not in raw_auth else None
    token = x_api_key or bearer or raw_token
    if GATEWAY_TOKEN and token != GATEWAY_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _upstream_headers(stream: bool) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    if stream:
        headers["accept"] = "text/event-stream"
    if UPSTREAM_API_KEY:
        headers["authorization"] = f"Bearer {UPSTREAM_API_KEY}"
    return headers


# -------------------- Anthropic -> OpenAI mapping --------------------
def _sys_to_instructions(system: Any) -> Optional[str]:
    # Anthropic Messages：system 是 top-level（不是 role=system）[<sup>3</sup>](https://platform.claude.com/docs/en/api/go/messages/count_tokens)
    if system is None:
        return None
    if isinstance(system, str):
        s = system.strip()
        return s or None
    if isinstance(system, list):
        out = []
        for b in system:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(b.get("text", ""))
        s = "".join(out).strip()
        return s or None
    return None


def _content_to_blocks(content: Any) -> List[Dict[str, Any]]:
    # content 可以是 string 或 blocks [<sup>3</sup>](https://platform.claude.com/docs/en/api/go/messages/count_tokens)
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if isinstance(content, list):
        return content
    return [{"type": "text", "text": str(content)}]


def _anthropic_tools_to_openai_tools(tools: Optional[List[Dict[str, Any]]]) -> Optional[List[Dict[str, Any]]]:
    """
    OpenAI Responses create: tools 支持 FunctionTool {type:"function", name, parameters, strict, description} [<sup>2</sup>](https://developers.openai.com/api/reference/resources/responses/methods/create)
    """
    if not tools:
        return None
    out = []
    for t in tools:
        out.append({
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            "strict": True,  # 文档里 strict 默认 true [<sup>2</sup>](https://developers.openai.com/api/reference/resources/responses/methods/create)
        })
    return out


def _anthropic_messages_to_openai_input(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    关键点：
    - 文本：转成 Responses 的 message item（role + content: [{type:"input_text", text}]）
    - tool_result：转成 {type:"function_call_output", call_id, output} [<sup>2</sup>](https://developers.openai.com/api/reference/resources/responses/methods/create)
    - tool_use（如果历史里有）：转成 {type:"function_call", call_id, name, arguments} [<sup>2</sup>](https://developers.openai.com/api/reference/resources/responses/methods/create)
    """
    items: List[Dict[str, Any]] = []

    for m in messages or []:
        role = m.get("role", "user")
        blocks = _content_to_blocks(m.get("content"))

        # 先收集 text blocks（合并成一条 message item）
        text_chunks = []
        for b in blocks:
            if b.get("type") == "text":
                text_chunks.append(b.get("text", ""))

        if text_chunks:
            items.append({
                "type": "message",
                "role": role,
                "content": [{"type": "input_text", "text": "".join(text_chunks)}],
            })

        # 处理 tool_use / tool_result（按出现顺序追加 items）
        for b in blocks:
            if b.get("type") == "tool_use":
                # Anthropic: {type:"tool_use", id, name, input:{...}}
                call_id = b.get("id")
                name = b.get("name")
                arguments = json.dumps(b.get("input", {}), ensure_ascii=False)
                if call_id and name:
                    items.append({
                        "type": "function_call",
                        "call_id": call_id,
                        "name": name,
                        "arguments": arguments,
                    })

            if b.get("type") == "tool_result":
                # Anthropic: {type:"tool_result", tool_use_id, content:[{type:"text",text:"..."}]}
                call_id = b.get("tool_use_id")
                out_text = ""
                for cb in _content_to_blocks(b.get("content")):
                    if cb.get("type") == "text":
                        out_text += cb.get("text", "")
                if call_id:
                    items.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": out_text,  # 文档：output 可以是 string [<sup>2</sup>](https://developers.openai.com/api/reference/resources/responses/methods/create)
                    })

    return items


# -------------------- OpenAI -> Anthropic mapping (non-stream) --------------------
def _openai_response_to_anthropic_message(oai: Dict[str, Any], model: str) -> Dict[str, Any]:
    content_blocks: List[Dict[str, Any]] = []

    # Responses 返回 output items，其中 function_call / message 都是 items [<sup>4</sup>](https://platform.openai.com/docs/guides/responses-vs-chat-completions)
    for item in oai.get("output", []) or []:
        itype = item.get("type")
        if itype == "message":
            for part in item.get("content", []) or []:
                if part.get("type") == "output_text":
                    content_blocks.append({"type": "text", "text": part.get("text", "")})
        elif itype == "function_call":
            # function_call: {call_id,name,arguments(JSON string)} [<sup>2</sup>](https://developers.openai.com/api/reference/resources/responses/methods/create)
            args_s = item.get("arguments", "{}")
            try:
                args_obj = json.loads(args_s) if isinstance(args_s, str) else args_s
            except Exception:
                args_obj = {}
            content_blocks.append({
                "type": "tool_use",
                "id": item.get("call_id"),   # 让 tool_use_id == call_id，回填就能对上 [<sup>2</sup>](https://developers.openai.com/api/reference/resources/responses/methods/create)
                "name": item.get("name"),
                "input": args_obj,
            })

    stop_reason = "tool_use" if any(b.get("type") == "tool_use" for b in content_blocks) else "end_turn"
    usage = oai.get("usage") or {}
    msg_id = f"msg_{int(time.time()*1000)}"

    return {
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content_blocks or [{"type": "text", "text": ""}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
        }
    }


# -------------------- /v1/messages --------------------
@app.post("/v1/messages")
async def messages(req: Request):
    _check_auth(req)
    body = await req.json()

    model = body.get("model") or DEFAULT_MODEL
    stream = bool(body.get("stream", False))
    reasoning_effort = body.get("model_reasoning_effort") or DEFAULT_REASONING_EFFORT

    oai_payload: Dict[str, Any] = {
        "model": model,
        # OpenAI Responses：instructions 是 system/developer 注入 [<sup>2</sup>](https://developers.openai.com/api/reference/resources/responses/methods/create)
        "instructions": _sys_to_instructions(body.get("system")),
        "input": _anthropic_messages_to_openai_input(body.get("messages", [])),
        "tools": _anthropic_tools_to_openai_tools(body.get("tools")),
        "max_output_tokens": body.get("max_tokens", 1024),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "stream": stream,  # Responses 支持 stream=true [<sup>2</sup>](https://developers.openai.com/api/reference/resources/responses/methods/create)
    }
    if DISABLE_RESPONSE_STORAGE:
        oai_payload["store"] = False
    if reasoning_effort:
        _apply_reasoning_effort(oai_payload, str(reasoning_effort))
    oai_payload = {k: v for k, v in oai_payload.items() if v is not None}

    async with httpx.AsyncClient(timeout=None) as client:
        if not stream:
            try:
                r = await client.post(UPSTREAM_RESPONSES_URL, json=oai_payload, headers=_upstream_headers(stream=False))
            except httpx.RequestError as e:
                raise HTTPException(status_code=502, detail={"message": "Upstream request failed", "error": str(e)})

            if r.status_code < 200 or r.status_code >= 300:
                raise HTTPException(
                    status_code=502,
                    detail={
                        "message": "Upstream returned non-2xx status",
                        "upstream_status": r.status_code,
                        "upstream_body": _truncate_text(r.text or ""),
                    },
                )

            return JSONResponse(_openai_response_to_anthropic_message(r.json(), model))

        # ------------- stream: OpenAI SSE -> Anthropic SSE -------------
        async def sse():
            msg_id = f"msg_{int(time.time()*1000)}"

            # Claude streaming 结构：message_start -> content_block_* -> message_delta -> message_stop [<sup>5</sup>](https://platform.claude.com/docs/en/build-with-claude/streaming?utm_source=openai)
            yield "event: message_start\n"
            yield "data: " + json.dumps({
                "type": "message_start",
                "message": {
                    "id": msg_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
            }, ensure_ascii=False) + "\n\n"

            next_index = 0
            text_block_index: Optional[int] = None
            text_block_open = False
            stop_reason: Optional[str] = None
            final_usage: Dict[str, Any] = {"input_tokens": 0, "output_tokens": 0}

            def open_text_block_if_needed():
                nonlocal next_index, text_block_index, text_block_open
                if text_block_open:
                    return
                text_block_index = next_index
                next_index += 1
                text_block_open = True
                yield "event: content_block_start\n"
                yield "data: " + json.dumps({
                    "type": "content_block_start",
                    "index": text_block_index,
                    "content_block": {"type": "text", "text": ""}
                }, ensure_ascii=False) + "\n\n"

            def close_text_block_if_open():
                nonlocal text_block_open, text_block_index
                if not text_block_open or text_block_index is None:
                    return
                yield "event: content_block_stop\n"
                yield "data: " + json.dumps({
                    "type": "content_block_stop",
                    "index": text_block_index
                }, ensure_ascii=False) + "\n\n"
                text_block_open = False
                text_block_index = None

            try:
                async with client.stream(
                    "POST",
                    UPSTREAM_RESPONSES_URL,
                    json=oai_payload,
                    headers=_upstream_headers(stream=True),
                ) as r:
                    if r.status_code < 200 or r.status_code >= 300:
                        body_bytes = await r.aread()
                        body_text = body_bytes.decode("utf-8", errors="replace")
                        err_text = _truncate_text(body_text)

                        # Best-effort: surface upstream error to the SSE client.
                        if not text_block_open:
                            for chunk in open_text_block_if_needed():
                                yield chunk
                        yield "event: content_block_delta\n"
                        yield "data: " + json.dumps({
                            "type": "content_block_delta",
                            "index": text_block_index,
                            "delta": {"type": "text_delta", "text": f"[upstream {r.status_code}] {err_text}"},
                        }, ensure_ascii=False) + "\n\n"
                        for chunk in close_text_block_if_open():
                            yield chunk

                        yield "event: message_delta\n"
                        yield "data: " + json.dumps({
                            "type": "message_delta",
                            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                            "usage": {"input_tokens": 0, "output_tokens": 0},
                        }, ensure_ascii=False) + "\n\n"
                        yield "event: message_stop\n"
                        yield "data: " + json.dumps({"type": "message_stop"}, ensure_ascii=False) + "\n\n"
                        return

                    async for line in r.aiter_lines():
                        if not line:
                            continue
                        if not line.startswith("data:"):
                            continue
                        data = line[len("data:"):].strip()
                        if data == "[DONE]":
                            break

                        try:
                            evt = json.loads(data)
                        except Exception:
                            continue

                        etype = evt.get("type")

                        # 文本增量：response.output_text.delta 带 delta 字段 [<sup>6</sup>](https://platform.openai.com/docs/api-reference/realtime-server-events/response/output_item/created?utm_source=openai)
                        if etype == "response.output_text.delta":
                            delta = evt.get("delta", "")
                            if delta:
                                # 开 text block
                                if not text_block_open:
                                    for chunk in open_text_block_if_needed():
                                        yield chunk
                                yield "event: content_block_delta\n"
                                yield "data: " + json.dumps({
                                    "type": "content_block_delta",
                                    "index": text_block_index,
                                    "delta": {"type": "text_delta", "text": delta}
                                }, ensure_ascii=False) + "\n\n"

                        # item 完成：response.output_item.done 包含 item [<sup>7</sup>](https://platform.openai.com/docs/api-reference/realtime-server-events/response-text-done?utm_source=openai)
                        if etype == "response.output_item.done":
                            item = evt.get("item") or {}
                            if item.get("type") == "function_call":
                                # function_call: call_id/name/arguments [<sup>2</sup>](https://developers.openai.com/api/reference/resources/responses/methods/create)
                                for chunk in close_text_block_if_open():
                                    yield chunk

                                call_id = item.get("call_id")
                                name = item.get("name")
                                args_s = item.get("arguments", "{}")
                                try:
                                    args_obj = json.loads(args_s) if isinstance(args_s, str) else args_s
                                except Exception:
                                    args_obj = {}

                                tool_index = next_index
                                next_index += 1

                                yield "event: content_block_start\n"
                                yield "data: " + json.dumps({
                                    "type": "content_block_start",
                                    "index": tool_index,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": call_id,
                                        "name": name,
                                        "input": args_obj
                                    }
                                }, ensure_ascii=False) + "\n\n"

                                yield "event: content_block_stop\n"
                                yield "data: " + json.dumps({
                                    "type": "content_block_stop",
                                    "index": tool_index
                                }, ensure_ascii=False) + "\n\n"

                                stop_reason = "tool_use"

                        # 结束事件（不同实现可能是 response.completed / response.done）
                        if etype in ("response.completed", "response.done"):
                            resp = evt.get("response") or {}
                            usage = resp.get("usage") or {}
                            final_usage = {
                                "input_tokens": usage.get("input_tokens", 0),
                                "output_tokens": usage.get("output_tokens", 0),
                            }
                            break
            except httpx.RequestError as e:
                # Best-effort: surface upstream request errors to the SSE client.
                if not text_block_open:
                    for chunk in open_text_block_if_needed():
                        yield chunk
                yield "event: content_block_delta\n"
                yield "data: " + json.dumps({
                    "type": "content_block_delta",
                    "index": text_block_index,
                    "delta": {"type": "text_delta", "text": f"[upstream error] {str(e)}"},
                }, ensure_ascii=False) + "\n\n"
                for chunk in close_text_block_if_open():
                    yield chunk

                yield "event: message_delta\n"
                yield "data: " + json.dumps({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }, ensure_ascii=False) + "\n\n"
                yield "event: message_stop\n"
                yield "data: " + json.dumps({"type": "message_stop"}, ensure_ascii=False) + "\n\n"
                return

            # 收尾：关掉 text block（如果还开着）
            for chunk in close_text_block_if_open():
                yield chunk

            if stop_reason is None:
                stop_reason = "end_turn"

            yield "event: message_delta\n"
            yield "data: " + json.dumps({
                "type": "message_delta",
                "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                "usage": final_usage
            }, ensure_ascii=False) + "\n\n"

            yield "event: message_stop\n"
            yield "data: " + json.dumps({"type": "message_stop"}, ensure_ascii=False) + "\n\n"

        return StreamingResponse(sse(), media_type="text/event-stream")


# -------------------- /v1/messages/count_tokens --------------------
@app.post("/v1/messages/count_tokens")
async def count_tokens(req: Request):
    _check_auth(req)
    body = await req.json()

    # Claude Token Count API：也是 messages + system + tools 的同结构 [<sup>3</sup>](https://platform.claude.com/docs/en/api/go/messages/count_tokens)
    # 这里给一个可用的“粗略估算”。想精确需要你接入自己的 tokenizer。
    system = body.get("system")
    sys_text = _sys_to_instructions(system) or ""

    txt = sys_text
    for m in body.get("messages", []) or []:
        for b in _content_to_blocks(m.get("content")):
            if b.get("type") == "text":
                txt += b.get("text", "")

    approx = max(1, (len(txt) + 3) // 4)
    return JSONResponse({"input_tokens": approx})

