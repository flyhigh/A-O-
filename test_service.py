import argparse
import json
import os
import sys

import httpx
from dotenv import load_dotenv

from codex_cli_config import CodexCliConfig


def _extract_text(message: dict) -> str:
    chunks: list[str] = []
    for block in message.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            chunks.append(block.get("text") or "")
    return "".join(chunks)


def main() -> int:
    load_dotenv(override=True)
    codex = CodexCliConfig.from_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("AO_BASE_URL") or "http://127.0.0.1:8000")
    parser.add_argument("--token", default=os.getenv("GATEWAY_TOKEN") or os.getenv("ANTHROPIC_AUTH_TOKEN"))
    parser.add_argument("--model", default=os.getenv("ANTHROPIC_MODEL") or os.getenv("AO_MODEL") or codex.model)
    parser.add_argument(
        "--reasoning-effort",
        default=os.getenv("MODEL_REASONING_EFFORT") or os.getenv("AO_REASONING_EFFORT") or codex.model_reasoning_effort,
    )
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    if not args.token:
        print("Missing token. Set GATEWAY_TOKEN (or pass --token).", file=sys.stderr)
        return 2

    url = f"{args.base_url.rstrip('/')}/v1/messages"
    headers = {"x-api-key": args.token}
    payload: dict = {
        "messages": [{"role": "user", "content": "你好"}],
        "max_tokens": 128,
        "stream": bool(args.stream),
    }
    if args.model:
        payload["model"] = args.model
    if args.reasoning_effort:
        payload["model_reasoning_effort"] = args.reasoning_effort

    # Avoid accidental proxying of localhost via HTTP_PROXY/HTTPS_PROXY.
    with httpx.Client(timeout=None, trust_env=False) as client:
        if not args.stream:
            r = client.post(url, headers=headers, json=payload)
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError:
                body = r.text or ""
                print(f"HTTP {r.status_code} ({r.headers.get('content-type') or 'unknown content-type'}):", file=sys.stderr)
                if body:
                    print(body, file=sys.stderr)
                else:
                    print("<empty body>", file=sys.stderr)
                try:
                    obj = r.json()
                    print("\n--- parsed json ---", file=sys.stderr)
                    print(json.dumps(obj, ensure_ascii=False, indent=2), file=sys.stderr)
                except Exception:
                    pass
                return 1

            data = r.json()
            print(json.dumps(data, ensure_ascii=False, indent=2))
            text = _extract_text(data)
            if text:
                print("\n--- assistant text ---")
                print(text)
            return 0

        with client.stream("POST", url, headers=headers, json=payload) as r:
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError:
                body = r.read().decode("utf-8", errors="replace")
                print(f"HTTP {r.status_code} ({r.headers.get('content-type') or 'unknown content-type'}):", file=sys.stderr)
                if body:
                    print(body, file=sys.stderr)
                else:
                    print("<empty body>", file=sys.stderr)
                return 1

            for line in r.iter_lines():
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    break
                try:
                    evt = json.loads(data)
                except Exception:
                    continue
                if evt.get("type") == "content_block_delta":
                    delta = evt.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text = delta.get("text") or ""
                        if text:
                            print(text, end="", flush=True)
            print()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
