import argparse
import json
import os
import sys

import httpx
from dotenv import load_dotenv

from codex_cli_config import CodexCliConfig


def _extract_output_text(oai: dict) -> str:
    if isinstance(oai.get("output_text"), str) and oai.get("output_text"):
        return oai["output_text"]

    chunks: list[str] = []
    for item in oai.get("output") or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "message":
            continue
        for part in item.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "output_text":
                chunks.append(part.get("text") or "")
    return "".join(chunks)


def main() -> int:
    load_dotenv(override=True)

    codex = CodexCliConfig.from_env()

    parser = argparse.ArgumentParser(
        description="Simulate Codex CLI by calling an OpenAI-compatible Responses endpoint using ~/.codex/auth.json + ~/.codex/config.toml (or env overrides)."
    )
    parser.add_argument("--prompt", default="你好")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--url", default=os.getenv("UPSTREAM_RESPONSES_URL") or codex.responses_url)
    parser.add_argument("--api-key", default=os.getenv("OPENAI_API_KEY") or os.getenv("UPSTREAM_API_KEY") or codex.api_key)
    parser.add_argument("--model", default=os.getenv("CODEX_MODEL") or codex.model or os.getenv("ANTHROPIC_MODEL"))
    parser.add_argument("--reasoning-effort", default=codex.model_reasoning_effort)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    args = parser.parse_args()

    if not args.url:
        print(
            "Missing Responses URL. Set --url, UPSTREAM_RESPONSES_URL, or ensure ~/.codex/config.toml has model_providers.<provider>.base_url.",
            file=sys.stderr,
        )
        return 2

    payload: dict = {
        "model": args.model or "gpt-5.2",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": args.prompt}],
            }
        ],
        "max_output_tokens": int(args.max_output_tokens),
        "stream": bool(args.stream),
    }

    if codex.disable_response_storage:
        payload["store"] = False

    if args.reasoning_effort:
        payload["reasoning"] = {"effort": str(args.reasoning_effort)}

    headers: dict[str, str] = {"content-type": "application/json"}
    if args.api_key:
        headers["authorization"] = f"Bearer {args.api_key}"

    with httpx.Client(timeout=None) as client:
        if not args.stream:
            r = client.post(args.url, headers=headers, json=payload)
            if r.status_code < 200 or r.status_code >= 300:
                print(f"HTTP {r.status_code}: {r.text}", file=sys.stderr)
                return 1
            data = r.json()
            print(json.dumps(data, ensure_ascii=False, indent=2))
            text = _extract_output_text(data)
            if text:
                print("\n--- assistant text ---")
                print(text)
            return 0

        with client.stream("POST", args.url, headers=headers, json=payload) as r:
            if r.status_code < 200 or r.status_code >= 300:
                print(f"HTTP {r.status_code}: {r.read().decode('utf-8', errors='replace')}", file=sys.stderr)
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
                if evt.get("type") == "response.output_text.delta":
                    delta = evt.get("delta") or ""
                    if delta:
                        print(delta, end="", flush=True)
            print()
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
