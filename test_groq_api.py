"""Simple Groq API connectivity test for chat completions."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from xai_experiments.backends import (
    DEFAULT_GROQ_BASE_URL,
    DEFAULT_GROQ_MODEL,
    DEFAULT_HTTP_USER_AGENT,
)


DEFAULT_PROMPT = "Reply with exactly: GROQ_OK"
DEFAULT_SYSTEM_PROMPT = "You are a concise test assistant."
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send one simple chat-completions request to Groq and print the response."
    )
    parser.add_argument("--base-url", default=DEFAULT_GROQ_BASE_URL)
    parser.add_argument("--api-key-env", default="GROQ_API_KEY")
    parser.add_argument("--model", default=DEFAULT_GROQ_MODEL)
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--experiment-log",
        type=Path,
        default=None,
        help=(
            "Optional experiment log JSON file. When provided, the script replays "
            "the logged prompt.messages payload and uses the logged model by default."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout-seconds", type=int, default=60)
    parser.add_argument(
        "--print-json",
        action="store_true",
        help="Print the full JSON response body.",
    )
    parser.add_argument(
        "--print-headers",
        action="store_true",
        help="Print response headers as well.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        print(
            f"Missing API key. Set {args.api_key_env} in the environment.",
            file=sys.stderr,
        )
        return 2

    url = f"{args.base_url.rstrip('/')}/chat/completions"
    payload = build_payload(args)
    payload_bytes = json.dumps(payload).encode("utf-8")

    request = Request(
        url,
        data=payload_bytes,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": DEFAULT_HTTP_USER_AGENT,
        },
        method="POST",
    )

    print(f"URL: {url}")
    print(f"Model: {payload['model']}")
    print(f"API key env: {args.api_key_env}")
    print(f"Messages: {len(payload['messages'])}")
    if args.experiment_log is None:
        print(f"Prompt: {args.prompt}")
    else:
        print(f"Replay log: {args.experiment_log}")

    started = time.perf_counter()
    try:
        with urlopen(request, timeout=args.timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            elapsed = time.perf_counter() - started
            print(f"HTTP status: {response.status}")
            print(f"Elapsed seconds: {elapsed:.3f}")
            if args.print_headers:
                print("Headers:")
                for key, value in response.headers.items():
                    print(f"  {key}: {value}")
            return print_success(body, args.print_json)
    except HTTPError as exc:
        elapsed = time.perf_counter() - started
        error_body = exc.read().decode("utf-8", errors="replace")
        print(f"HTTP status: {exc.code}", file=sys.stderr)
        print(f"Elapsed seconds: {elapsed:.3f}", file=sys.stderr)
        print("Error body:", file=sys.stderr)
        print(error_body, file=sys.stderr)
        return 1
    except URLError as exc:
        elapsed = time.perf_counter() - started
        print(f"Request failed after {elapsed:.3f} seconds: {exc.reason}", file=sys.stderr)
        return 1


def build_payload(args: argparse.Namespace) -> dict[str, object]:
    if args.experiment_log is None:
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": args.prompt},
        ]
        model = args.model
        temperature = args.temperature
        top_p = args.top_p
        max_tokens = args.max_tokens
    else:
        log_data = json.loads(args.experiment_log.read_text(encoding="utf-8"))
        prompt = log_data.get("prompt") or {}
        messages = prompt.get("messages") or []
        if not messages:
            raise SystemExit(f"No prompt.messages found in {args.experiment_log}")
        generation_parameters = log_data.get("generation_parameters") or {}
        model = str(generation_parameters.get("model") or args.model)
        temperature = float(generation_parameters.get("temperature", args.temperature))
        top_p = float(generation_parameters.get("top_p", args.top_p))
        max_tokens = int(generation_parameters.get("max_tokens", args.max_tokens))

    return {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }


def print_success(body: str, print_json: bool) -> int:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        print("Response body is not valid JSON:")
        print(body)
        return 1

    if print_json:
        print("Response JSON:")
        print(json.dumps(data, indent=2, ensure_ascii=False))

    choices = data.get("choices") or []
    first_choice = choices[0] if choices else {}
    message = first_choice.get("message") or {}
    content = message.get("content")

    print("Assistant response:")
    print(content if content is not None else "<no message content>")

    usage = data.get("usage")
    if usage:
        print("Usage:")
        print(json.dumps(usage, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
