from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any, Callable


GenerateFunc = Callable[[str, str, dict[str, Any]], dict[str, Any]]


def call_local_llm_with_reliability(
    prompt: str,
    config: dict[str, Any],
    case_id: str,
    retrieval_mode: str,
    *,
    generate_func: GenerateFunc | None = None,
) -> dict[str, Any]:
    """Call local Ollama with retry/empty-answer handling.

    The wrapper is intentionally local-only and auditable. Tests can inject a
    generate_func to avoid network access.
    """

    llm_config = dict(config or {})
    primary_model = str(llm_config.get("model", "")).strip()
    fallback_models = [str(item).strip() for item in llm_config.get("fallback_models", []) or [] if str(item).strip()]
    models = [primary_model] if primary_model else []
    if bool(llm_config.get("retry_with_fallback_model", False)):
        models.extend(model for model in fallback_models if model not in models)
    if not models:
        return make_not_called_result(
            case_id=case_id,
            retrieval_mode=retrieval_mode,
            failure_type="llm_unavailable",
            reason="no local model configured",
        )

    max_retries = max(0, int(llm_config.get("max_retries", 1)))
    retry_on_empty = bool(llm_config.get("retry_on_empty", True))
    retry_on_timeout = bool(llm_config.get("retry_on_timeout", False))
    retry_with_same_model = bool(llm_config.get("retry_with_same_model", True))
    max_attempts = 1 + max_retries
    attempts: list[dict[str, Any]] = []
    generator = generate_func or ollama_generate
    current_model_index = 0
    final_answer = ""
    final_status = "llm_exception"
    failure_type = "llm_exception"

    for attempt_index in range(1, max_attempts + 1):
        model = models[min(current_model_index, len(models) - 1)]
        try:
            response = generator(prompt, model, llm_config)
            raw_text = str(response.get("raw_response", response.get("answer_text", "")))
            answer_text = clean_answer(raw_text, llm_config)
            is_empty = is_empty_answer(answer_text, llm_config)
            status = "empty_answer" if is_empty else "success"
            attempts.append(
                build_attempt(
                    attempt_index=attempt_index,
                    model=model,
                    status=status,
                    raw_response=raw_text,
                    answer_text=answer_text,
                    config=llm_config,
                    error="",
                )
            )
            if not is_empty:
                final_answer = answer_text
                final_status = "answered_after_retry" if attempt_index > 1 else "answered"
                failure_type = ""
                break
            final_status = "llm_empty_answer"
            failure_type = "llm_empty_answer"
            if not retry_on_empty:
                break
        except TimeoutError as exc:
            final_status = "llm_timeout"
            failure_type = "llm_timeout"
            attempts.append(
                build_attempt(
                    attempt_index=attempt_index,
                    model=model,
                    status="timeout",
                    raw_response="",
                    answer_text="",
                    config=llm_config,
                    error=str(exc),
                )
            )
            if not retry_on_timeout:
                break
        except Exception as exc:  # noqa: BLE001 - preserve local LLM failures in eval output.
            final_status = "llm_exception"
            failure_type = "llm_exception"
            attempts.append(
                build_attempt(
                    attempt_index=attempt_index,
                    model=model,
                    status="exception",
                    raw_response="",
                    answer_text="",
                    config=llm_config,
                    error=str(exc),
                )
            )
            break

        if attempt_index < max_attempts:
            if not retry_with_same_model and current_model_index + 1 < len(models):
                current_model_index += 1
        else:
            break

    retry_count = max(0, len(attempts) - 1)
    selected_model = str(attempts[-1]["model"]) if attempts else models[0]
    return {
        "case_id": case_id,
        "retrieval_mode": retrieval_mode,
        "llm_called": bool(attempts),
        "llm_model": selected_model,
        "answer_text": final_answer,
        "answer_generated": bool(final_answer),
        "status": final_status,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "retry_count": retry_count,
        "failure_type": failure_type,
        "raw_response_preview": attempts[-1]["raw_response_preview"] if attempts else "",
        "raw_response_saved": bool(llm_config.get("save_raw_llm_response", True)),
    }


def ollama_generate(prompt: str, model: str, config: dict[str, Any]) -> dict[str, Any]:
    base_url = str(config.get("base_url", "http://localhost:11434"))
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(config.get("temperature", 0)),
            "num_predict": int(config.get("num_predict", 512)),
        },
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout_sec = int(config.get("timeout_sec", 120))
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:  # noqa: S310 - local Ollama only.
            body = response.read().decode("utf-8", errors="replace")
    except TimeoutError:
        raise
    except urllib.error.URLError as exc:
        raise RuntimeError(f"local Ollama request failed: {exc}") from exc
    parsed = json.loads(body)
    return {
        "raw_response": str(parsed.get("response", "")),
        "ollama_done": parsed.get("done"),
        "raw_json_preview": truncate(body, int(config.get("max_raw_response_chars", 4000))),
    }


def clean_answer(raw_text: str, config: dict[str, Any]) -> str:
    text = str(raw_text or "")
    if bool(config.get("strip_thinking_tags", True)):
        text = strip_thinking_tags(text)
    return text.strip() if bool(config.get("treat_whitespace_as_empty", True)) else text


def is_empty_answer(answer_text: str, config: dict[str, Any] | None = None) -> bool:
    cfg = config or {}
    text = clean_answer(answer_text, cfg)
    if bool(cfg.get("treat_whitespace_as_empty", True)) and not text.strip():
        return True
    min_chars = int(cfg.get("min_answer_chars", 1))
    return len(text.strip()) < min_chars


def strip_thinking_tags(text: str) -> str:
    value = str(text or "")
    patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"```thinking.*?```",
    ]
    for pattern in patterns:
        value = re.sub(pattern, "", value, flags=re.DOTALL | re.IGNORECASE)
    return value.strip()


def build_attempt(
    *,
    attempt_index: int,
    model: str,
    status: str,
    raw_response: str,
    answer_text: str,
    config: dict[str, Any],
    error: str,
) -> dict[str, Any]:
    max_raw = int(config.get("max_raw_response_chars", 4000))
    return {
        "attempt_index": attempt_index,
        "model": model,
        "status": status,
        "raw_response_preview": truncate(raw_response, max_raw),
        "answer_preview": truncate(answer_text, int(config.get("max_answer_preview_chars", 800))),
        "answer_char_count": len(str(answer_text or "").strip()),
        "error": error,
    }


def make_not_called_result(
    *,
    case_id: str,
    retrieval_mode: str,
    failure_type: str,
    reason: str,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "retrieval_mode": retrieval_mode,
        "llm_called": False,
        "llm_model": "",
        "answer_text": "",
        "answer_generated": False,
        "status": failure_type,
        "attempts": [],
        "attempt_count": 0,
        "retry_count": 0,
        "failure_type": failure_type,
        "raw_response_preview": "",
        "raw_response_saved": False,
        "reason": reason,
    }


def truncate(text: str, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 3)] + "..."
