from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = PROJECT_ROOT / "storage" / "logs" / "ollama_check.json"
DEFAULT_BASE_URL = "http://localhost:11434"


def run_command(args: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            args,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        return {
            "command": args,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except FileNotFoundError as exc:
        return {
            "command": args,
            "returncode": None,
            "stdout": "",
            "stderr": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": args,
            "returncode": None,
            "stdout": (exc.stdout or "").strip() if isinstance(exc.stdout, str) else "",
            "stderr": f"Command timed out after {timeout}s.",
        }


def fetch_ollama_tags(base_url: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            body = response.read().decode("utf-8", errors="replace")
            return {
                "ok": 200 <= response.status < 300,
                "status": response.status,
                "url": url,
                "body": json.loads(body),
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        return {
            "ok": False,
            "status": exc.code,
            "url": url,
            "body": None,
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001 - this is an environment probe.
        return {
            "ok": False,
            "status": None,
            "url": url,
            "body": None,
            "error": str(exc),
        }


def parse_model_names_from_list(stdout: str) -> list[str]:
    names: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped or stripped.upper().startswith("NAME"):
            continue
        parts = stripped.split()
        if parts:
            names.append(parts[0])
    return names


def get_api_models(api_result: dict[str, Any]) -> list[dict[str, Any]]:
    if not api_result.get("ok") or not isinstance(api_result.get("body"), dict):
        return []
    models = api_result["body"].get("models", [])
    return models if isinstance(models, list) else []


def get_model_names(api_models: list[dict[str, Any]], list_result: dict[str, Any]) -> list[str]:
    names: list[str] = []
    for model in api_models:
        name = model.get("name") or model.get("model")
        if isinstance(name, str) and name:
            names.append(name)
    if not names and list_result.get("stdout"):
        names = parse_model_names_from_list(list_result["stdout"])
    return list(dict.fromkeys(names))


def build_capability_map(api_models: list[dict[str, Any]]) -> dict[str, list[str]]:
    capabilities: dict[str, list[str]] = {}
    for model in api_models:
        name = model.get("name") or model.get("model")
        raw_caps = model.get("capabilities", [])
        if isinstance(name, str):
            capabilities[name] = raw_caps if isinstance(raw_caps, list) else []
    return capabilities


def select_llm_model(model_names: list[str], capabilities: dict[str, list[str]]) -> str:
    if "qwen3:14b" in model_names and is_generation_model("qwen3:14b", capabilities):
        return "qwen3:14b"

    qwen_candidates = [
        name
        for name in model_names
        if "qwen" in name.lower() and is_generation_model(name, capabilities)
    ]
    qwen_candidates.sort(key=lambda name: (0 if "qwen3" in name.lower() else 1, name))
    return qwen_candidates[0] if qwen_candidates else ""


def is_generation_model(model_name: str, capabilities: dict[str, list[str]]) -> bool:
    caps = capabilities.get(model_name)
    lowered = model_name.lower()
    if caps:
        return "completion" in caps or "tools" in caps or "thinking" in caps
    return "embedding" not in lowered and "embed" not in lowered


def select_embedding_model(model_names: list[str], capabilities: dict[str, list[str]]) -> str:
    priority_terms = ["qwen3-embedding", "embeddinggemma", "all-minilm"]
    for term in priority_terms:
        candidates = [
            name
            for name in model_names
            if term in name.lower() and is_embedding_model(name, capabilities)
        ]
        if candidates:
            candidates.sort(key=lambda name: (0 if name.endswith(":latest") else 1, name))
            return candidates[0]
    return ""


def is_embedding_model(model_name: str, capabilities: dict[str, list[str]]) -> bool:
    caps = capabilities.get(model_name)
    lowered = model_name.lower()
    if caps:
        return "embedding" in caps
    return (
        "embedding" in lowered
        or "embed" in lowered
        or "embeddinggemma" in lowered
        or "all-minilm" in lowered
    )


def print_command_result(title: str, result: dict[str, Any]) -> None:
    print(f"\n== {title} ==")
    print(f"command: {' '.join(result.get('command', []))}")
    print(f"returncode: {result.get('returncode')}")
    if result.get("stdout"):
        print(result["stdout"])
    if result.get("stderr"):
        print(f"stderr: {result['stderr']}")


def main() -> int:
    ollama_path = shutil.which("ollama")
    command_exists = ollama_path is not None

    print("== Ollama environment check ==")
    print(f"ollama command found: {command_exists}")
    print(f"ollama path: {ollama_path or ''}")

    list_result = run_command(["ollama", "list"]) if command_exists else {}
    ps_result = run_command(["ollama", "ps"]) if command_exists else {}
    api_result = fetch_ollama_tags(DEFAULT_BASE_URL)

    if command_exists:
        print_command_result("ollama list", list_result)
        print_command_result("ollama ps", ps_result)
    else:
        print("\nOllama command was not found on PATH.")

    print("\n== Ollama HTTP API ==")
    print(f"url: {api_result['url']}")
    print(f"reachable: {api_result['ok']}")
    print(f"status: {api_result['status']}")
    if api_result.get("error"):
        print(f"error: {api_result['error']}")

    api_models = get_api_models(api_result)
    model_names = get_model_names(api_models, list_result)
    capabilities = build_capability_map(api_models)
    selected_llm = select_llm_model(model_names, capabilities)
    selected_embedding = select_embedding_model(model_names, capabilities)

    print("\n== Local models ==")
    if model_names:
        for name in model_names:
            print(f"- {name}")
    else:
        print("(none detected)")

    print("\n== Recommended model selection ==")
    print(f"llm_model: {selected_llm or '(empty)'}")
    print(f"embedding_model: {selected_embedding or '(empty)'}")

    warnings: list[str] = []
    if not selected_llm:
        warnings.append("No qwen/qwen3 generation model was detected.")
    if not selected_embedding:
        warnings.append(
            "No supported embedding model was detected. Run: ollama pull qwen3-embedding "
            "or ollama pull embeddinggemma"
        )
        print("\nNo supported embedding model was detected.")
        print("Please run one of:")
        print("  ollama pull qwen3-embedding")
        print("  ollama pull embeddinggemma")

    result = {
        "checked_at": datetime.now(timezone.utc).isoformat(),
        "ollama_command_found": command_exists,
        "ollama_path": ollama_path or "",
        "ollama_list": list_result,
        "ollama_ps": ps_result,
        "ollama_api": api_result,
        "local_models": model_names,
        "selected_llm_model": selected_llm,
        "selected_embedding_model": selected_embedding,
        "warnings": warnings,
    }

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved check result to: {LOG_PATH}")

    return 0 if command_exists and api_result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
