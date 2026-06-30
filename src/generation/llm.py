from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG_PATH = PROJECT_ROOT / "config" / "model.yaml"


class OllamaLLM:
    def __init__(self, config_path: Path = MODEL_CONFIG_PATH) -> None:
        config = _load_model_config(config_path)
        self.model = str(config.get("llm_model") or "").strip()
        self.base_url = str(config.get("ollama_base_url") or "http://localhost:11434").strip()
        if not self.model:
            raise ValueError("config/model.yaml has an empty llm_model.")

    def generate(self, prompt: str) -> str:
        url = f"{self.base_url.rstrip('/')}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0,
                "num_predict": 768,
            },
        }
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            request = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                body = response.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            text = str(parsed.get("response", "")).strip()
            return strip_thinking(text)
        except Exception as exc:  # noqa: BLE001 - expose local model/API failures clearly.
            raise RuntimeError(
                f"Failed to generate answer with Ollama model '{self.model}' at {self.base_url}: {exc}"
            ) from exc


def strip_thinking(text: str) -> str:
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _load_model_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Invalid model config: {config_path}")
    return config
