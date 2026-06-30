from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEMO_FILES = {
    "data/raw/notes/rag_intro.md": """# RAG 基础说明

synthetic demo data: 本文件是 V1 lightweight check 使用的合成示例资料，不代表真实论文或真实实验。

RAG（Retrieval-Augmented Generation）是一种先检索资料、再基于证据生成回答的方法。
在本项目中，回答必须引用检索到的来源，资料不足时应说明当前知识库没有足够依据。
""",
    "data/raw/notes/aircombat_intro.md": """# 敌机意图预测简介

synthetic demo data: 本文件是 V1 lightweight check 使用的合成示例资料，不代表真实空战资料。

敌机意图预测是指根据敌机的历史轨迹、相对位置、速度变化和战术动作，推测其可能的下一步行为。
在多无人机空战场景中，意图预测可用于辅助编队决策、威胁评估和策略选择。
""",
    "data/raw/code/demo_policy.py": '''"""synthetic demo data for V1 lightweight checks."""


class DemoEvasivePolicy:
    """A tiny synthetic policy used only for retrieval testing."""

    def select_action(self, threat_level: float) -> str:
        if threat_level > 0.7:
            return "evade"
        return "hold_course"


def predict_opponent_intent(relative_distance: float, closing_speed: float) -> str:
    if relative_distance < 1200 and closing_speed > 80:
        return "possible_attack"
    return "unknown"
''',
    "data/raw/configs/demo_train.yaml": """# synthetic demo data for V1 lightweight checks
experiment_id: synthetic_demo_train
seed: 42
stage: v1_smoke
method: dense_rag_demo
training:
  batch_size: 8
  max_steps: 20
policy:
  name: DemoEvasivePolicy
  threat_threshold: 0.7
""",
    "data/raw/experiments/demo_results.csv": """experiment_id,seed,method,metric,value,note
synthetic_demo_eval,42,dense_rag_demo,accuracy,0.75,synthetic demo data only
synthetic_demo_eval,42,dense_rag_demo,latency_ms,123,synthetic demo data only
""",
}


def create_demo_data() -> list[Path]:
    created: list[Path] = []
    for relative_path, content in DEMO_FILES.items():
        path = PROJECT_ROOT / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created.append(path)
    return created


def main() -> int:
    created = create_demo_data()
    print("Created synthetic demo data files:")
    for path in created:
        print(f"- {path.relative_to(PROJECT_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
