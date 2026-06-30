from __future__ import annotations

from langchain_core.documents import Document


RAG_SYSTEM_INSTRUCTIONS = """你是 Domain-RAG Agent 的 V1 问答模块。
只能根据提供的检索资料回答问题。
如果检索资料不足，必须明确说明“当前知识库没有足够依据”。
回答事实、代码位置、配置含义或实验结论时必须给出来源编号，例如 [S1]。
不允许编造不存在的文件、指标、实验结果或代码行为。
请优先使用中文回答，保持简洁。"""


def build_rag_prompt(question: str, documents: list[Document]) -> str:
    evidence = format_evidence(documents)
    return f"""{RAG_SYSTEM_INSTRUCTIONS}

检索资料：
{evidence}

用户问题：
{question}

请输出：
1. 回答
2. 来源依据"""


def format_evidence(documents: list[Document]) -> str:
    if not documents:
        return "没有检索到资料。"

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        metadata = document.metadata
        content = document.page_content.strip()
        if len(content) > 1800:
            content = f"{content[:1800]}\n...[truncated]"
        blocks.append(
            "\n".join(
                [
                    f"[S{index}]",
                    f"source: {metadata.get('source', 'unknown_source')}",
                    f"chunk_id: {metadata.get('chunk_id', '')}",
                    f"section: {metadata.get('section', 'default')}",
                    f"doc_type: {metadata.get('doc_type', 'unknown')}",
                    "content:",
                    content,
                ]
            )
        )
    return "\n\n".join(blocks)
