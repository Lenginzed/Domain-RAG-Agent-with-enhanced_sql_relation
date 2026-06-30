# Human Review Workflow

This project includes a local workflow for inspecting answer quality and turning failure cases into review tasks.

The workflow supports human review, but does not replace human judgment.

## 1. Answer Eval

Answer eval runs a small local set of cases through retrieval and, when evidence is sufficient, local answer generation. It records:

- retrieval mode;
- evidence gate status;
- top sources;
- answer status;
- citation checker result;
- evidence quality score;
- relation path availability;
- human review decision signals.

The public repository includes sample output shapes only. Full local eval artifacts are excluded.

## 2. Local LLM Reliability Wrapper

Answer generation can use a local Ollama-compatible model. The reliability wrapper records:

- model name;
- attempt count;
- retry count;
- answer generated flag;
- failure type;
- raw response preview;
- prompt preview.

If a local model is unavailable or returns an empty answer, the workflow records that condition instead of fabricating an answer.

## 3. Retry on Empty Answer

The local wrapper can retry once when an answer is empty. Empty answer detection treats these cases as empty:

- empty string;
- whitespace only;
- response that only contains thinking-style tags after stripping;
- response shorter than a configured minimum after cleanup.

Retry metadata is preserved for review.

## 4. Failure Taxonomy

Common labels include:

- `retrieval_insufficient`;
- `llm_unavailable`;
- `llm_timeout`;
- `llm_empty_answer`;
- `llm_exception`;
- `citation_failure`;
- `unsupported_claims_present`;
- `weak_claims_present`;
- `medium_or_lower_quality`.

These labels are used for filtering and review queue construction.

## 5. Review Task Export

Failure triage creates review tasks with:

- case ID;
- question;
- retrieval mode;
- failure types;
- evidence quality;
- review reasons;
- answer preview;
- top sources;
- unsupported or weak claims;
- relation paths;
- raw response preview;
- prompt preview;
- LLM attempt summary;
- suggested review action;
- blank manual fields.

The manual fields are intentionally empty so a reviewer can fill them later.

## 6. Manual Adjudication Import

The adjudication step reads a filled review CSV and validates:

- `manual_judgment`;
- `manual_citation_judgment`;
- `manual_should_refuse`;
- `manual_fix_suggestion`;
- `manual_notes`.

It then infers adjudication status and fix categories such as:

- prompt fix;
- retrieval fix;
- citation checker fix;
- LLM reliability fix;
- refusal policy review;
- relation fusion review;
- no action needed.

## 7. Calibration Notes

Adjudication summaries can produce local calibration notes:

- prompt improvement candidates;
- citation checker false positive examples;
- retrieval/fusion issue examples;
- empty answer examples;
- negative refusal confirmation examples.

These notes are engineering inputs, not final labels unless reviewed by a human.

## 8. Streamlit Browser

The Streamlit UI can inspect answer eval records, filter by failure type, view retry attempts, browse review tasks, and inspect adjudication summaries. It does not run new evals when browsing existing reports.
