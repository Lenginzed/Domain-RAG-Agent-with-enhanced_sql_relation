# Human Review Workflow

The project includes a rule-based workflow for inspecting answer quality and turning failure cases into review tasks.

## Steps

1. Run answer evaluation on a small local set.
2. Record evidence gate status and retrieval diagnostics.
3. Capture local LLM attempts, retry count, prompt preview, and raw response preview.
4. Run citation checking.
5. Compute evidence quality.
6. Classify failure types.
7. Export review tasks.
8. Import manual judgments.
9. Summarize adjudication status and fix categories.
10. Produce calibration notes for future improvements.

## Failure Taxonomy

Common failure labels include:

- `retrieval_insufficient`
- `llm_unavailable`
- `llm_timeout`
- `llm_empty_answer`
- `llm_exception`
- `citation_failure`
- `unsupported_claims_present`
- `weak_claims_present`
- `medium_or_lower_quality`

## Manual Review Fields

Review tasks include blank manual fields:

- `manual_judgment`
- `manual_citation_judgment`
- `manual_should_refuse`
- `manual_fix_suggestion`
- `manual_notes`

## Adjudication

Manual labels are imported into an adjudication summary. The summary can identify pending tasks, accepted answers, valid refusals, citation checker issues, retrieval issues, prompt/retry issues, and discussion-needed cases.

This workflow is intended for local inspection and engineering calibration. It is not a fully automated review system.

