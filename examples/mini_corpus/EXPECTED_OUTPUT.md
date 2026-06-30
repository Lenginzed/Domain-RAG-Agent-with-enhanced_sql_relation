# Expected Output

## Build Command

```powershell
conda run -n Lenginzed_RAG python scripts/build_mini_demo_index_v51.py
```

Expected summary:

```text
documents = 4
chunks = 4
events = 4
entities ~= 40
event_entities ~= 53
```

Small count differences are acceptable if the synthetic extraction rules are adjusted.

## Run Command

```powershell
conda run -n Lenginzed_RAG python scripts/run_mini_demo_v51.py
```

Expected retrieval hits:

```text
EventDrivenReward -> event_driven_reward.py
HierarchySelfplay -> HierarchySelfplay.yaml
reward/risk -> stage13_risk_report.md
```

## Relation Path Example

```text
query_entity:event_driven_reward
-> seed_entity:event_driven_reward
-> seed_event:examples/mini_corpus/docs/event_driven_reward.py
```

## Generated Files

```text
tmp/mini_demo/mini_demo_relation_index.db
tmp/mini_demo/mini_demo_results.json
tmp/mini_demo/mini_demo_results.md
```

These files are local generated artifacts and are ignored by git.
