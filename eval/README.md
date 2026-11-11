# Evaluation (Day 4)

`eval/questions.jsonl` will hold hand-checked questions, one per line:

```json
{"id": "q001", "question": "Can ibuprofen be taken with warfarin?", "relevant_doc_ids": ["fda:<set_id>"], "relevant_sections": ["drug_interactions"]}
```

Metrics planned: Recall@k and MRR for retrieval, faithfulness and answer relevance for generated answers.
