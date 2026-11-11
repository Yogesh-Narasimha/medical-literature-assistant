# Evaluation

`questions.jsonl` holds hand-written questions with a gold drug and acceptable label sections (`drug: null` = out of scope, should be refused).

```bash
python -m eval.run_eval retrieval   # Recall@k, MRR, score distribution
python -m eval.run_eval agent       # tool routing, citations, refusals
```
