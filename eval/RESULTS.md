# Retrieval Evaluation Results

- Eval set: 37 questions across 4 insurance policy documents
- Context budget per question: 6000 characters
- Metric: **hit rate** — the labelled evidence clause is contained in the retrieved context

| Retriever | Hit rate |
|---|---|
| Fixed char-window chunks + TF-IDF (baseline) | 33/37 (**89.2%**) |
| Semantic clause-aware chunking + TF-IDF | 32/37 (**86.5%**) |
| Semantic chunking + embedding re-ranker (RRF) | 36/37 (**97.3%**) |

Best per-category breakdown (reranked):

- benefit: 10/10
- claim_process: 1/1
- cost_sharing: 5/5
- definition: 9/9
- renewal: 4/5
- waiting_period: 7/7
