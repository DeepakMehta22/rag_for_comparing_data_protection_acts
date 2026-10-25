# GDPR vs DPDP Act: grounded RAG

Ask questions, or ask for comparisons, across exactly two documents:
the EU GDPR and India's Digital Personal Data Protection Act, 2023. Answers are confined to them,
and each answer is scored with RAG evaluation metrics.

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...        # Windows: set ANTHROPIC_API_KEY=...
```
Download and save into `data/`:
- `gdpr.pdf`: Regulation (EU) 2016/679, from eur-lex.europa.eu
- `dpdp_act_2023.pdf`: Digital Personal Data Protection Act, 2023, from meity.gov.in

```bash
python ingest.py            # builds ./index (or click the button in the UI)
streamlit run app.py
```
Optional: `GEN_MODEL` and `JUDGE_MODEL` env vars change the Claude models.

## How it works
- **ingest.py**: PDF to line units, packed into ~1200-char chunks with overlap, tagged with doc, page and Article/Section. Embedded locally (MiniLM), stored as numpy.
- **rag.py**: a router decides simple vs comparison. Comparisons run one search per document, using each law's terminology (controller vs data fiduciary), so both sides are always represented.
- **Confinement**, three layers: a similarity floor (`MIN_SCORE`) refuses off-topic questions without calling the LLM; the prompt forbids outside knowledge and requires citations; the model answers `NOT_IN_DOCUMENTS` when the excerpts lack the answer.
- **metrics.py**: LLM-judge metrics.

| Metric | What it checks | Needs reference? |
|---|---|---|
| Faithfulness | answer claims supported by retrieved chunks | no |
| Answer relevancy | question vs questions regenerated from the answer | no |
| Context precision | useful chunks, rank-weighted | optional |
| Context recall | reference answer covered by retrieved chunks | yes |
| FActScore | atomic facts supported anywhere in the KB (top-3 evidence per fact) | no |

Faithfulness vs FActScore: a fact that is true in the documents but wasn't retrieved lowers faithfulness
but not FActScore, which points to a retrieval problem rather than a hallucination.

## Tuning ideas
Raise top-k, add a reranker or BM25 hybrid search, or add a CSV of questions with reference answers for batch evaluation.
