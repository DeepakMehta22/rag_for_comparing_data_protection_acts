"""LLM-as-judge evaluation metrics (RAGAS-style + FActScore-style), implemented directly.

faithfulness       claims in the answer supported by the RETRIEVED context
answer_relevancy   cosine(question, questions regenerated from the answer)
context_precision  rank-weighted share of retrieved chunks that are useful (average precision)
context_recall     share of a reference answer's sentences found in the context (needs reference)
factscore          atomic facts supported by the WHOLE knowledge base (top-3 evidence per fact)
"""
import numpy as np

from config import JUDGE_MODEL
from llm import complete_json


def _verdicts(raw, n):
    """Normalise a judge reply into a list of n booleans."""
    v = raw.get("verdicts") if isinstance(raw, dict) else raw
    if not isinstance(v, list):
        return [False] * n
    v = [bool(x) for x in v][:n]
    return v + [False] * (n - len(v))


def extract_claims(answer):
    r = complete_json(
        f"Answer:\n{answer}\n\nBreak the answer into atomic, self-contained factual claims. Resolve pronouns, "
        "drop citation markers like [1], skip headings and filler. Return a JSON list of strings.",
        "You decompose text into atomic facts.", JUDGE_MODEL)
    return [c for c in r if isinstance(c, str)] if isinstance(r, list) else []


def _verify(items):
    """items: list of (claim, evidence). Returns list of bool."""
    if not items:
        return []
    body = "\n\n".join(f"### Claim {i}\n{c}\n--- Evidence {i}\n{e}" for i, (c, e) in enumerate(items, 1))
    r = complete_json(
        body + f"\n\nFor each claim, is it directly stated or clearly entailed by ITS evidence? "
        f'Return {{"verdicts": [true/false, ...]}} with exactly {len(items)} entries in order.',
        "You are a strict fact-checker. Judge only from the given evidence, never from your own knowledge.",
        JUDGE_MODEL, 1000)
    return _verdicts(r, len(items))


def faithfulness(claims, contexts):
    ev = "\n\n".join(c["text"] for c in contexts)
    v = _verify([(c, ev) for c in claims])
    return (sum(v) / len(v) if v else None), v


def factscore(claims, kb):
    items = []
    for c in claims:
        hits = kb.search(c, k=3)
        items.append((c, "\n\n".join(h["text"] for h in hits)))
    v = _verify(items)
    return (sum(v) / len(v) if v else None), v


def answer_relevancy(question, answer, kb, n=3):
    r = complete_json(
        f"Answer:\n{answer}\n\nWrite {n} different questions that this answer would be a direct response to. "
        'Also say whether the answer is evasive/noncommittal (e.g. "I don\'t know"). '
        'Return {"questions": [...], "noncommittal": true/false}.',
        "You generate questions from answers.", JUDGE_MODEL)
    if not isinstance(r, dict) or not r.get("questions") or r.get("noncommittal"):
        return 0.0
    embs = kb.embed([question] + r["questions"])
    return float(np.clip(np.mean(embs[1:] @ embs[0]), 0, 1))


def context_precision(question, answer, contexts, reference=None):
    target = reference or answer
    body = "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(contexts, 1))
    r = complete_json(
        f"Question: {question}\nGround truth / answer: {target}\n\nChunks (in retrieval order):\n{body}\n\n"
        f'For each chunk, is it useful for arriving at the answer above? Return {{"verdicts": [true/false, ...]}} '
        f"with exactly {len(contexts)} entries.",
        "You judge retrieval relevance.", JUDGE_MODEL)
    v = _verdicts(r, len(contexts))
    if not any(v):
        return 0.0, v
    hits, ap = 0, 0.0
    for k, rel in enumerate(v, 1):
        if rel:
            hits += 1
            ap += hits / k
    return ap / hits, v


def context_recall(reference, contexts):
    ev = "\n\n".join(c["text"] for c in contexts)
    r = complete_json(
        f"Reference answer:\n{reference}\n\nContext:\n{ev}\n\nSplit the reference answer into sentences. For each, "
        'can it be attributed to the context? Return {"verdicts": [true/false, ...]}.',
        "You judge whether context covers a reference answer.", JUDGE_MODEL)
    v = r.get("verdicts") if isinstance(r, dict) else None
    return (sum(map(bool, v)) / len(v)) if v else None


def evaluate(kb, result, reference=None):
    q, a, ctx = result["question"], result["answer"], result["contexts"]
    claims = extract_claims(a)
    f, f_v = faithfulness(claims, ctx)
    fs, fs_v = factscore(claims, kb)
    cp, cp_v = context_precision(q, a, ctx, reference)
    return {
        "faithfulness": f,
        "answer_relevancy": answer_relevancy(q, a, kb),
        "context_precision": cp,
        "context_recall": context_recall(reference, ctx) if reference else None,
        "factscore": fs,
        "details": {
            "claims": [{"claim": c, "in_retrieved_context": x, "in_knowledge_base": y}
                       for c, x, y in zip(claims, f_v, fs_v)],
            "chunk_relevant": cp_v,
        },
    }
