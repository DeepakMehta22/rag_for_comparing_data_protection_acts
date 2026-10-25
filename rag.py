"""Retrieval + grounded generation, including cross-document comparison."""
import json
import os
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from config import DOCS, EMBED_MODEL, GEN_MODEL, INDEX_DIR, JUDGE_MODEL, MIN_SCORE, TOP_K
from llm import complete, complete_json

REFUSAL = ("I can only answer from the GDPR and India's DPDP Act 2023, and I couldn't find "
           "support for this question in those two documents.")


class KnowledgeBase:
    def __init__(self):
        with open(os.path.join(INDEX_DIR, "chunks.json")) as f:
            self.chunks = json.load(f)
        self.emb = np.load(os.path.join(INDEX_DIR, "emb.npy"))
        self.model = SentenceTransformer(EMBED_MODEL)

    def embed(self, texts):
        return self.model.encode(texts, normalize_embeddings=True)

    def search(self, query, k=TOP_K, doc=None):
        sims = self.emb @ self.embed([query])[0]
        out = []
        for i in np.argsort(-sims):
            if doc and self.chunks[i]["doc"] != doc:
                continue
            out.append({**self.chunks[i], "score": float(sims[i])})
            if len(out) == k:
                break
        return out


PLAN_SYSTEM = f"""You route questions about two documents: {DOCS['GDPR']['title']} (id GDPR) and {DOCS['DPDP']['title']} (id DPDP).
Terminology differs: GDPR uses {DOCS['GDPR']['terms']}; DPDP uses {DOCS['DPDP']['terms']}.
Return JSON: {{"intent": "simple" | "comparison", "query": "<standalone search query>", "subqueries": {{"GDPR": "<search query using GDPR terminology>", "DPDP": "<search query using DPDP terminology>"}}}}
Use "comparison" when the user asks to compare/contrast/differences/similarities of a term or concept across the two documents. For "simple" questions subqueries may be empty."""

ANSWER_SYSTEM = """You answer questions ONLY from the numbered excerpts of two documents: the EU GDPR and India's Digital Personal Data Protection (DPDP) Act 2023.
Rules:
- Use only information in the excerpts. Never use outside knowledge, even if you know the answer.
- Cite excerpts inline like [1] or [2][3].
- If the excerpts do not contain the answer, reply with exactly: NOT_IN_DOCUMENTS
- Equivalent terms: controller = data fiduciary, data subject = data principal, supervisory authority = Data Protection Board (functional analogy only; say so if relevant).
- For comparisons use these headings: **GDPR**, **DPDP Act**, **Similarities**, **Differences**. If one document does not address the term, say so explicitly rather than guessing.
- Be concise and precise. This is information from the texts, not legal advice."""


def plan(question):
    r = complete_json(f"Question: {question}", PLAN_SYSTEM, JUDGE_MODEL)
    if not isinstance(r, dict) or r.get("intent") not in ("simple", "comparison"):
        return {"intent": "simple", "query": question, "subqueries": {}}
    return r


def answer(kb: KnowledgeBase, question: str, k: int = TOP_K):
    t0 = time.time()
    p = plan(question)
    if p["intent"] == "comparison":
        per_doc = max(3, k // 2 + 1)
        subs = p.get("subqueries") or {}
        ctx = []
        for doc in DOCS:  # balanced retrieval so neither document dominates
            ctx += kb.search(subs.get(doc) or question, per_doc, doc=doc)
    else:
        ctx = kb.search(p.get("query") or question, k)

    out = {"question": question, "intent": p["intent"], "plan": p, "contexts": ctx, "refused": False}

    if not ctx or max(c["score"] for c in ctx) < MIN_SCORE:  # out-of-scope guard
        out.update(answer=REFUSAL, refused=True)
    else:
        block = "\n\n".join(
            f"[{i}] ({c['doc']}, {c['section'] or 'n/a'}, p.{c['page']})\n{c['text']}" for i, c in enumerate(ctx, 1)
        )
        text = complete(f"Excerpts:\n{block}\n\nQuestion: {question}", ANSWER_SYSTEM, GEN_MODEL)
        if "NOT_IN_DOCUMENTS" in text:
            out.update(answer=REFUSAL, refused=True)
        else:
            out["answer"] = text
    out["latency_s"] = round(time.time() - t0, 2)
    return out
