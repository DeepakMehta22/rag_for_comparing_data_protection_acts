"""Streamlit UI.   Run: streamlit run app.py"""
import os

import pandas as pd
import streamlit as st

from dotenv import load_dotenv
load_dotenv()

from config import DATA_DIR, DOCS, INDEX_DIR, TOP_K
from metrics import evaluate

st.set_page_config(page_title="GDPR vs DPDP Q&A", layout="wide")
st.title("GDPR and DPDP Act: grounded Q&A")
st.caption("Answers come only from the EU GDPR and India's Digital Personal Data Protection Act, 2023. "
           "Not legal advice.")

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("Set up API Key and restart the app.")
    st.stop()

index_ready = os.path.exists(os.path.join(INDEX_DIR, "chunks.json"))
with st.sidebar:
    st.header("Knowledge base")
    for d, m in DOCS.items():
        ok = os.path.exists(os.path.join(DATA_DIR, m["file"]))
        st.write(f"{'✅' if ok else '❌'} **{d}**: `{DATA_DIR}/{m['file']}`")
    if st.button("Build / rebuild index"):
        from ingest import build_index
        with st.spinner("Chunking and embedding…"):
            try:
                st.success(f"Indexed {build_index()} chunks")
                st.cache_resource.clear()
                index_ready = True
            except FileNotFoundError as e:
                st.error(str(e))
    st.header("Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 3, 10, TOP_K)
    run_eval = st.toggle("Compute evaluation metrics", True,
                         help="Adds ~5 extra LLM calls per question.")
    reference = st.text_area("Reference answer (optional)",
                             help="Enables context recall and a stricter context precision for the next question.")

if not index_ready:
    st.info("Put the two PDFs in the `data/` folder, then click **Build / rebuild index** in the sidebar.")
    st.stop()


@st.cache_resource
def get_kb():
    from rag import KnowledgeBase
    return KnowledgeBase()


kb = get_kb()
from rag import answer  # noqa: E402

st.session_state.setdefault("history", [])
examples = [
    "What is the lawful basis for processing personal data?",
    "Compare how consent is defined and handled in both documents.",
    "What penalties can be imposed for a personal data breach?",
    "Compare the rights of a data subject and a data principal.",
]
cols = st.columns(len(examples))
picked = None
for c, ex in zip(cols, examples):
    if c.button(ex, use_container_width=True):
        picked = ex

question = st.chat_input("Ask about either document, or ask for a comparison") or picked

METRIC_HELP = {
    "faithfulness": "Share of answer claims supported by the retrieved chunks.",
    "answer_relevancy": "Similarity between the question and questions regenerated from the answer.",
    "context_precision": "Rank-weighted share of retrieved chunks that were useful.",
    "context_recall": "Share of the reference answer covered by retrieved chunks (needs a reference).",
    "factscore": "Share of atomic facts supported anywhere in the knowledge base.",
}

if question:
    with st.spinner("Retrieving and answering…"):
        res = answer(kb, question, top_k)
    metrics = None
    if run_eval and not res["refused"]:
        with st.spinner("Evaluating…"):
            metrics = evaluate(kb, res, reference.strip() or None)
    st.session_state.history.append({"res": res, "metrics": metrics})

for turn in reversed(st.session_state.history[-5:]):
    res, m = turn["res"], turn["metrics"]
    with st.container(border=True):
        st.markdown(f"**Q:** {res['question']}")
        st.caption(f"Query type: {res['intent']} · latency {res['latency_s']}s · {len(res['contexts'])} chunks")
        (st.warning if res["refused"] else st.markdown)(res["answer"])

        if m:
            st.subheader("Performance metrics")
            mc = st.columns(5)
            for col, name in zip(mc, METRIC_HELP):
                v = m[name]
                col.metric(name.replace("_", " ").title(), "n/a" if v is None else f"{v:.2f}",
                           help=METRIC_HELP[name])
            with st.expander("Claim-level verification"):
                st.dataframe(pd.DataFrame(m["details"]["claims"]), use_container_width=True)
        elif res["refused"]:
            st.caption("Metrics skipped: no grounded answer was produced.")

        with st.expander("Retrieved passages"):
            rel = (m or {}).get("details", {}).get("chunk_relevant", [])
            for i, c in enumerate(res["contexts"], 1):
                tag = "" if i > len(rel) else (" ✔ useful" if rel[i - 1] else " ✘ not useful")
                st.markdown(f"**[{i}] {c['doc']}** · {c['section'] or 'n/a'} · p.{c['page']} · "
                            f"sim {c['score']:.2f}{tag}")
                st.text(c["text"])

scored = [t["metrics"] for t in st.session_state.history if t["metrics"]]
if len(scored) > 1:
    st.subheader("Session averages")
    df = pd.DataFrame([{k: s[k] for k in METRIC_HELP} for s in scored])
    st.bar_chart(df.mean(numeric_only=True))
    st.dataframe(df.round(2), use_container_width=True)
