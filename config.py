"""Central configuration for the privacy-law RAG project."""
import os

DATA_DIR = "data"
INDEX_DIR = "index"

# The ONLY two documents in the knowledge base. Put the PDFs in ./data with these file names.
DOCS = {
    "GDPR": {
        "file": "gdpr.pdf",
        "title": "EU General Data Protection Regulation (Regulation (EU) 2016/679)",
        "terms": "controller, processor, data subject, supervisory authority",
    },
    "DPDP": {
        "file": "dpdp_act_2023.pdf",
        "title": "India Digital Personal Data Protection Act, 2023",
        "terms": "data fiduciary, data processor, data principal, Data Protection Board",
    },
}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL = os.getenv("GEN_MODEL", "claude-sonnet-5-5")     # answers
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-haiku-5-5")  # planning + evaluation (cheaper)

CHUNK_CHARS = 1200
CHUNK_OVERLAP = 200
TOP_K = 5
MIN_SCORE = 0.25  # best cosine similarity below this => question treated as out of scope
