"""Build the vector index from the two PDFs.   Usage: python ingest.py"""
import json
import os
import re

import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from config import CHUNK_CHARS, CHUNK_OVERLAP, DATA_DIR, DOCS, EMBED_MODEL, INDEX_DIR

HEADING = re.compile(r"^(Article\s+\d+|CHAPTER\s+[IVXL]+|Section\s+\d+|CHAPTER\s+[IVXL]+|\d{1,2}\.\s+[A-Z])")


def read_units(path):
    """Return a list of (page_no, heading, line) units from a PDF."""
    units, heading = [], ""
    for pno, page in enumerate(PdfReader(path).pages, start=1):
        for line in (page.extract_text() or "").splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            m = HEADING.match(line)
            if m:
                heading = line[:80]
            units.append((pno, heading, line))
    return units


def chunk_units(doc_id, units):
    chunks, buf, size = [], [], 0

    def flush():
        if not buf:
            return
        chunks.append({
            "doc": doc_id,
            "page": buf[0][0],
            "section": buf[0][1],
            "text": " ".join(u[2] for u in buf),
        })

    for u in units:
        if size + len(u[2]) > CHUNK_CHARS and buf:
            flush()
            # keep a tail of the previous chunk as overlap
            tail, tsize = [], 0
            for t in reversed(buf):
                if tsize + len(t[2]) > CHUNK_OVERLAP:
                    break
                tail.insert(0, t)
                tsize += len(t[2])
            buf, size = tail, tsize
        buf.append(u)
        size += len(u[2])
    flush()
    return chunks


def build_index():
    all_chunks = []
    for doc_id, meta in DOCS.items():
        path = os.path.join(DATA_DIR, meta["file"])
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {path}. Download the PDF and save it with exactly this name.")
        chunks = chunk_units(doc_id, read_units(path))
        print(f"{doc_id}: {len(chunks)} chunks")
        all_chunks += chunks
    for i, c in enumerate(all_chunks):
        c["id"] = i
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode([c["text"] for c in all_chunks], normalize_embeddings=True, show_progress_bar=True)
    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "emb.npy"), emb)
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w") as f:
        json.dump(all_chunks, f)
    return len(all_chunks)


if __name__ == "__main__":
    print("Indexed", build_index(), "chunks")
