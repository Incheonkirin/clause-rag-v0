"""query → top-K 검색."""

from __future__ import annotations

from pathlib import Path

from src.rag.embed import get_collection, get_embedder


def retrieve(query: str, persist_dir: Path, k: int = 3) -> list[dict]:
    embedder = get_embedder()
    q_emb = embedder.encode([query], normalize_embeddings=True).tolist()
    col = get_collection(persist_dir)
    res = col.query(query_embeddings=q_emb, n_results=k)

    out = []
    for i in range(len(res["ids"][0])):
        out.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "distance": float(res["distances"][0][i]),
            "metadata": res["metadatas"][0][i],
        })
    return out
