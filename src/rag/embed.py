"""BGE-M3 임베딩 + Chroma 인덱싱.

chunk size 300자, overlap 50자. 단순 split (조 단위 split 아님 — v1).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
COLLECTION = "clause_v0"
EMBED_MODEL = "BAAI/bge-m3"


def split_chunks(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """단순 슬라이딩 윈도우. 문장 경계 미고려 (v1에서 split)."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunks.append(text[start:end])
        if end >= n:
            break
        start += size - overlap
    return chunks


_embedder = None
_client = None
_collection = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print(f"  [embed] {EMBED_MODEL} 로드 ...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection(persist_dir: Path):
    global _client, _collection
    if _collection is None:
        import chromadb
        _client = chromadb.PersistentClient(path=str(persist_dir))
        _collection = _client.get_or_create_collection(
            name=COLLECTION,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def index_clause(md_path: Path, persist_dir: Path) -> dict[str, Any]:
    """약관 markdown을 chunk + 임베딩 + Chroma 저장."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    text = md_path.read_text(encoding="utf-8")
    chunks = split_chunks(text)
    print(f"  [embed] {len(chunks)} chunks")

    embedder = get_embedder()
    embeddings = embedder.encode(chunks, normalize_embeddings=True).tolist()

    col = get_collection(persist_dir)
    # 기존 데이터 제거 (재실행 시 중복 방지)
    try:
        existing = col.get()["ids"]
        if existing:
            col.delete(ids=existing)
    except Exception:
        pass

    ids = [f"chunk_{i:03d}" for i in range(len(chunks))]
    col.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=[{"source": md_path.name, "chunk_idx": i} for i in range(len(chunks))],
    )
    return {"n_chunks": len(chunks), "collection": COLLECTION}
