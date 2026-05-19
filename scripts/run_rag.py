"""RAG 5질문 batch + JSON + HTML 출력.

실행:
    uv run python scripts/run_rag.py
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.rag.answer import answer
from src.rag.embed import index_clause
from src.rag.retrieve import retrieve


def main():
    root = Path(__file__).resolve().parents[1]
    md = root / "data" / "clause_v0.md"
    persist = root / "data" / "chroma_store"
    qs = yaml.safe_load((root / "data" / "questions_v0.yaml").read_text(encoding="utf-8"))["questions"]

    # 1) 인덱싱
    print("[1] 약관 인덱싱")
    t0 = time.time()
    info = index_clause(md, persist)
    print(f"  {info['n_chunks']} chunks, {time.time()-t0:.1f}초")

    # 2) RAG 5질문
    print(f"\n[2] {len(qs)} 질문 RAG")
    results = []
    for i, q in enumerate(qs, 1):
        t0 = time.time()
        chunks = retrieve(q["question"], persist, k=3)
        ans = answer(q["question"], chunks)
        elapsed = time.time() - t0
        results.append({
            "code": q["code"],
            "question": q["question"],
            "expected": q["expected"],
            "retrieved_ids": [c["id"] for c in chunks],
            "retrieved_distances": [round(c["distance"], 3) for c in chunks],
            "answer": ans,
            "elapsed_sec": round(elapsed, 1),
        })
        print(f"  [{i}/{len(qs)}] {q['code']}  {elapsed:.1f}초")
        print(f"    Q: {q['question']}")
        print(f"    A: {ans[:120]}...")
        print()

    # 3) 저장
    out_dir = root / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")
    json_path = out_dir / f"rag_{today}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"generated_at": datetime.now().isoformat(),
                   "n_chunks": info["n_chunks"], "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"[저장] {json_path}")

    # 4) HTML
    html = _render(results, info["n_chunks"])
    html_path = root / "docs" / "index.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(html, encoding="utf-8")
    print(f"[site] {html_path}")


def _render(results, n_chunks) -> str:
    cards = "\n".join(_card(r) for r in results)
    return f"""<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="UTF-8"><title>Clause RAG v0 — 가상 자동차보험 약관 QA</title>
<style>
  body {{ font-family: -apple-system, "Apple SD Gothic Neo", sans-serif;
          max-width: 820px; margin: 30px auto; padding: 0 20px; line-height: 1.6; color: #222; }}
  h1 {{ font-size: 1.7rem; margin-bottom: 4px; }}
  .meta {{ color: #777; font-size: 0.88rem; margin-bottom: 24px; }}
  .card {{ background: white; border: 1px solid #e0e0e0;
           border-radius: 6px; padding: 18px 20px; margin-bottom: 14px; }}
  .q {{ font-weight: 600; margin-bottom: 8px; }}
  .a {{ background: #f5f5f5; padding: 10px 12px; border-radius: 4px; margin-top: 8px; }}
  .src {{ color: #1976d2; font-size: 0.82rem; margin-top: 6px; }}
  .footer {{ color: #888; font-size: 0.82rem; margin-top: 40px; padding-top: 16px; border-top: 1px solid #ddd; }}
</style></head><body>
<h1>Clause RAG v0 — 가상 자동차보험 약관 QA</h1>
<p class="meta">BGE-M3 + Chroma + MLX Qwen3.5-122B-A10B  ·  {n_chunks} chunks  ·  top-3 retrieval</p>
{cards}
<div class="footer">
v0 한계: 가상 약관 1개·질문 5개·평가 정량 X.<br/>
<a href="https://github.com/Incheonkirin/clause-rag-v0">github.com/Incheonkirin/clause-rag-v0</a>
</div>
</body></html>"""


def _card(r):
    sources = ", ".join(r["retrieved_ids"])
    return f"""<div class="card">
  <div class="q">{r['code']} · {r['question']}</div>
  <div class="a">{r['answer']}</div>
  <div class="src">retrieved: {sources}  ·  distances: {r['retrieved_distances']}  ·  {r['elapsed_sec']}초</div>
</div>"""


if __name__ == "__main__":
    main()
