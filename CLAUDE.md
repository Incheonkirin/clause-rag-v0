# CLAUDE.md — Clause RAG v0

## Mission

보험 표준약관 RAG QA 챗봇 v0. 가상 자동차보험 표준약관 1개 + BGE-M3 임베딩 + Chroma + MLX Qwen3.5-122B 답변 + 출처 인용.

## Voice

- 평어체 (~다)
- "박다" 금지 — "넣다·추가하다·기록한다"
- "책임진다" 금지
- 회사 자료 절대 X

## 본인 카드 직접 매칭

포티투마루 4년 RAG/MRC + B2B 약관 RAG QA (할루시네이션 -90%) → KP손보 *상담 RAG·Document AI 즉시지급* 매칭.

## 구조

```
data/clause_v0.md           # 가상 자동차보험 표준약관
src/rag/embed.py            # BGE-M3 + chunk + Chroma 인덱싱
src/rag/retrieve.py         # query → top-3 chunk
src/rag/answer.py           # MLX Qwen 답변 + 출처
data/questions_v0.yaml      # 5 QA
scripts/run_rag.py          # batch + JSON + HTML
docs/index.html             # Pages
```

## v0 한계

- 약관 1개 (가상)
- 질문 5개
- 단일 LLM (Qwen3.5-122B)
- chunk 단순 (300자, overlap 50)
- 평가 정량값 X (v1에서 RAGAS faithfulness 추가)

## v1 계획

- 실제 공개 표준약관 (손보협회) 다중 약관
- RAGAS faithfulness·answer_relevancy
- BGE-M3 vs KURE-v1 vs multilingual-e5 비교
- 100 QA 확장
