# Clause RAG v0 — 보험 표준약관 QA 챗봇

가상 자동차보험 표준약관 1개를 BGE-M3로 임베딩하고, MLX Qwen3.5-122B-A10B가 질문에 출처 인용과 함께 답한다.

## 무엇을

- 가상 표준약관 (`data/clause_v0.md`)
- BGE-M3 임베딩 + Chroma vector store
- top-3 chunk 검색 → MLX Qwen 답변 + 출처 chunk id 인용
- 5 질문 batch 결과를 정적 HTML로 publish

## 한계

- 약관 1개·질문 5개·단일 LLM
- 평가 정량 지표(faithfulness·answer relevancy) X
- chunk 단순 (300자, overlap 50, 정교한 split X)

## 가치

본인 4년 RAG/MRC 경험을 *공개 데이터*로 재현하는 최소 단위. RAGAS·다중 약관·정량 측정은 v1 작업.

## 회사 자료 분리

가상 약관만 사용. 실제 보험사 약관·QA 데이터 일절 X.

## 라이선스

MIT
