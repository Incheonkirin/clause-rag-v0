"""MLX Qwen3.5-122B 답변 + 출처 인용."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_MODEL = "mlx-community/Qwen3.5-122B-A10B-4bit"
_cache: dict[str, Any] = {}


def _load():
    if "model" not in _cache:
        from mlx_lm import load
        name = os.environ.get("MLX_MODEL", DEFAULT_MODEL)
        print(f"  [MLX] {name} 로드 ...")
        _cache["model"], _cache["tokenizer"] = load(name)
    return _cache["model"], _cache["tokenizer"]


SYSTEM = """너는 보험 약관 QA 챗봇이다.
주어진 약관 chunk 3개만 근거로 답해라.

규칙:
1. chunk에 없는 내용 추측 금지. 정보 부족이면 "약관에 명시되지 않음"이라고 답해라
2. 답변 끝에 사용한 chunk id를 "출처: [chunk_001, chunk_005]" 형식으로 인용
3. 한국어 평어체 2-4 문장
4. 투자·법률 자문 금지"""


def answer(query: str, chunks: list[dict], max_tokens: int = 400) -> str:
    from mlx_lm import generate

    chunks_text = "\n\n".join(
        f"[{c['id']}] {c['text']}" for c in chunks
    )
    user = f"[약관 chunk]\n{chunks_text}\n\n[질문] {query}"

    model, tokenizer = _load()
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]
    try:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
        )
    return generate(model, tokenizer, prompt=prompt,
                    max_tokens=max_tokens, verbose=False).strip()
