"""
infer.py
────────
파인튜닝된 모델로 일기 텍스트를 추론해 감정 분석 JSON을 반환.
파이프라인 다음 단계(이미지 프롬프트 변환)에 직접 연결 가능.

사용 예시
---------
    from peft import PeftModel
    from emotion_finetune.model import load_base_model
    from emotion_finetune.infer import load_finetuned_model, infer_emotion

    base_model, tokenizer = load_base_model()
    model = load_finetuned_model(base_model, "./outputs/LoRA-r16")
    result = infer_emotion("오늘 비가 왔다...", model, tokenizer)
    # → {"primary_emotion": "슬픔", "intensity": 7, ...}
"""

import json

import torch
from peft import PeftModel

from .config import EMOTION_LABELS


# ── 시스템 프롬프트 (추론용 간결 버전) ───────────────────────

def _inference_system_prompt() -> str:
    label_list = ", ".join(EMOTION_LABELS)
    return (
        "당신은 감정 분석 전문가입니다. "
        "주어진 일기 텍스트를 분석하여 JSON 형식으로 감정을 출력하세요. "
        f"primary_emotion은 반드시 다음 중 하나: {label_list}, intensity는 0.0~1.0 사이 실수\n"
        "반드시 JSON만 출력하고 다른 텍스트는 쓰지 마세요."
    )


# ── 어댑터 로드 ───────────────────────────────────────────────

def load_finetuned_model(base_model, adapter_path: str) -> PeftModel:
    """
    저장된 LoRA/DoRA/IA³ 어댑터를 베이스 모델에 올려 반환.

    Parameters
    ----------
    base_model   : load_base_model()이 반환한 베이스 모델
    adapter_path : run_experiment()가 저장한 어댑터 디렉터리 경로
    """
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    print(f"[infer] 어댑터 로드: {adapter_path}")
    return model


# ── 단일 샘플 추론 ────────────────────────────────────────────

def infer_emotion(
    diary_text: str,
    model,
    tokenizer,
    max_new_tokens: int = 256,
) -> dict:
    """
    일기 텍스트를 입력받아 감정 분석 결과를 dict로 반환.

    Parameters
    ----------
    diary_text     : 분석할 일기 텍스트
    model          : 파인튜닝된 모델 (load_finetuned_model 반환값)
    tokenizer      : 대응 토크나이저
    max_new_tokens : 생성 최대 토큰 수

    Returns
    -------
    dict
        성공: {"primary_emotion": ..., "intensity": ..., ...}
        실패: {"raw_output": ..., "parse_error": True}
    """
    system = _inference_system_prompt()
    prompt = (
        f"[|system|] {system}[|endofturn|]\n"
        f"[|user|] {diary_text}[|endofturn|]\n"
        f"[|assistant|]"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens = max_new_tokens,
            do_sample      = False,
            temperature    = 1.0,
            eos_token_id   = tokenizer.eos_token_id,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # 프롬프트 토큰 제거 후 응답만 디코딩
    gen_ids  = output_ids[0][inputs["input_ids"].shape[-1]:]
    raw_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

    return _parse_json_output(raw_text)


# ── 배치 추론 ─────────────────────────────────────────────────

def infer_batch(
    diary_texts: list[str],
    model,
    tokenizer,
    max_new_tokens: int = 256,
) -> list[dict]:
    """
    여러 일기를 순차적으로 추론해 결과 리스트 반환.
    (배치 병렬 처리가 필요하면 DataLoader + generate()로 확장 가능)
    """
    return [
        infer_emotion(text, model, tokenizer, max_new_tokens)
        for text in diary_texts
    ]


# ── JSON 파싱 헬퍼 ────────────────────────────────────────────

def _parse_json_output(raw_text: str) -> dict:
    """
    모델 출력에서 JSON 블록을 추출해 파싱.
    실패 시 raw_output과 parse_error 플래그를 담아 반환.
    """
    try:
        start = raw_text.find("{")
        end   = raw_text.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(raw_text[start:end])
    except (json.JSONDecodeError, ValueError):
        pass

    return {"raw_output": raw_text, "parse_error": True}
