"""
model.py
────────
베이스 모델 로드(4-bit QLoRA) 및 PEFT 설정 팩토리.

지원 PEFT 방식
--------------
- LoRA  : get_lora_config(r, use_dora=False)
- DoRA  : get_lora_config(r, use_dora=True)
- IA³   : get_ia3_config()
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig,
    IA3Config,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from .config import (
    MODEL_ID,
    EXAONE_ATTN_MODULES,
    EXAONE_FF_MODULES,
)


# ── 베이스 모델 로드 ──────────────────────────────────────────

def load_base_model(
    model_id: str = MODEL_ID,
) -> tuple:
    """
    4-bit NF4 양자화 + kbit 학습 준비 상태의 모델과 토크나이저 반환.

    Returns
    -------
    model      : 양자화된 CausalLM (gradient_checkpointing 활성화)
    tokenizer  : 패딩 토큰 설정 완료
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token    = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit             = True,
        bnb_4bit_use_double_quant= True,
        bnb_4bit_quant_type      = "nf4",
        bnb_4bit_compute_dtype   = torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config = bnb_config,
        device_map          = "auto",
        trust_remote_code   = True,
    )

    # kbit 학습 준비 (순서 고정: enable → prepare)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    print(f"[model] 로드 완료: {model_id}")
    return model, tokenizer


# ── PEFT 설정 팩토리 ──────────────────────────────────────────

def get_lora_config(r: int = 16, use_dora: bool = False) -> LoraConfig:
    """
    LoRA 또는 DoRA 설정 반환.

    Parameters
    ----------
    r        : LoRA rank — 8 / 16 / 32 실험 권장
    use_dora : True → Weight-Decomposed LoRA(DoRA) 활성화
    """
    return LoraConfig(
        r              = r,
        lora_alpha     = r * 2,       # 통상 alpha = 2 × r
        target_modules = EXAONE_ATTN_MODULES,
        lora_dropout   = 0.05,
        bias           = "none",
        task_type      = "CAUSAL_LM",
        use_dora       = use_dora,
    )


def get_ia3_config() -> IA3Config:
    """
    IA³ 설정 반환.
    학습 파라미터가 극히 적어 경량 비교 실험에 적합.
    """
    return IA3Config(
        target_modules      = EXAONE_ATTN_MODULES + EXAONE_FF_MODULES,
        feedforward_modules = EXAONE_FF_MODULES,
        task_type           = "CAUSAL_LM",
    )


def apply_peft(base_model, peft_config):
    """
    베이스 모델에 PEFT 어댑터를 적용하고 학습 가능 파라미터 수를 출력.

    Returns
    -------
    peft_model : PEFT 어댑터가 적용된 모델
    """
    peft_model = get_peft_model(base_model, peft_config)
    peft_model.print_trainable_parameters()
    return peft_model


# ── 실험 정의 목록 ────────────────────────────────────────────

def get_default_experiments() -> list[tuple[str, object]]:
    """
    기본 PEFT 실험 목록 반환.
    main.py에서 그대로 사용하거나 원하는 항목만 선택 가능.

    Returns
    -------
    list of (experiment_name, peft_config)
    """
    return [
        ("LoRA-r8",  get_lora_config(r=8,  use_dora=False)),
        ("LoRA-r16", get_lora_config(r=16, use_dora=False)),
        ("DoRA-r16", get_lora_config(r=16, use_dora=True)),
        ("IA3",      get_ia3_config()),
    ]
