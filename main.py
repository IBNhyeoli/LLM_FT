"""
main.py
───────
CLI 진입점. 전체 PEFT 비교 실험 파이프라인을 실행한다.

실행 방법
---------
# 로컬 JSONL (기본)
    python main.py --source local_jsonl --data final_diary_data.jsonl

# 영문 공개 데이터셋 (빠른 기능 검증)
    python main.py --source dair_emotion --experiments LoRA-r16 --epochs 1

# 한국어 공개 데이터셋 (ke-t5 기반 7감정)
    python main.py --source ke_t5_kor --experiments LoRA-r16 DoRA-r16

# 한국어 7감정 대화 데이터셋
    python main.py --source kor_7class

# 특정 실험만 선택
    python main.py --source ke_t5_kor --experiments LoRA-r16 DoRA-r16

# 추론만 실행 (저장된 어댑터 로드)
    python main.py --infer-only --adapter ./outputs/LoRA-r16
"""

import argparse
import os

from emotion_finetune import (
    load_and_prepare,
    load_base_model,
    get_default_experiments,
    get_lora_config,
    get_ia3_config,
    run_experiment,
    print_comparison_table,
    load_finetuned_model,
    infer_emotion,
    OUTPUT_ROOT,
    DATA_FILE,
)
from emotion_finetune.data import SUPPORTED_SOURCES


# ── 실험 이름 → PEFT 설정 매핑 ───────────────────────────────

EXPERIMENT_REGISTRY: dict[str, callable] = {
    "LoRA-r8":  lambda: get_lora_config(r=8,  use_dora=False),
    "LoRA-r16": lambda: get_lora_config(r=16, use_dora=False),
    "LoRA-r32": lambda: get_lora_config(r=32, use_dora=False),
    "DoRA-r16": lambda: get_lora_config(r=16, use_dora=True),
    "DoRA-r32": lambda: get_lora_config(r=32, use_dora=True),
    "IA3":      lambda: get_ia3_config(),
}


# ── argparse ─────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EXAONE 3.5 감정 분석 파인튜닝 및 PEFT 방식 비교",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--source",
        type=str,
        default="local_jsonl",
        choices=SUPPORTED_SOURCES,
        help=(
            "데이터 소스 선택 (기본: local_jsonl)\n"
            "  local_jsonl  : 로컬 JSONL 파일 (--data 옵션 필요)\n"
            "  dair_emotion : dair-ai/emotion, 영문 6감정, 436K건 — 빠른 검증용\n"
            "  ke_t5_kor    : KETI-AIR 한국어 7감정 SNS 데이터, 약 30K건\n"
            "  kor_7class   : 한국어 감성 대화 7감정, 약 40K건"
        ),
    )
    parser.add_argument(
        "--data", type=str, default=DATA_FILE,
        help="source=local_jsonl 시 JSONL 파일 경로 (기본: final_diary_data.jsonl)",
    )
    parser.add_argument(
        "--experiments", nargs="+",
        choices=list(EXPERIMENT_REGISTRY.keys()),
        default=None,
        help="실행할 실험 목록 (기본: 전체 4종)\n예: --experiments LoRA-r16 DoRA-r16",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="학습 에폭 수 (기본: 3)",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.1,
        help="평가 데이터 비율 (기본: 0.1)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        metavar="N",
        help=(
            "학습+평가에 사용할 최대 샘플 수 (기본: 전체)\n"
            "빠른 비교 실험 시 권장 값:\n"
            "  smoke test  : --max-samples 200  (수 분)\n"
            "  quick run   : --max-samples 500  (10~20분)\n"
            "  mini run    : --max-samples 2000 (1~2시간)"
        ),
    )
    parser.add_argument(
        "--infer-only", action="store_true",
        help="학습 없이 추론 테스트만 실행",
    )
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="--infer-only 시 로드할 어댑터 경로",
    )
    parser.add_argument(
        "--infer-text", type=str,
        default="오늘은 비가 왔다. 오랜만에 혼자 카페에 앉아 창밖을 바라봤다.",
        help="추론 테스트에 사용할 텍스트",
    )
    return parser.parse_args()


# ── 학습 파이프라인 ───────────────────────────────────────────

def run_training_pipeline(args: argparse.Namespace) -> None:
    # 1. 데이터
    datasets = load_and_prepare(
        source      = args.source,
        data_file   = args.data,
        test_size   = args.test_size,
        max_samples = args.max_samples,
    )

    # 2. 베이스 모델 (1회 로드 후 모든 실험에서 공유)
    base_model, tokenizer = load_base_model()

    # 3. 실험 목록 결정
    experiments = (
        [(name, EXPERIMENT_REGISTRY[name]()) for name in args.experiments]
        if args.experiments
        else get_default_experiments()
    )

    # 4. 순차 실험
    all_results = []
    for name, peft_cfg in experiments:
        result = run_experiment(
            peft_name        = name,
            peft_config      = peft_cfg,
            datasets         = datasets,
            tokenizer        = tokenizer,
            base_model       = base_model,
            num_train_epochs = args.epochs,
        )
        all_results.append(result)

    # 5. 비교 테이블 + CSV 저장
    print_comparison_table(all_results)

    # 6. 최고 성능 모델 추론 예시
    best        = max(all_results, key=lambda x: x["f1_macro"])
    best_adapter = os.path.join(OUTPUT_ROOT, best["experiment"].replace(" ", "_"))
    print(f"\n  최고 성능: {best['experiment']} (F1={best['f1_macro']:.4f})")
    _run_inference_demo(base_model, tokenizer, best_adapter, args.infer_text)


# ── 추론 파이프라인 ───────────────────────────────────────────

def run_inference_pipeline(args: argparse.Namespace) -> None:
    if not args.adapter:
        raise ValueError("--infer-only 모드에서는 --adapter 경로를 지정해야 합니다.")
    base_model, tokenizer = load_base_model()
    _run_inference_demo(base_model, tokenizer, args.adapter, args.infer_text)


def _run_inference_demo(base_model, tokenizer, adapter_path, diary_text):
    import json as _json
    model  = load_finetuned_model(base_model, adapter_path)
    result = infer_emotion(diary_text, model, tokenizer)
    print("\n" + "─" * 46)
    print("  추론 결과")
    print("─" * 46)
    print(f"  입력: {diary_text[:60]}{'...' if len(diary_text) > 60 else ''}")
    print(f"  출력:\n{_json.dumps(result, ensure_ascii=False, indent=4)}")
    print("─" * 46)


# ── 진입점 ────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if args.infer_only:
        run_inference_pipeline(args)
    else:
        run_training_pipeline(args)


if __name__ == "__main__":
    main()
