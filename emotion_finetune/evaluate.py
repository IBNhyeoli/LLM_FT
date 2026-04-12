"""
evaluate.py
───────────
감정 분류 평가 관련 함수 모음.

- extract_emotion_from_text  : 모델 출력 텍스트 → 감정 ID
- make_compute_metrics       : SFTTrainer용 compute_metrics 팩토리
- run_final_evaluation       : 학습 후 상세 평가
- print_comparison_table     : PEFT 방식 비교 테이블 출력 및 CSV 저장
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
)
from transformers import EvalPrediction

from .config import EMOTION_LABELS, LABEL2ID, OUTPUT_ROOT


# ── 감정 추출 ─────────────────────────────────────────────────

def extract_emotion_from_text(text: str) -> int:
    """
    모델 출력 텍스트에서 primary_emotion 레이블 ID 추출.

    1순위: JSON 블록 파싱 → "primary_emotion" 키
    2순위: 텍스트 내 레이블 문자열 탐색
    실패 시: -1 반환 (미탐지)
    """
    # JSON 파싱 시도
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start != -1 and end > start:
            data  = json.loads(text[start:end])
            label = data.get("primary_emotion", "")
            if label in LABEL2ID:
                return LABEL2ID[label]
    except (json.JSONDecodeError, KeyError, ValueError):
        pass

    # 폴백: 텍스트 직접 탐색
    for label in EMOTION_LABELS:
        if label in text:
            return LABEL2ID[label]

    return -1   # 미탐지


# ── compute_metrics 팩토리 ────────────────────────────────────

def make_compute_metrics(tokenizer):
    """
    SFTTrainer의 compute_metrics 인자로 전달할 함수를 생성.
    tokenizer가 클로저로 캡처되어 매 에폭마다 자동 호출됨.

    Returns
    -------
    compute_metrics : Callable[[EvalPrediction], dict]
    """
    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        pred_ids  = eval_pred.predictions   # (N, seq_len, vocab) or (N, seq_len)
        label_ids = eval_pred.label_ids     # (N, seq_len)

        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        pred_labels, true_labels = [], []
        for pred_seq, label_seq in zip(pred_ids, label_ids):
            valid_mask = label_seq != -100
            pred_text  = tokenizer.decode(pred_seq[valid_mask],  skip_special_tokens=True)
            label_text = tokenizer.decode(label_seq[valid_mask], skip_special_tokens=True)
            pred_labels.append(extract_emotion_from_text(pred_text))
            true_labels.append(extract_emotion_from_text(label_text))

        p     = np.array(pred_labels)
        t     = np.array(true_labels)
        valid = (p != -1) & (t != -1)

        if valid.sum() == 0:
            return {"accuracy": 0.0, "f1_macro": 0.0, "kappa": 0.0}

        pv, tv = p[valid], t[valid]
        kappa  = cohen_kappa_score(tv, pv) if len(np.unique(tv)) > 1 else 0.0

        return {
            "accuracy": float(accuracy_score(tv, pv)),
            "f1_macro": float(f1_score(
                tv, pv,
                average = "macro",
                labels  = list(range(len(EMOTION_LABELS))),
                zero_division = 0,
            )),
            "kappa": float(kappa),
        }

    return compute_metrics


# ── 최종 상세 평가 ────────────────────────────────────────────

def run_final_evaluation(
    trainer,
    eval_dataset,
    tokenizer,
    experiment_name: str,
) -> dict:
    """
    trainer.predict()를 호출해 클래스별 F1 포함 상세 지표를 계산하고 출력.

    Returns
    -------
    dict
        {experiment, accuracy, f1_macro, kappa, per_class_f1, invalid}
    """
    predictions = trainer.predict(eval_dataset)
    pred_ids    = predictions.predictions
    label_ids   = predictions.label_ids

    if pred_ids.ndim == 3:
        pred_ids = np.argmax(pred_ids, axis=-1)

    pred_labels, true_labels = [], []
    for pred_seq, label_seq in zip(pred_ids, label_ids):
        valid_mask = label_seq != -100
        pred_text  = tokenizer.decode(pred_seq[valid_mask],  skip_special_tokens=True)
        label_text = tokenizer.decode(label_seq[valid_mask], skip_special_tokens=True)
        pred_labels.append(extract_emotion_from_text(pred_text))
        true_labels.append(extract_emotion_from_text(label_text))

    p     = np.array(pred_labels)
    t     = np.array(true_labels)
    valid = (p != -1) & (t != -1)
    pv, tv = p[valid], t[valid]

    acc       = accuracy_score(tv, pv)
    f1        = f1_score(tv, pv, average="macro",
                          labels=list(range(len(EMOTION_LABELS))), zero_division=0)
    kappa     = cohen_kappa_score(tv, pv) if len(np.unique(tv)) > 1 else 0.0
    per_class = f1_score(tv, pv, average=None,
                          labels=list(range(len(EMOTION_LABELS))), zero_division=0)

    # 콘솔 출력
    sep = "─" * 46
    print(f"\n{sep}")
    print(f"  {experiment_name} — 최종 평가 결과")
    print(sep)
    print(f"  Accuracy       : {acc:.4f}")
    print(f"  Macro F1-Score : {f1:.4f}")
    print(f"  Cohen's Kappa  : {kappa:.4f}")
    print(f"  미탐지 샘플    : {(~valid).sum()}개 제외됨")
    print("\n  클래스별 F1-Score:")
    for label, score in zip(EMOTION_LABELS, per_class):
        bar = "█" * int(score * 24)
        print(f"    {label:4s}: {score:.4f}  {bar}")
    print(f"\n{classification_report(tv, pv, target_names=EMOTION_LABELS, zero_division=0)}")

    return {
        "experiment":   experiment_name,
        "accuracy":     float(acc),
        "f1_macro":     float(f1),
        "kappa":        float(kappa),
        "per_class_f1": dict(zip(EMOTION_LABELS, per_class.tolist())),
        "invalid":      int((~valid).sum()),
    }


# ── 비교 테이블 ───────────────────────────────────────────────

def print_comparison_table(results: list[dict]) -> None:
    """
    PEFT 실험 결과 비교 테이블을 콘솔에 출력하고
    OUTPUT_ROOT/peft_comparison.csv 로 저장.
    """
    sep = "═" * 64
    print(f"\n{sep}")
    print("  PEFT 방식 비교 요약  (f1_macro 기준 내림차순)")
    print(sep)
    print(f"  {'실험':20s}  {'Accuracy':>10}  {'Macro F1':>10}  {'Kappa':>8}")
    print("  " + "─" * 56)

    for r in sorted(results, key=lambda x: x["f1_macro"], reverse=True):
        print(
            f"  {r['experiment']:20s}  "
            f"{r['accuracy']:>10.4f}  "
            f"{r['f1_macro']:>10.4f}  "
            f"{r['kappa']:>8.4f}"
        )
    print(sep)

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    save_path = os.path.join(OUTPUT_ROOT, "peft_comparison.csv")
    pd.DataFrame(results).to_csv(save_path, index=False)
    print(f"\n  결과 저장: {save_path}")
