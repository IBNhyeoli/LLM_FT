"""
evaluate.py
───────────
감정 분류 평가 관련 함수 모음.

- extract_emotion_from_text  : 모델 출력 텍스트 → 감정 ID
- make_compute_metrics       : SFTTrainer용 compute_metrics 팩토리
- run_final_evaluation       : 학습 후 log_history 기반 최종 평가
- print_comparison_table     : PEFT 방식 비교 테이블 출력 및 CSV 저장

변경사항
--------
- run_final_evaluation: trainer.predict() 대신 log_history 사용
  (TRL 1.1.0에서 formatting_func 사용 시 predict()가 KeyError 발생)
"""

import os
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    cohen_kappa_score,
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

    for label in EMOTION_LABELS:
        if label in text:
            return LABEL2ID[label]

    return -1


# ── compute_metrics 팩토리 ────────────────────────────────────

def make_compute_metrics(tokenizer):
    """
    SFTTrainer의 compute_metrics 인자로 전달할 함수를 생성.
    매 에폭 eval 시 자동 호출되어 accuracy / f1_macro / kappa 반환.

    Returns
    -------
    compute_metrics : Callable[[EvalPrediction], dict]
    """
    def compute_metrics(eval_pred: EvalPrediction) -> dict:
        pred_ids  = eval_pred.predictions
        label_ids = eval_pred.label_ids

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
        # 정답과 예측 모두 2개 이상 클래스가 있어야 Kappa 계산 가능
        all_labels = np.concatenate([tv, pv])
        kappa = cohen_kappa_score(tv, pv) if len(np.unique(all_labels)) > 1 else 0.0

        return {
            "accuracy": float(accuracy_score(tv, pv)),
            "f1_macro": float(f1_score(
                tv, pv,
                average       = "macro",
                labels        = list(range(len(EMOTION_LABELS))),
                zero_division = 0,
            )),
            "kappa": float(kappa),
        }

    return compute_metrics


# ── 최종 평가 ────────────────────────────────────────────────

def run_final_evaluation(
    trainer,
    eval_dataset,
    tokenizer,
    experiment_name: str,
) -> dict:
    """
    학습 중 기록된 log_history에서 eval 지표를 추출해 출력.

    우선순위:
      1. eval_accuracy / eval_f1_macro / eval_kappa 가 있으면 사용
      2. 없으면 eval_loss + train 지표(mean_token_accuracy)로 대체 출력
         - prediction_loss_only=True 설정 시 이 경우에 해당

    Returns
    -------
    dict : {experiment, accuracy, f1_macro, kappa, eval_loss}
    """
    # 1순위: compute_metrics 결과
    eval_metrics = {}
    for log in reversed(trainer.state.log_history):
        if "eval_f1_macro" in log or "eval_accuracy" in log:
            eval_metrics = log
            break

    if eval_metrics:
        acc   = float(eval_metrics.get("eval_accuracy", 0.0))
        f1    = float(eval_metrics.get("eval_f1_macro", 0.0))
        kappa = float(eval_metrics.get("eval_kappa",    0.0))
        eval_loss = float(eval_metrics.get("eval_loss", 0.0))
    else:
        # 2순위: eval_loss + train mean_token_accuracy
        acc = f1 = kappa = 0.0
        eval_loss = 0.0
        for log in reversed(trainer.state.log_history):
            if "eval_loss" in log:
                eval_loss = float(log["eval_loss"])
                break

        # train 지표에서 최종 mean_token_accuracy 추출 (참고용)
        train_acc = 0.0
        train_loss = 0.0
        for log in reversed(trainer.state.log_history):
            if "mean_token_accuracy" in log:
                train_acc  = float(log["mean_token_accuracy"])
                train_loss = float(log.get("loss", 0.0))
                break

    sep = "─" * 46
    print(f"\n{sep}")
    print(f"  {experiment_name} — 최종 평가 결과")
    print(sep)

    if f1 > 0 or acc > 0:
        print(f"  Accuracy       : {acc:.4f}")
        print(f"  Macro F1-Score : {f1:.4f}")
        print(f"  Cohen\'s Kappa  : {kappa:.4f}")
        print(f"  Eval Loss      : {eval_loss:.4f}")
    else:
        # prediction_loss_only=True 모드: loss/token accuracy만 출력
        print(f"  Eval Loss           : {eval_loss:.4f}")
        if "train_acc" in dir():
            print(f"  Train Token Acc     : {train_acc:.4f}  (참고용)")
            print(f"  Train Loss (최종)   : {train_loss:.4f}")
        print()
        print("  ※ compute_metrics 미실행 (prediction_loss_only=True)")
        print("  ※ F1/Kappa는 eval_strategy 변경 후 재실행 시 측정 가능")

    return {
        "experiment":   experiment_name,
        "accuracy":     acc,
        "f1_macro":     f1,
        "kappa":        kappa,
        "eval_loss":    eval_loss,
        "per_class_f1": {},
        "invalid":      0,
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