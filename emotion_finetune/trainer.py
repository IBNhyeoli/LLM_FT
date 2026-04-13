"""
trainer.py
──────────
단일 PEFT 실험의 학습 전 과정을 담당.

- ProgressCallback      : 에폭/스텝 진행도 + 예상 대기 시간 출력
- build_training_args   : TrainingArguments 생성
- run_experiment        : PEFT 적용 → 학습 → 평가 → 저장
"""

import os
import time
import math

import torch
import pandas as pd
from transformers import (
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
from trl import SFTTrainer
from datasets import DatasetDict

from .config import OUTPUT_ROOT, TRAIN_DEFAULTS
from .model import apply_peft
from .evaluate import make_compute_metrics, run_final_evaluation


# ══════════════════════════════════════════════════════════════
# 진행도 콜백
# ══════════════════════════════════════════════════════════════

class ProgressCallback(TrainerCallback):
    """
    학습 중 에폭/스텝 진행도와 예상 남은 시간(ETA)을 출력하는 콜백.

    출력 예시 (스텝):
        [LoRA-r16] Epoch 1/3 | Step   10/120 |  8.3% [████░░░░░░░░░░░░░░░░] | Loss 1.2345
          스텝 평균 00:42 | 에폭 ETA 08:24 | 전체 ETA 25:12

    출력 예시 (에폭 완료):
        [LoRA-r16] Epoch 1/3 완료 | 소요 08:45 | 남은 에폭 ETA 17:30
          eval_f1_macro=0.6891  eval_accuracy=0.7123  eval_kappa=0.6234
    """

    def __init__(self, experiment_name: str):
        self.name             = experiment_name
        self.epoch_start      = None
        self.train_start      = None
        self.step_times: list[float] = []
        self.last_step_time   = None
        self.total_steps      = 0
        self.steps_per_epoch  = 0

    # ── 내부 헬퍼 ──────────────────────────────────────────────

    @staticmethod
    def _fmt(seconds: float) -> str:
        """초 → MM:SS 문자열"""
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _avg_step_time(self) -> float:
        """최근 20스텝 평균 소요 시간(초)"""
        recent = self.step_times[-20:]
        return sum(recent) / len(recent) if recent else 0.0

    @staticmethod
    def _bar(ratio: float, width: int = 20) -> str:
        """비율(0~1) → ASCII 진행 바"""
        filled = int(ratio * width)
        return "[" + "█" * filled + "░" * (width - filled) + "]"

    # ── 콜백 훅 ────────────────────────────────────────────────

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.train_start     = time.time()
        self.total_steps     = state.max_steps
        self.steps_per_epoch = math.ceil(
            self.total_steps / max(args.num_train_epochs, 1)
        )
        print(f"\n{'━'*64}")
        print(f"  [{self.name}] 학습 시작")
        print(
            f"  총 에폭: {int(args.num_train_epochs)} | "
            f"총 스텝: {self.total_steps} | "
            f"에폭당 스텝: {self.steps_per_epoch}"
        )
        print(f"{'━'*64}\n")

    def on_epoch_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.epoch_start    = time.time()
        self.last_step_time = time.time()
        epoch_num = int(state.epoch) + 1
        print(f"\n  ── Epoch {epoch_num}/{int(args.num_train_epochs)} 시작 ──")

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        now      = time.time()
        step_sec = now - (self.last_step_time or now)
        self.last_step_time = now
        self.step_times.append(step_sec)

        # logging_steps 주기에만 출력
        if state.global_step % args.logging_steps != 0:
            return

        avg_step      = self._avg_step_time()
        cur_step      = state.global_step
        epoch_num     = int(state.epoch) + 1
        total_ep      = int(args.num_train_epochs)
        step_in_epoch = cur_step % self.steps_per_epoch or self.steps_per_epoch
        total_ratio   = cur_step / max(self.total_steps, 1)

        # ETA
        eta_total_sec = avg_step * (self.total_steps - cur_step)
        eta_epoch_sec = avg_step * (self.steps_per_epoch - step_in_epoch)

        # Loss
        loss_str = ""
        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                loss_str = f"| Loss {last_log['loss']:.4f} "

        print(
            f"  [{self.name}] "
            f"Epoch {epoch_num}/{total_ep} | "
            f"Step {cur_step:4d}/{self.total_steps} | "
            f"{total_ratio*100:5.1f}% {self._bar(total_ratio)} "
            f"{loss_str}\n"
            f"    스텝 평균 {self._fmt(avg_step)} | "
            f"에폭 ETA {self._fmt(eta_epoch_sec)} | "
            f"전체 ETA {self._fmt(eta_total_sec)}"
        )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        elapsed       = time.time() - (self.epoch_start or time.time())
        epoch_num     = int(state.epoch)
        total_ep      = int(args.num_train_epochs)
        remaining_ep  = total_ep - epoch_num
        eta_remaining = elapsed * remaining_ep

        print(
            f"\n  [{self.name}] Epoch {epoch_num}/{total_ep} 완료 | "
            f"소요 {self._fmt(elapsed)} | "
            f"남은 에폭 ETA {self._fmt(eta_remaining)}"
        )

        # 평가 지표 출력
        if state.log_history:
            last    = state.log_history[-1]
            metrics = {
                k: v for k, v in last.items()
                if any(k.startswith(p) for p in ("eval_", "eval/"))
                and isinstance(v, float)
            }
            if metrics:
                metric_str = "  ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                print(f"    {metric_str}")

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        total_elapsed = time.time() - (self.train_start or time.time())
        print(f"\n{'━'*64}")
        print(f"  [{self.name}] 학습 완료")
        print(
            f"  총 소요: {self._fmt(total_elapsed)} | "
            f"스텝 평균: {self._fmt(self._avg_step_time())} | "
            f"총 스텝: {state.global_step}"
        )
        print(f"{'━'*64}\n")


# ══════════════════════════════════════════════════════════════
# TrainingArguments 빌더
# ══════════════════════════════════════════════════════════════

def build_training_args(
    experiment_name: str,
    num_train_epochs: int = TRAIN_DEFAULTS["num_train_epochs"],
    learning_rate: float  = TRAIN_DEFAULTS["learning_rate"],
    per_device_batch: int = TRAIN_DEFAULTS["per_device_train_batch_size"],
    grad_accum: int       = TRAIN_DEFAULTS["gradient_accumulation_steps"],
    logging_steps: int    = TRAIN_DEFAULTS["logging_steps"],
) -> TrainingArguments:
    """
    실험별 출력 디렉터리를 자동 생성하는 TrainingArguments 반환.
    인자를 직접 넘겨 학습 조건을 유연하게 변경 가능.
    """
    output_dir = os.path.join(OUTPUT_ROOT, experiment_name.replace(" ", "_"))
    return TrainingArguments(
        output_dir                  = output_dir,
        per_device_train_batch_size = per_device_batch,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = learning_rate,
        num_train_epochs            = num_train_epochs,
        logging_steps               = logging_steps,
        save_strategy               = "epoch",
        eval_strategy               = "epoch",
        load_best_model_at_end      = True,
        metric_for_best_model       = "f1_macro",
        greater_is_better           = True,
        bf16                        = True,
        push_to_hub                 = False,
        report_to                   = "none",
        max_grad_norm               = 0.3,
    )


# ══════════════════════════════════════════════════════════════
# 단일 실험 실행
# ══════════════════════════════════════════════════════════════

def run_experiment(
    peft_name: str,
    peft_config,
    datasets: DatasetDict,
    tokenizer,
    base_model,
    num_train_epochs: int = TRAIN_DEFAULTS["num_train_epochs"],
) -> dict:
    """
    하나의 PEFT 실험 전 과정(PEFT 적용 → 학습 → 평가 → 저장)을 실행.

    Parameters
    ----------
    peft_name        : 실험 식별자 (예: 'LoRA-r16', 'DoRA-r16', 'IA3')
    peft_config      : LoraConfig | IA3Config
    datasets         : {"train": Dataset, "eval": Dataset}
    tokenizer        : 베이스 모델의 토크나이저
    base_model       : prepare_model_for_kbit_training 완료된 베이스 모델
    num_train_epochs : 학습 에폭 수

    Returns
    -------
    dict : 실험 평가 결과 {experiment, accuracy, f1_macro, kappa, ...}
    """
    # PEFT 어댑터 적용
    model = apply_peft(base_model, peft_config)

    training_args   = build_training_args(peft_name, num_train_epochs=num_train_epochs)
    compute_metrics = make_compute_metrics(tokenizer)
    output_dir      = training_args.output_dir

    # ── TRL 1.1.0 + transformers 5.x 기준 SFTTrainer 초기화 ──────
    # formatting_func 으로 데이터를 전달하고
    # processing_class 로 토크나이저를 전달하는 방식 사용
    tokenizer.model_max_length = TRAIN_DEFAULTS["max_seq_length"]

    def formatting_func(example):
        """Dataset의 'text' 컬럼을 그대로 반환"""
        if isinstance(example["text"], list):
            return example["text"]
        return [example["text"]]

    trainer = SFTTrainer(
        model            = model,
        train_dataset    = datasets["train"],
        eval_dataset     = datasets["eval"],
        processing_class = tokenizer,
        formatting_func  = formatting_func,
        args             = training_args,
        compute_metrics  = compute_metrics,
        callbacks        = [ProgressCallback(peft_name)],
    )

    trainer.train()

    # 학습 커브 CSV 저장
    pd.DataFrame(trainer.state.log_history).to_csv(
        os.path.join(output_dir, "train_log.csv"), index=False
    )

    # 최종 상세 평가
    result = run_final_evaluation(trainer, datasets["eval"], tokenizer, peft_name)

    # 어댑터 저장
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  어댑터 저장: {output_dir}")

    # GPU 메모리 정리 (연속 실험 대비)
    del model
    torch.cuda.empty_cache()

    return result