"""
trainer.py
──────────
단일 PEFT 실험의 학습 전 과정을 담당.

- ProgressCallback      : tqdm 기반 진행도 바 + 소요/ETA 시간 출력
- build_training_args   : TrainingArguments 생성
- run_experiment        : PEFT 적용 → 학습 → 평가 → 저장

TRL 1.1.0 + transformers 5.x 호환
----------------------------------
- SFTTrainer: processing_class + formatting_func 방식
- eval_strategy="no" : eval 중 로짓 누적으로 인한 GPU OOM 방지
- gradient_checkpointing=False : EXAONE 미지원
- remove_unused_columns=False  : formatting_func 사용 시 필수
"""

import os
import time
import math

import torch
import pandas as pd
from tqdm.auto import tqdm
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
from .evaluate import run_final_evaluation


# ══════════════════════════════════════════════════════════════
# tqdm 기반 진행도 콜백
# ══════════════════════════════════════════════════════════════

class ProgressCallback(TrainerCallback):
    """
    tqdm 진행 바 + 소요 시간 / ETA를 실시간으로 표시하는 콜백.

    표시 항목
    ---------
    - 전체 학습 진행 바  : 총 스텝 기준
    - 에폭 진행 바       : 에폭 내 스텝 기준
    - postfix 정보       : Loss / lr / 스텝평균 / 에폭ETA / 전체ETA
    - 에폭 완료 요약     : 소요 시간 출력
    - 학습 완료 요약     : 총 소요 시간 출력
    """

    def __init__(self, experiment_name: str):
        self.name              = experiment_name
        self.train_bar         = None   # 전체 진행 바
        self.epoch_bar         = None   # 에폭 내 진행 바
        self.train_start       = None
        self.epoch_start       = None
        self.last_step_time    = None
        self.step_times: list[float] = []
        self.total_steps       = 0
        self.steps_per_epoch   = 0

    # ── 내부 헬퍼 ──────────────────────────────────────────────

    @staticmethod
    def _fmt(seconds: float) -> str:
        """초 → MM:SS 문자열"""
        seconds = max(0, int(seconds))
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _avg_step_time(self) -> float:
        recent = self.step_times[-20:]
        return sum(recent) / len(recent) if recent else 0.0

    # ── 콜백 훅 ────────────────────────────────────────────────

    def on_train_begin(self, args, state, control, **kwargs):
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
        print(f"{'━'*64}")

        # 전체 학습 진행 바 생성
        self.train_bar = tqdm(
            total       = self.total_steps,
            desc        = f"[{self.name}] 전체",
            unit        = "step",
            dynamic_ncols= True,
            colour      = "green",
            position    = 0,
            leave       = True,
        )

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start    = time.time()
        self.last_step_time = time.time()
        epoch_num = int(state.epoch) + 1
        total_ep  = int(args.num_train_epochs)

        # 에폭 진행 바 생성
        if self.epoch_bar is not None:
            self.epoch_bar.close()
        self.epoch_bar = tqdm(
            total        = self.steps_per_epoch,
            desc         = f"  Epoch {epoch_num}/{total_ep}",
            unit         = "step",
            dynamic_ncols = True,
            colour       = "blue",
            position     = 1,
            leave        = False,
        )

    def on_step_end(self, args, state, control, **kwargs):
        now      = time.time()
        step_sec = now - (self.last_step_time or now)
        self.last_step_time = now
        self.step_times.append(step_sec)

        # 진행 바 업데이트 (매 스텝)
        if self.train_bar is not None:
            self.train_bar.update(1)
        if self.epoch_bar is not None:
            self.epoch_bar.update(1)

        # logging_steps 주기마다 postfix 갱신
        if state.global_step % args.logging_steps != 0:
            return

        avg_step      = self._avg_step_time()
        cur_step      = state.global_step
        step_in_epoch = cur_step % self.steps_per_epoch or self.steps_per_epoch
        eta_total_sec = avg_step * (self.total_steps - cur_step)
        eta_epoch_sec = avg_step * (self.steps_per_epoch - step_in_epoch)

        # Loss / lr 추출
        postfix = {}
        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                postfix["loss"] = f"{last_log['loss']:.4f}"
            if "learning_rate" in last_log:
                postfix["lr"] = f"{last_log['learning_rate']:.2e}"

        postfix["step/s"]    = f"{self._fmt(avg_step)}"
        postfix["ep_ETA"]    = f"{self._fmt(eta_epoch_sec)}"
        postfix["total_ETA"] = f"{self._fmt(eta_total_sec)}"

        if self.train_bar is not None:
            self.train_bar.set_postfix(postfix)
        if self.epoch_bar is not None:
            self.epoch_bar.set_postfix(
                {"loss": postfix.get("loss", "-"), "ETA": self._fmt(eta_epoch_sec)}
            )

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed   = time.time() - (self.epoch_start or time.time())
        epoch_num = int(state.epoch)
        total_ep  = int(args.num_train_epochs)

        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None

        print(
            f"\n  [{self.name}] Epoch {epoch_num}/{total_ep} 완료 | "
            f"소요 {self._fmt(elapsed)}"
        )

    def on_train_end(self, args, state, control, **kwargs):
        total_elapsed = time.time() - (self.train_start or time.time())

        if self.train_bar is not None:
            self.train_bar.close()
            self.train_bar = None
        if self.epoch_bar is not None:
            self.epoch_bar.close()
            self.epoch_bar = None

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
    """실험별 출력 디렉터리를 자동 생성하는 TrainingArguments 반환."""
    output_dir = os.path.join(OUTPUT_ROOT, experiment_name.replace(" ", "_"))
    return TrainingArguments(
        output_dir                  = output_dir,
        per_device_train_batch_size = per_device_batch,
        gradient_accumulation_steps = grad_accum,
        learning_rate               = learning_rate,
        num_train_epochs            = num_train_epochs,
        logging_steps               = logging_steps,
        save_strategy               = "no",    # OOM 방지: eval 미실행
        eval_strategy               = "no",    # eval 중 로짓 누적 → GPU OOM 방지
        load_best_model_at_end      = False,   # eval_strategy=no 시 비활성화
        bf16                        = True,
        push_to_hub                 = False,
        report_to                   = "none",
        max_grad_norm               = 0.3,
        gradient_checkpointing      = False,   # EXAONE 미지원
        remove_unused_columns       = False,   # formatting_func 사용 시 필수
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
    model      = apply_peft(base_model, peft_config)
    output_dir = os.path.join(OUTPUT_ROOT, peft_name.replace(" ", "_"))

    training_args = build_training_args(
        peft_name,
        num_train_epochs=num_train_epochs,
    )

    # ── TRL 1.1.0 + transformers 5.x SFTTrainer 초기화 ──────────
    tokenizer.model_max_length = TRAIN_DEFAULTS["max_seq_length"]

    def formatting_func(example):
        """text 컬럼을 문자열로 반환 (TRL 1.1.0 요구사항)"""
        if isinstance(example["text"], list):
            return example["text"][0]
        return example["text"]

    trainer = SFTTrainer(
        model            = model,
        train_dataset    = datasets["train"],
        eval_dataset     = datasets["eval"],
        processing_class = tokenizer,
        formatting_func  = formatting_func,
        args             = training_args,
        callbacks        = [ProgressCallback(peft_name)],
    )

    trainer.train()

    # 학습 커브 CSV 저장
    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame(trainer.state.log_history).to_csv(
        os.path.join(output_dir, "train_log.csv"), index=False
    )

    # 최종 평가 (log_history 기반)
    result = run_final_evaluation(trainer, datasets["eval"], tokenizer, peft_name)

    # 어댑터 저장
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"  어댑터 저장: {output_dir}")

    # GPU 메모리 정리
    del model
    torch.cuda.empty_cache()

    return result