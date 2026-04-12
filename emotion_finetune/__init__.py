"""
emotion_finetune
────────────────
EXAONE 3.5 기반 일기 감정 분석 파인튜닝 패키지.

공개 API
--------
설정
    config.EMOTION_LABELS, LABEL2ID, MODEL_ID, OUTPUT_ROOT, ...

데이터
    data.load_and_prepare(data_file, test_size) -> DatasetDict

모델
    model.load_base_model(model_id) -> (model, tokenizer)
    model.get_lora_config(r, use_dora) -> LoraConfig
    model.get_ia3_config() -> IA3Config
    model.apply_peft(base_model, peft_config) -> peft_model
    model.get_default_experiments() -> list[(name, config)]

학습
    trainer.run_experiment(name, peft_config, datasets, tokenizer, base_model) -> dict
    trainer.build_training_args(name, ...) -> TrainingArguments

평가
    evaluate.make_compute_metrics(tokenizer) -> Callable
    evaluate.run_final_evaluation(trainer, eval_ds, tokenizer, name) -> dict
    evaluate.print_comparison_table(results) -> None
    evaluate.extract_emotion_from_text(text) -> int

추론
    infer.load_finetuned_model(base_model, adapter_path) -> PeftModel
    infer.infer_emotion(diary_text, model, tokenizer) -> dict
    infer.infer_batch(diary_texts, model, tokenizer) -> list[dict]
"""

from . import config, data, model, trainer, evaluate, infer

from .config import (
    MODEL_ID,
    DATA_FILE,
    OUTPUT_ROOT,
    SEED,
    EMOTION_LABELS,
    LABEL2ID,
    ID2LABEL,
)
from .data      import load_and_prepare
from .model     import (
    load_base_model,
    get_lora_config,
    get_ia3_config,
    apply_peft,
    get_default_experiments,
)
from .trainer   import run_experiment, build_training_args
from .evaluate  import (
    make_compute_metrics,
    run_final_evaluation,
    print_comparison_table,
    extract_emotion_from_text,
)
from .infer     import load_finetuned_model, infer_emotion, infer_batch

__all__ = [
    "config", "data", "model", "trainer", "evaluate", "infer",
    "MODEL_ID", "DATA_FILE", "OUTPUT_ROOT", "SEED",
    "EMOTION_LABELS", "LABEL2ID", "ID2LABEL",
    "load_and_prepare",
    "load_base_model", "get_lora_config", "get_ia3_config",
    "apply_peft", "get_default_experiments",
    "run_experiment", "build_training_args",
    "make_compute_metrics", "run_final_evaluation",
    "print_comparison_table", "extract_emotion_from_text",
    "load_finetuned_model", "infer_emotion", "infer_batch",
]
