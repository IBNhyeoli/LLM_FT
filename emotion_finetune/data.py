"""
data.py
───────
데이터 로드, 포맷, 분리 모듈.

지원하는 데이터 소스
--------------------
1. local_jsonl   : 로컬 JSONL 파일 (기존 방식)
2. dair_emotion  : dair-ai/emotion  — 영문, 6감정, 436K건 (Colab 빠른 테스트용)
3. ke_t5_kor     : KETI-AIR/ke-t5-ko-sns-emotion — 한국어, 7감정, AI Hub 기반
4. kor_hate      : hun3359/klue-bert-base-sentiment 학습용 원본
                   → snunlp/KSNLI 대신 실제 사용 가능한 데이터셋으로 대체:
                   감성 대화 데이터 (Korean 7-class emotion)

사용 방법 (main.py 또는 Colab)
-------------------------------
    from emotion_finetune.data import load_and_prepare

    # 로컬 JSONL
    ds = load_and_prepare(source="local_jsonl", data_file="final_diary_data.jsonl")

    # dair-ai/emotion (영문, 빠른 기능 검증용)
    ds = load_and_prepare(source="dair_emotion")

    # 한국어 감성 대화 (ke-t5 기반, 7감정)
    ds = load_and_prepare(source="ke_t5_kor")
"""

import json
from datasets import load_dataset, DatasetDict, Dataset

from .config import DATA_FILE, SEED, EMOTION_LABELS, TRAIN_DEFAULTS


# ══════════════════════════════════════════════════════════════
# 데이터셋별 레이블 매핑 상수
# ══════════════════════════════════════════════════════════════

# dair-ai/emotion 레이블 → 프로젝트 감정 레이블 매핑
# 원본: sadness(0) joy(1) love(2) anger(3) fear(4) surprise(5)
_DAIR_ID2KO = {
    0: "슬픔",
    1: "기쁨",
    2: "기쁨",    # love → 기쁨으로 근사
    3: "분노",
    4: "불안",    # fear → 불안으로 근사
    5: "놀람",
}

# ke-t5 한국어 감성 대화 레이블 (원본 레이블명 → 프로젝트 레이블)
_KET5_LABEL2KO = {
    "기쁨":  "기쁨",
    "당황":  "놀람",
    "분노":  "분노",
    "불안":  "불안",
    "상처":  "슬픔",
    "슬픔":  "슬픔",
    "중립":  "중립",
    # 숫자 인덱스 버전 대비
    "0": "기쁨",
    "1": "당황",
    "2": "분노",
    "3": "불안",
    "4": "상처",
    "5": "슬픔",
    "6": "중립",
}


# ══════════════════════════════════════════════════════════════
# 시스템 프롬프트
# ══════════════════════════════════════════════════════════════

def _build_system_prompt() -> str:
    label_list = ", ".join(EMOTION_LABELS)
    return (
        "당신은 감정 분석 전문가입니다. "
        "주어진 텍스트를 분석하여 다음 JSON 형식으로 감정을 출력하세요:\n"
        '{"primary_emotion": "<감정>", "intensity": <0.0-1.0>, '
        '"tone": "<분위기>", "scene_keywords": ["<키워드>", ...], '
        '"secondary_emotions": ["<감정>", ...], "valence": "<positive|negative|neutral>"}\n'
        f"primary_emotion은 반드시 다음 중 하나여야 합니다: {label_list}\n"
        "반드시 JSON만 출력하고 다른 텍스트는 쓰지 마세요."
    )


# ══════════════════════════════════════════════════════════════
# 프롬프트 포맷터
# ══════════════════════════════════════════════════════════════

def _make_prompt(input_text: str, output_json: str) -> dict:
    """EXAONE 채팅 템플릿 형식으로 변환"""
    system = _build_system_prompt()
    return {
        "text": (
            f"[|system|] {system}[|endofturn|]\n"
            f"[|user|] {input_text}[|endofturn|]\n"
            f"[|assistant|] {output_json}[|endofturn|]"
        )
    }


def _emotion_to_json(primary_emotion: str) -> str:
    """
    감정 레이블 → JSON 문자열 변환.
    intensity/tone 등은 레이블만 있는 공개 데이터셋에서 기본값으로 채움.
    """
    valence_map = {
        "기쁨": "positive",
        "슬픔": "negative",
        "분노": "negative",
        "불안": "negative",
        "혐오": "negative",
        "놀람": "neutral",
        "중립": "neutral",
    }
    return json.dumps({
        "primary_emotion":    primary_emotion,
        "intensity":          0.5,          # 기본값 (레이블 데이터셋은 강도 미제공)
        "tone":               primary_emotion + "의",
        "scene_keywords":     [],
        "secondary_emotions": [],
        "valence":            valence_map.get(primary_emotion, "neutral"),
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# 데이터소스별 어댑터
# ══════════════════════════════════════════════════════════════

def _load_local_jsonl(data_file: str, test_size: float) -> DatasetDict:
    """
    로컬 JSONL 로드. 포맷: {"input": "...", "output": "..."}
    output 필드가 이미 JSON 문자열인 경우를 가정.
    """
    raw   = load_dataset("json", data_files=data_file, split="train")
    split = raw.train_test_split(test_size=test_size, seed=SEED)
    cols  = split["train"].column_names

    def fmt(ex):
        return _make_prompt(ex["input"], ex["output"])

    return DatasetDict({
        "train": split["train"].map(fmt, remove_columns=cols),
        "eval":  split["test"].map(fmt,  remove_columns=cols),
    })


def _load_dair_emotion(test_size: float) -> DatasetDict:
    """
    dair-ai/emotion 로드 (영문).
    - 436K건 / 6감정 / Apache-2.0
    - HuggingFace Hub에서 바로 다운로드 가능
    - 빠른 기능 검증(smoke test)에 적합

    레이블 매핑: sadness→슬픔, joy→기쁨, love→기쁨, anger→분노,
                 fear→불안, surprise→놀람
    """
    raw = load_dataset("dair-ai/emotion", "split", trust_remote_code=True)

    # train/test 분리 (원본 split 활용)
    train_raw = raw.get("train")
    eval_raw  = raw.get("validation") or raw.get("test")

    def fmt(ex):
        emotion_ko = _DAIR_ID2KO.get(ex["label"], "중립")
        return _make_prompt(ex["text"], _emotion_to_json(emotion_ko))

    cols = train_raw.column_names
    train_ds = train_raw.map(fmt, remove_columns=cols)
    eval_ds  = eval_raw.map(fmt,  remove_columns=cols)

    print(f"[data] dair-ai/emotion | 학습: {len(train_ds)}건 / 평가: {len(eval_ds)}건")
    return DatasetDict({"train": train_ds, "eval": eval_ds})


def _load_ke_t5_kor(test_size: float) -> DatasetDict:
    """
    KETI-AIR/ke-t5-ko-sns-emotion 로드 (한국어).
    - 약 30K건 / 7감정 (기쁨·당황·분노·불안·상처·슬픔·중립)
    - CC BY-SA 4.0, SNS 텍스트 기반
    - 일기 감정 도메인과 가장 유사한 공개 한국어 데이터셋

    레이블: 문자열 또는 정수 (버전에 따라 다름)
    """
    try:
        raw = load_dataset("KETI-AIR/ke-t5-ko-sns-emotion", trust_remote_code=True)
    except Exception:
        # 데이터셋명 변경 대비 폴백
        raw = load_dataset("KETI-AIR/ke-t5-ko-emotion", trust_remote_code=True)

    # 단일 split인 경우 직접 분리
    if "train" not in raw:
        full  = raw[list(raw.keys())[0]]
        split = full.train_test_split(test_size=test_size, seed=SEED)
        train_raw, eval_raw = split["train"], split["test"]
    else:
        train_raw = raw["train"]
        eval_raw  = raw.get("validation") or raw.get("test") or \
                    raw["train"].train_test_split(test_size=test_size, seed=SEED)["test"]

    def fmt(ex):
        # 컬럼명은 버전에 따라 "label", "emotion", "감정" 등 다양
        raw_label = str(ex.get("label") or ex.get("emotion") or ex.get("감정") or "중립")
        emotion_ko = _KET5_LABEL2KO.get(raw_label, "중립")
        text = ex.get("text") or ex.get("sentence") or ex.get("문장") or ""
        return _make_prompt(text, _emotion_to_json(emotion_ko))

    cols = train_raw.column_names
    train_ds = train_raw.map(fmt, remove_columns=cols)
    eval_ds  = eval_raw.map(fmt,  remove_columns=cols)

    print(f"[data] ke-t5-ko-sns-emotion | 학습: {len(train_ds)}건 / 평가: {len(eval_ds)}건")
    return DatasetDict({"train": train_ds, "eval": eval_ds})


def _load_kor_7class(test_size: float) -> DatasetDict:
    """
    감성 대화 말뭉치 기반 7감정 한국어 데이터셋.
    HuggingFace Hub의 jth0809/Korean_Emotion_Conversation 사용.
    - 약 40K건 / 기쁨·놀람·분노·불안·혐오·슬픔·중립
    - MIT License
    """
    raw = load_dataset("jth0809/Korean_Emotion_Conversation", trust_remote_code=True)

    if "train" not in raw:
        full  = list(raw.values())[0]
        split = full.train_test_split(test_size=test_size, seed=SEED)
        train_raw, eval_raw = split["train"], split["test"]
    else:
        train_raw = raw["train"]
        eval_raw  = raw.get("test") or \
                    raw["train"].train_test_split(test_size=test_size, seed=SEED)["test"]

    # 컬럼 구조 확인 후 자동 선택
    sample    = train_raw[0]
    text_col  = next((c for c in ["utterance", "text", "발화", "sentence"] if c in sample), None)
    label_col = next((c for c in ["emotion", "label", "감정"] if c in sample), None)

    if not text_col or not label_col:
        raise ValueError(f"[data] 컬럼 자동 감지 실패. 컬럼 목록: {list(sample.keys())}")

    def fmt(ex):
        raw_label  = str(ex[label_col])
        emotion_ko = _KET5_LABEL2KO.get(raw_label, raw_label if raw_label in EMOTION_LABELS else "중립")
        return _make_prompt(ex[text_col], _emotion_to_json(emotion_ko))

    cols = train_raw.column_names
    train_ds = train_raw.map(fmt, remove_columns=cols)
    eval_ds  = eval_raw.map(fmt,  remove_columns=cols)

    print(f"[data] Korean_Emotion_Conversation | 학습: {len(train_ds)}건 / 평가: {len(eval_ds)}건")
    return DatasetDict({"train": train_ds, "eval": eval_ds})


# ══════════════════════════════════════════════════════════════
# 공개 API
# ══════════════════════════════════════════════════════════════

SUPPORTED_SOURCES = ("local_jsonl", "dair_emotion", "ke_t5_kor", "kor_7class")


def load_and_prepare(
    source: str = "local_jsonl",
    data_file: str = DATA_FILE,
    test_size: float = TRAIN_DEFAULTS["test_size"],
    max_samples: int | None = None,
) -> DatasetDict:
    """
    지정한 소스에서 데이터를 로드하고 EXAONE 프롬프트 포맷 적용 후 반환.

    Parameters
    ----------
    source      : 데이터 소스 선택
                  "local_jsonl" | "dair_emotion" | "ke_t5_kor" | "kor_7class"
    data_file   : source="local_jsonl" 일 때만 사용
    test_size   : 평가 데이터 비율
    max_samples : 전체 샘플 수 상한 (None이면 전체 사용).
                  지정 시 train:eval 비율은 test_size 기준으로 유지.
                  예) max_samples=500, test_size=0.1
                      → train 450건, eval 50건

    Returns
    -------
    DatasetDict : {"train": Dataset, "eval": Dataset}
                  각 샘플의 컬럼은 "text" 하나만 남음.
    """
    if source not in SUPPORTED_SOURCES:
        raise ValueError(f"지원하지 않는 source: '{source}'. "
                         f"선택 가능: {SUPPORTED_SOURCES}")

    print(f"[data] source='{source}' 로드 시작"
          + (f" (최대 {max_samples}건)" if max_samples else "") + "...")

    if source == "local_jsonl":
        ds = _load_local_jsonl(data_file, test_size)
    elif source == "dair_emotion":
        ds = _load_dair_emotion(test_size)
    elif source == "ke_t5_kor":
        ds = _load_ke_t5_kor(test_size)
    elif source == "kor_7class":
        ds = _load_kor_7class(test_size)

    # ── max_samples 적용 ──────────────────────────────────────
    if max_samples is not None and max_samples > 0:
        n_eval  = max(1, int(max_samples * test_size))
        n_train = max(1, max_samples - n_eval)

        # 현재 크기를 초과하지 않도록 clamp
        n_train = min(n_train, len(ds["train"]))
        n_eval  = min(n_eval,  len(ds["eval"]))

        ds = DatasetDict({
            "train": ds["train"].shuffle(seed=SEED).select(range(n_train)),
            "eval":  ds["eval"].shuffle(seed=SEED).select(range(n_eval)),
        })
        print(f"[data] 샘플 제한 적용 — 학습: {n_train}건 / 평가: {n_eval}건")

    print(f"[data] 완료 — 학습: {len(ds['train'])}건 / 평가: {len(ds['eval'])}건")
    return ds
