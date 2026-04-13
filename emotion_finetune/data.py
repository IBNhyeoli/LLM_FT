"""
data.py
───────
데이터 로드, 포맷, 분리 모듈.

지원 데이터 소스
----------------
- local_jsonl  : 로컬 JSONL 파일 (기존 방식)
- dair_emotion : dair-ai/emotion — 영문 6감정, 436K건 (smoke test용)
- ke_t5_kor    : KETI-AIR/ke-t5-ko-sns-emotion — 한국어 7감정, 약 30K건
- kor_7class   : passing2961/KorEmpatheticDialogues — 한국어 32감정, 24K건
"""

from datasets import load_dataset, DatasetDict
import json

from .config import DATA_FILE, SEED, EMOTION_LABELS, TRAIN_DEFAULTS


# ══════════════════════════════════════════════════════════════
# 레이블 매핑 상수
# ══════════════════════════════════════════════════════════════

# dair-ai/emotion: sadness(0) joy(1) love(2) anger(3) fear(4) surprise(5)
_DAIR_ID2KO = {
    0: "슬픔",
    1: "기쁨",
    2: "기쁨",
    3: "분노",
    4: "불안",
    5: "놀람",
}

# ke-t5 한국어 감성 레이블 → 프로젝트 7감정
_KET5_LABEL2KO = {
    "기쁨": "기쁨", "당황": "놀람", "분노": "분노",
    "불안": "불안", "상처": "슬픔", "슬픔": "슬픔", "중립": "중립",
    "0": "기쁨", "1": "당황", "2": "분노",
    "3": "불안", "4": "상처", "5": "슬픔", "6": "중립",
}

# KorEmpatheticDialogues 32감정 → 프로젝트 7감정
_KOR_EMP_TO_7 = {
    "joyful": "기쁨", "excited": "기쁨", "proud": "기쁨",
    "grateful": "기쁨", "content": "기쁨", "hopeful": "기쁨",
    "impressed": "기쁨", "confident": "기쁨",
    "trusting": "기쁨", "anticipating": "기쁨",
    "sad": "슬픔", "lonely": "슬픔", "sentimental": "슬픔",
    "nostalgic": "슬픔", "disappointed": "슬픔", "guilty": "슬픔",
    "ashamed": "슬픔", "devastated": "슬픔",
    "angry": "분노", "furious": "분노", "annoyed": "분노",
    "disgusted": "혐오",
    "afraid": "불안", "anxious": "불안", "apprehensive": "불안",
    "terrified": "불안",
    "surprised": "놀람",
    "caring": "중립", "faithful": "중립", "prepared": "중립",
    "jealous": "중립", "embarrassed": "중립",
}


# ══════════════════════════════════════════════════════════════
# 공통 유틸리티
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
    """감정 레이블 → JSON 문자열 변환 (intensity 등 기본값으로 채움)"""
    valence_map = {
        "기쁨": "positive", "슬픔": "negative", "분노": "negative",
        "불안": "negative", "혐오": "negative", "놀람": "neutral", "중립": "neutral",
    }
    return json.dumps({
        "primary_emotion":    primary_emotion,
        "intensity":          0.5,
        "tone":               primary_emotion + "의",
        "scene_keywords":     [],
        "secondary_emotions": [],
        "valence":            valence_map.get(primary_emotion, "neutral"),
    }, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════
# 데이터소스별 어댑터
# ══════════════════════════════════════════════════════════════

def _load_local_jsonl(data_file: str, test_size: float) -> DatasetDict:
    """로컬 JSONL 로드. 포맷: {"input": "...", "output": "..."}"""
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
    dair-ai/emotion (영문, 6감정, 436K건)
    smoke test용 — 빠른 기능 검증에 적합
    """
    raw = load_dataset("dair-ai/emotion", "split")

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
    KETI-AIR/ke-t5-ko-sns-emotion (한국어, 7감정, 약 30K건)
    CC BY-SA 4.0
    """
    try:
        raw = load_dataset("KETI-AIR/ke-t5-ko-sns-emotion")
    except Exception:
        raw = load_dataset("KETI-AIR/ke-t5-ko-emotion")

    if "train" not in raw:
        full  = raw[list(raw.keys())[0]]
        split = full.train_test_split(test_size=test_size, seed=SEED)
        train_raw, eval_raw = split["train"], split["test"]
    else:
        train_raw = raw["train"]
        eval_raw  = raw.get("validation") or raw.get("test") or \
                    raw["train"].train_test_split(test_size=test_size, seed=SEED)["test"]

    def fmt(ex):
        raw_label  = str(ex.get("label") or ex.get("emotion") or ex.get("감정") or "중립")
        emotion_ko = _KET5_LABEL2KO.get(raw_label, "중립")
        text       = ex.get("text") or ex.get("sentence") or ex.get("문장") or ""
        return _make_prompt(text, _emotion_to_json(emotion_ko))

    cols = train_raw.column_names
    train_ds = train_raw.map(fmt, remove_columns=cols)
    eval_ds  = eval_raw.map(fmt,  remove_columns=cols)

    print(f"[data] ke-t5-ko-sns-emotion | 학습: {len(train_ds)}건 / 평가: {len(eval_ds)}건")
    return DatasetDict({"train": train_ds, "eval": eval_ds})


def _load_kor_7class(test_size: float) -> DatasetDict:
    """
    passing2961/KorEmpatheticDialogues (한국어, 32감정 → 7감정 매핑, 24K건)
    - Parquet 형식, 바로 로드 가능
    - CC BY-NC 4.0
    - situation 컬럼을 input 텍스트로 사용
    """
    raw = load_dataset("passing2961/KorEmpatheticDialogues")

    train_raw = raw["train"]
    eval_raw  = raw.get("validation") or raw.get("test")

    if eval_raw is None:
        split     = train_raw.train_test_split(test_size=test_size, seed=SEED)
        train_raw = split["train"]
        eval_raw  = split["test"]

    def fmt(ex):
        raw_label  = str(ex.get("emotion", "중립")).lower()
        emotion_ko = _KOR_EMP_TO_7.get(raw_label, "중립")
        text       = ex.get("situation", "")
        return _make_prompt(text, _emotion_to_json(emotion_ko))

    cols = train_raw.column_names
    train_ds = train_raw.map(fmt, remove_columns=cols)
    eval_ds  = eval_raw.map(fmt,  remove_columns=cols)

    print(f"[data] KorEmpatheticDialogues | 학습: {len(train_ds)}건 / 평가: {len(eval_ds)}건")
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
    source      : "local_jsonl" | "dair_emotion" | "ke_t5_kor" | "kor_7class"
    data_file   : source="local_jsonl" 일 때만 사용
    test_size   : 평가 데이터 비율
    max_samples : 전체 샘플 수 상한 (None이면 전체 사용)

    Returns
    -------
    DatasetDict : {"train": Dataset, "eval": Dataset}
    """
    if source not in SUPPORTED_SOURCES:
        raise ValueError(
            f"지원하지 않는 source: '{source}'. 선택 가능: {SUPPORTED_SOURCES}"
        )

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

    # max_samples 적용
    if max_samples is not None and max_samples > 0:
        n_eval  = max(1, int(max_samples * test_size))
        n_train = max(1, max_samples - n_eval)
        n_train = min(n_train, len(ds["train"]))
        n_eval  = min(n_eval,  len(ds["eval"]))
        ds = DatasetDict({
            "train": ds["train"].shuffle(seed=SEED).select(range(n_train)),
            "eval":  ds["eval"].shuffle(seed=SEED).select(range(n_eval)),
        })
        print(f"[data] 샘플 제한 적용 — 학습: {n_train}건 / 평가: {n_eval}건")

    print(f"[data] 완료 — 학습: {len(ds['train'])}건 / 평가: {len(ds['eval'])}건")
    return ds