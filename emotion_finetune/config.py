"""
config.py
─────────
전역 상수, 감정 레이블, 경로 설정.
다른 모든 모듈은 이 파일에서 상수를 import한다.

실행 환경 자동 감지
-------------------
환경 감지 우선순위:
  1. 환경변수 EMOTION_PROJECT_ROOT  → 직접 지정 (최우선)
  2. /content 존재 + Windows 아님   → Google Colab (브라우저)
  3. Windows OS                     → 로컬 Windows (VSCode 등)
  4. 그 외                          → 로컬 Linux/macOS

현재 프로젝트 구조 (Windows 로컬 기준)
---------------------------------------
  E:\\Dev\\LLM_FT\\
  ├── main.py
  ├── final_diary_data.jsonl
  └── emotion_finetune\\
      ├── __init__.py
      ├── config.py          <- 이 파일
      └── ...

경로를 수동으로 고정하고 싶다면 실행 전에 환경변수를 설정하세요:
  # Windows PowerShell
  $env:EMOTION_PROJECT_ROOT = "E:\\Dev\\LLM_FT"
  python main.py

  # Windows CMD
  set EMOTION_PROJECT_ROOT=E:\\Dev\\LLM_FT
  python main.py

  # Colab 셀 / Linux 터미널
  export EMOTION_PROJECT_ROOT="/my/custom/path"
  python main.py
"""

import os
import sys
from pathlib import Path


# ══════════════════════════════════════════════════════════════
# 실행 환경 감지
# ══════════════════════════════════════════════════════════════

def _detect_env() -> str:
    """
    실행 환경을 감지해 문자열로 반환.
    반환값: "colab" | "windows" | "local"
    """
    # 환경변수가 명시된 경우 — OS에 관계없이 해당 OS 기준으로 처리
    if os.environ.get("EMOTION_PROJECT_ROOT"):
        return "windows" if sys.platform == "win32" else "local"

    # Windows → 반드시 로컬
    if sys.platform == "win32":
        return "windows"

    # /content 존재 + 비Windows → Colab
    if Path("/content").exists():
        return "colab"

    return "local"


def _resolve_project_root() -> Path:
    """
    프로젝트 루트 경로를 환경에 맞게 결정.

    결정 순서:
      1. 환경변수 EMOTION_PROJECT_ROOT 우선
      2. Colab → /content
      3. 로컬(Windows/Linux/macOS) → config.py 기준 상위 폴더
         구조: <project_root>/emotion_finetune/config.py
               → parent = emotion_finetune/
               → parent.parent = <project_root>/
    """
    env_root = os.environ.get("EMOTION_PROJECT_ROOT")
    if env_root:
        # Windows 경로 구분자 혼용 방어 (슬래시 <-> 역슬래시)
        return Path(env_root).resolve()

    if _detect_env() == "colab":
        return Path("/content")

    # 로컬 (Windows 포함): config.py -> emotion_finetune/ -> project_root/
    return Path(__file__).resolve().parent.parent


# ── 경로 확정 ──────────────────────────────────────────────────
ENV          = _detect_env()
PROJECT_ROOT : Path = _resolve_project_root()
DATA_FILE    : str  = str(PROJECT_ROOT / "final_diary_data.jsonl")
OUTPUT_ROOT  : str  = str(PROJECT_ROOT / "outputs")

_ENV_LABEL = {
    "colab":   "Colab (브라우저)",
    "windows": "로컬 Windows (VSCode)",
    "local":   "로컬 Linux/macOS",
}

# 시작 시 경로 출력 (디버깅용)
print(f"[config] 환경:          {_ENV_LABEL.get(ENV, ENV)}")
print(f"[config] 프로젝트 루트: {PROJECT_ROOT}")
print(f"[config] 데이터 파일:   {DATA_FILE}")
print(f"[config] 출력 디렉터리: {OUTPUT_ROOT}")


# ══════════════════════════════════════════════════════════════
# 모델
# ══════════════════════════════════════════════════════════════

MODEL_ID = "LGAI-EXAONE/EXAONE-3.5-2.8B-Instruct"


# ══════════════════════════════════════════════════════════════
# 재현성
# ══════════════════════════════════════════════════════════════

SEED = 42


# ══════════════════════════════════════════════════════════════
# 감정 레이블
# ══════════════════════════════════════════════════════════════

# 데이터셋 레이블에 맞게 이 목록만 수정하면 전체에 반영됨
EMOTION_LABELS: list[str] = ["기쁨", "슬픔", "분노", "불안", "혐오", "놀람", "중립"]
LABEL2ID: dict[str, int]  = {label: idx for idx, label in enumerate(EMOTION_LABELS)}
ID2LABEL: dict[int, str]  = {idx: label for label, idx in LABEL2ID.items()}


# ══════════════════════════════════════════════════════════════
# EXAONE 레이어 모듈명
# ══════════════════════════════════════════════════════════════

# 실제 모듈명은 모델 로드 후 아래 명령으로 확인:
#   for name, _ in model.named_modules(): print(name)
EXAONE_ATTN_MODULES: list[str] = ["q_proj", "k_proj", "v_proj", "o_proj"]
EXAONE_FF_MODULES:   list[str] = ["gate_proj", "up_proj", "down_proj"]


# ══════════════════════════════════════════════════════════════
# 학습 기본값
# ══════════════════════════════════════════════════════════════

TRAIN_DEFAULTS = dict(
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate               = 2e-4,
    num_train_epochs            = 3,
    max_seq_length              = 1024,
    test_size                   = 0.1,
    logging_steps               = 10,
)


# ══════════════════════════════════════════════════════════════
# 출력 스키마 예시 (파이프라인 2단계 참고용)
# ══════════════════════════════════════════════════════════════

EMOTION_SCHEMA_EXAMPLE = {
    "primary_emotion":    "슬픔",
    "intensity":          0.7,       # 0.0~1.0
    "tone":               "침잠된",
    "scene_keywords":     ["비", "창가", "혼자"],
    "secondary_emotions": ["외로움", "그리움"],
    "valence":            "negative",  # positive / negative / neutral
}
