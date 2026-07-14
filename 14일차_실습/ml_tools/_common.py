"""ml_tools 공통 유틸 — 경로, 데이터 로드, 피처 정렬."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ML_TOOLS_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = ML_TOOLS_DIR.parent
DATA_DIR = WORKSPACE_DIR / "data"
MODELS_DIR = ML_TOOLS_DIR / "models"

FAILURE_CSV = DATA_DIR / "failure_equipment.csv"
ANOMALY_CSV = DATA_DIR / "anomaly_sensor.csv"

MEASURE_COLUMNS = [f"Measure{i}" for i in range(1, 16)]
FAILURE_INPUT_COLUMNS = [
    "Temperature",
    "Humidity",
    "Operator",
    *MEASURE_COLUMNS,
    "Hours Since Previous Failure",
]


def load_failure_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(FAILURE_CSV)
    y = df["Failure"].replace({"No": 0, "Yes": 1}).astype(int)
    x = df.drop(columns=["Failure"])
    return x, y


def load_anomaly_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(ANOMALY_CSV)
    x = df[["0", "1"]]
    y = df["Y"]
    return x, y


def build_failure_row(
    temperature: float,
    humidity: float,
    operator: str,
    hours_since_previous_failure: float,
    measure_values: dict[str, float] | None = None,
    medians: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Measure1~15 미입력 시 학습 데이터 중앙값으로 채움."""
    row: dict[str, Any] = {
        "Temperature": temperature,
        "Humidity": humidity,
        "Operator": operator,
        "Hours Since Previous Failure": hours_since_previous_failure,
    }
    measure_values = measure_values or {}
    medians = medians or {}
    for col in MEASURE_COLUMNS:
        row[col] = measure_values.get(col, medians.get(col, 0.0))
    return row


def one_hot_align(row: dict[str, Any], train_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([row])
    encoded = pd.get_dummies(df)
    return encoded.reindex(columns=train_columns, fill_value=0)


def safe_data_path(filename: str) -> Path:
    """data/ 폴더 내 CSV만 허용 (경로 탈취 방지)."""
    name = Path(filename).name
    if not name.endswith(".csv"):
        raise ValueError(f"CSV 파일만 사용할 수 있습니다: {filename}")
    path = DATA_DIR / name
    if not path.exists():
        available = ", ".join(p.name for p in sorted(DATA_DIR.glob("*.csv")))
        raise FileNotFoundError(f"파일 없음: {name}. 사용 가능: {available}")
    return path


def load_sample_row(filename: str, sample_index: int) -> pd.Series:
    """sample_index는 1부터 시작 (사용자 친화)."""
    if sample_index < 1:
        raise ValueError("sample_index는 1 이상이어야 합니다.")
    df = pd.read_csv(safe_data_path(filename))
    if sample_index > len(df):
        raise ValueError(
            f"{filename}에는 {len(df)}개 샘플만 있습니다. 요청: {sample_index}번"
        )
    return df.iloc[sample_index - 1]


def list_csv_files() -> list[str]:
    return sorted(p.name for p in DATA_DIR.glob("*.csv"))
