"""
data/ 폴더 CSV → 샘플 단위 ML 추론 Tool

Agent가 파일명 + 샘플 번호만으로 분석할 수 있게 합니다.
예: "failure_equipment.csv 3번 샘플 XGBoost로 확인해봐"
"""

from __future__ import annotations

import pandas as pd
from langchain_core.tools import tool

try:
    from ._common import (
        DATA_DIR,
        load_sample_row,
        list_csv_files,
        safe_data_path,
    )
    from .logistic_regression_tool import predict_failure_logistic_raw
    from .oneclass_svm_tool import detect_anomaly_oneclass_raw
    from .xgboost_classifier_tool import predict_failure_xgboost_raw
except ImportError:
    from _common import (
        DATA_DIR,
        load_sample_row,
        list_csv_files,
        safe_data_path,
    )
    from logistic_regression_tool import predict_failure_logistic_raw
    from oneclass_svm_tool import detect_anomaly_oneclass_raw
    from xgboost_classifier_tool import predict_failure_xgboost_raw


def _file_kind(filename: str) -> str:
    df = pd.read_csv(safe_data_path(filename), nrows=1)
    cols = set(df.columns)
    if "Failure" in cols:
        return "failure"
    if {"0", "1", "Y"}.issubset(cols):
        return "anomaly"
    raise ValueError(
        f"{filename} 형식을 알 수 없습니다. Failure 컬럼(고장) 또는 0,1,Y(이상탐지) 필요."
    )


def _failure_row_to_params(row: pd.Series) -> dict:
    measure_values = {c: float(row[c]) for c in row.index if str(c).startswith("Measure")}
    return {
        "temperature": float(row["Temperature"]),
        "humidity": float(row["Humidity"]),
        "operator": str(row["Operator"]),
        "hours_since_previous_failure": float(row["Hours Since Previous Failure"]),
        "measure_values": measure_values,
    }


def _format_failure_result(
    filename: str,
    sample_index: int,
    method: str,
    result: dict,
    actual: str | None,
) -> str:
    lines = [
        f"[{method}] 파일: {filename}, 샘플: {sample_index}번",
        f"- 예측: {result['prediction']}",
        f"- 고장(Failure) 확률: {result['failure_probability']:.2%}",
        f"- 정상(No) 확률: {result['normal_probability']:.2%}",
    ]
    if actual is not None:
        lines.append(f"- 실제 라벨(Failure): {actual}")
    return "\n".join(lines)


@tool
def list_data_files() -> str:
    """data/ 폴더에서 사용 가능한 CSV 파일 목록을 반환합니다."""
    files = list_csv_files()
    if not files:
        return f"data/ 폴더({DATA_DIR})에 CSV가 없습니다."
    lines = [f"data/ 폴더 CSV ({len(files)}개):"]
    for name in files:
        kind = _file_kind(name)
        df = pd.read_csv(safe_data_path(name))
        label = "설비 고장" if kind == "failure" else "이상 탐지"
        lines.append(f"- {name}: {len(df)}행, 유형={label}")
    return "\n".join(lines)


@tool
def preview_data_sample(filename: str, sample_index: int) -> str:
    """data/ CSV에서 N번째 샘플(1부터) 원본 값을 미리 봅니다.

    Args:
        filename: CSV 파일명 (예: failure_equipment.csv)
        sample_index: 샘플 번호 (1부터)
    """
    row = load_sample_row(filename, sample_index)
    kind = _file_kind(filename)
    header = f"{filename} — {sample_index}번 샘플 (유형: {kind})"
    body = row.to_string()
    return f"{header}\n{body}"


@tool
def xgboost_predict_from_data(filename: str, sample_index: int) -> str:
    """data/ CSV의 N번 샘플을 XGBoost로 설비 고장(Failure) 예측합니다.

    Args:
        filename: failure_equipment.csv 등 Failure 컬럼이 있는 CSV
        sample_index: 샘플 번호 (1부터). 예: 3 → 3번째 행
    """
    if _file_kind(filename) != "failure":
        raise ValueError(f"{filename}은 고장 예측용이 아닙니다. failure_equipment.csv를 사용하세요.")
    row = load_sample_row(filename, sample_index)
    params = _failure_row_to_params(row)
    result = predict_failure_xgboost_raw(**params)
    actual = str(row["Failure"]) if "Failure" in row.index else None
    return _format_failure_result(filename, sample_index, "XGBoost", result, actual)


@tool
def logistic_predict_from_data(filename: str, sample_index: int) -> str:
    """data/ CSV의 N번 샘플을 로지스틱 회귀로 설비 고장(Failure) 예측합니다.

    Args:
        filename: failure_equipment.csv 등
        sample_index: 샘플 번호 (1부터)
    """
    if _file_kind(filename) != "failure":
        raise ValueError(f"{filename}은 고장 예측용이 아닙니다.")
    row = load_sample_row(filename, sample_index)
    params = _failure_row_to_params(row)
    result = predict_failure_logistic_raw(**params)
    actual = str(row["Failure"]) if "Failure" in row.index else None
    return _format_failure_result(filename, sample_index, "로지스틱 회귀", result, actual)


@tool
def anomaly_detect_from_data(filename: str, sample_index: int) -> str:
    """data/ CSV의 N번 샘플을 One-Class SVM으로 정상/이상 판정합니다.

    Args:
        filename: anomaly_sensor.csv (컬럼 0, 1, Y)
        sample_index: 샘플 번호 (1부터)
    """
    if _file_kind(filename) != "anomaly":
        raise ValueError(f"{filename}은 이상 탐지용이 아닙니다. anomaly_sensor.csv를 사용하세요.")
    row = load_sample_row(filename, sample_index)
    result = detect_anomaly_oneclass_raw(float(row["0"]), float(row["1"]))
    actual = int(row["Y"])
    actual_label = "정상" if actual == 1 else "이상"
    return (
        f"[One-Class SVM] 파일: {filename}, 샘플: {sample_index}번\n"
        f"- 판정: {result['label']}\n"
        f"- decision_score: {result['decision_score']:.4f}\n"
        f"- 실제 라벨(Y): {actual} ({actual_label})"
    )


DATA_TOOLS = [
    list_data_files,
    preview_data_sample,
    xgboost_predict_from_data,
    logistic_predict_from_data,
    anomaly_detect_from_data,
]

if __name__ == "__main__":
    print(list_data_files.invoke({}))
    print(preview_data_sample.invoke({
        "filename": "failure_equipment.csv",
        "sample_index": 3,
    })[:300], "...")
    print(xgboost_predict_from_data.invoke({
        "filename": "failure_equipment.csv",
        "sample_index": 3,
    }))
    print(anomaly_detect_from_data.invoke({
        "filename": "anomaly_sensor.csv",
        "sample_index": 3,
    }))
