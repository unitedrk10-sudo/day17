"""
Day 9 — 로지스틱 회귀 (이진 분류)
설비 고장(Failure) Yes/No 예측 + 확률 반환.

Agent Tool: predict_failure_logistic
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

try:
    from ._common import (
        MEASURE_COLUMNS,
        MODELS_DIR,
        build_failure_row,
        load_failure_data,
        one_hot_align,
    )
except ImportError:
    from _common import (
        MEASURE_COLUMNS,
        MODELS_DIR,
        build_failure_row,
        load_failure_data,
        one_hot_align,
    )

MODEL_PATH = MODELS_DIR / "logistic_failure.joblib"


def train_logistic_model() -> dict:
    """Day 2-2 파이프라인: One-Hot → MinMaxScaler → LogisticRegression."""
    x, y = load_failure_data()
    medians = x[MEASURE_COLUMNS].median().to_dict()

    x_enc = pd.get_dummies(x)
    train_columns = x_enc.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x_enc, y, test_size=0.3, random_state=0, stratify=y
    )

    scaler = MinMaxScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    model = LogisticRegression(max_iter=1000, random_state=0)
    model.fit(x_train_s, y_train)

    y_pred = model.predict(x_test_s)
    metrics = {
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }

    bundle = {
        "model": model,
        "scaler": scaler,
        "train_columns": train_columns,
        "medians": medians,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def _load_bundle() -> dict:
    if not MODEL_PATH.exists():
        return train_logistic_model()

    bundle = joblib.load(MODEL_PATH)
    try:
        probe = one_hot_align(
            build_failure_row(50.0, 50.0, "Operator1", 24.0, medians=bundle["medians"]),
            bundle["train_columns"],
        )
        bundle["model"].predict_proba(bundle["scaler"].transform(probe))
    except AttributeError:
        # sklearn 버전이 바뀌어 예전 joblib 모델이 호환되지 않을 때 재학습
        return train_logistic_model()
    return bundle


def predict_failure_logistic_raw(
    temperature: float,
    humidity: float,
    operator: str,
    hours_since_previous_failure: float,
    measure_values: dict[str, float] | None = None,
) -> dict:
    bundle = _load_bundle()
    row = build_failure_row(
        temperature,
        humidity,
        operator,
        hours_since_previous_failure,
        measure_values,
        bundle["medians"],
    )
    x = one_hot_align(row, bundle["train_columns"])
    x_s = bundle["scaler"].transform(x)
    proba = bundle["model"].predict_proba(x_s)[0]
    pred = int(bundle["model"].predict(x_s)[0])
    return {
        "prediction": "Yes" if pred == 1 else "No",
        "failure_probability": float(proba[1]),
        "normal_probability": float(proba[0]),
    }


@tool
def predict_failure_logistic(
    temperature: float,
    humidity: float,
    operator: str,
    hours_since_previous_failure: float,
) -> str:
    """설비 고장(Failure) 여부를 로지스틱 회귀(Day 2-2)로 예측합니다.

    Args:
        temperature: 공정 온도 (예: 67)
        humidity: 습도 (예: 82)
        operator: 작업자 ID (예: Operator1, Operator2, Operator3)
        hours_since_previous_failure: 이전 고장 이후 경과 시간(시간)

    Measure1~15는 미입력 시 학습 데이터 중앙값으로 대체됩니다.
  """
    result = predict_failure_logistic_raw(
        temperature, humidity, operator, hours_since_previous_failure
    )
    return (
        f"[로지스틱 회귀] 예측: {result['prediction']}\n"
        f"- 고장(Failure) 확률: {result['failure_probability']:.2%}\n"
        f"- 정상(No) 확률: {result['normal_probability']:.2%}"
    )


if __name__ == "__main__":
    bundle = train_logistic_model()
    print("모델 저장:", MODEL_PATH)
    print("검증 F1:", bundle["metrics"]["f1"])
    print(predict_failure_logistic.invoke({
        "temperature": 67,
        "humidity": 82,
        "operator": "Operator1",
        "hours_since_previous_failure": 90,
    }))
