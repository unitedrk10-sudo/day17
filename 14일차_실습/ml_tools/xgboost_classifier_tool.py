"""
Day 10 — XGBoost 분류
불균형 설비 고장 데이터에서 GBM 계열로 Failure 예측.

Agent Tool: predict_failure_xgboost
"""

from __future__ import annotations

import joblib
import pandas as pd
from langchain_core.tools import tool
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

try:
    from ._common import (
        MODELS_DIR,
        build_failure_row,
        load_failure_data,
        one_hot_align,
    )
except ImportError:
    from _common import (
        MODELS_DIR,
        build_failure_row,
        load_failure_data,
        one_hot_align,
    )

MODEL_PATH = MODELS_DIR / "xgboost_failure.joblib"


def train_xgboost_model() -> dict:
    """Day 3-2 분류 파이프라인: One-Hot → XGBClassifier."""
    x, y = load_failure_data()
    medians = x.drop(columns=["Operator"]).median(numeric_only=True).to_dict()

    x_enc = pd.get_dummies(x)
    train_columns = x_enc.columns.tolist()

    x_train, x_test, y_train, y_test = train_test_split(
        x_enc, y, test_size=0.3, random_state=0, stratify=y
    )

    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / max(pos, 1)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=0,
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    importances = sorted(
        zip(train_columns, model.feature_importances_),
        key=lambda x: x[1],
        reverse=True,
    )[:5]

    bundle = {
        "model": model,
        "train_columns": train_columns,
        "medians": medians,
        "top_features": importances,
        "metrics": {
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "report": classification_report(y_test, y_pred, zero_division=0),
        },
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def _load_bundle() -> dict:
    if not MODEL_PATH.exists():
        return train_xgboost_model()
    return joblib.load(MODEL_PATH)


def predict_failure_xgboost_raw(
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
    proba = bundle["model"].predict_proba(x)[0]
    pred = int(bundle["model"].predict(x)[0])
    return {
        "prediction": "Yes" if pred == 1 else "No",
        "failure_probability": float(proba[1]),
        "normal_probability": float(proba[0]),
    }


@tool
def predict_failure_xgboost(
    temperature: float,
    humidity: float,
    operator: str,
    hours_since_previous_failure: float,
) -> str:
    """설비 고장(Failure) 여부를 XGBoost(Day 3-2)로 예측합니다.

    Args:
        temperature: 공정 온도
        humidity: 습도
        operator: 작업자 ID (Operator1~3)
        hours_since_previous_failure: 이전 고장 이후 경과 시간(시간)

    불균형 데이터에 scale_pos_weight를 적용한 그래디언트 부스팅 분류입니다.
    """
    result = predict_failure_xgboost_raw(
        temperature, humidity, operator, hours_since_previous_failure
    )
    bundle = _load_bundle()
    top = ", ".join(f"{name}({score:.3f})" for name, score in bundle["top_features"][:3])
    return (
        f"[XGBoost] 예측: {result['prediction']}\n"
        f"- 고장(Failure) 확률: {result['failure_probability']:.2%}\n"
        f"- 정상(No) 확률: {result['normal_probability']:.2%}\n"
        f"- 주요 변수(학습 기준): {top}"
    )


if __name__ == "__main__":
    bundle = train_xgboost_model()
    print("모델 저장:", MODEL_PATH)
    print("검증 F1:", bundle["metrics"]["f1"])
    print(predict_failure_xgboost.invoke({
        "temperature": 67,
        "humidity": 82,
        "operator": "Operator1",
        "hours_since_previous_failure": 90,
    }))
