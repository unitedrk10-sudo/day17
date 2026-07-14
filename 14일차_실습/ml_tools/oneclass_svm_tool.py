"""
Day 11 — One-Class SVM (SVDD 실습 대응)
정상 데이터만으로 학습한 뒤 이상치(Anomaly) 탐지.

Agent Tool: detect_anomaly_oneclass
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from langchain_core.tools import tool
from sklearn.metrics import classification_report, f1_score
from sklearn.svm import OneClassSVM

try:
    from ._common import ANOMALY_CSV, MODELS_DIR, load_anomaly_data
except ImportError:
    from _common import ANOMALY_CSV, MODELS_DIR, load_anomaly_data

MODEL_PATH = MODELS_DIR / "oneclass_svm.joblib"


def train_oneclass_model(nu: float = 0.1) -> dict:
    """Day 4-3: Y=1(정상) 데이터만으로 OneClassSVM 학습."""
    x, y = load_anomaly_data()
    x_normal = x[y == 1]

    model = OneClassSVM(kernel="rbf", gamma="auto", nu=nu)
    model.fit(x_normal)

    # 전체 데이터로 참고용 성능 (학습 시에는 정상만 사용)
    pred = model.predict(x)
    pred_label = np.where(pred == 1, 1, -1)
    metrics = {
        "f1": float(f1_score(y, pred_label, pos_label=1, zero_division=0)),
        "report": classification_report(y, pred_label, zero_division=0),
    }

    bundle = {
        "model": model,
        "nu": nu,
        "metrics": metrics,
        "feature_names": ["0", "1"],
    }
    joblib.dump(bundle, MODEL_PATH)
    return bundle


def _load_bundle() -> dict:
    if not MODEL_PATH.exists():
        return train_oneclass_model()
    return joblib.load(MODEL_PATH)


def detect_anomaly_oneclass_raw(feature_0: float, feature_1: float) -> dict:
    bundle = _load_bundle()
    x = pd.DataFrame([[feature_0, feature_1]], columns=bundle["feature_names"])
    pred = int(bundle["model"].predict(x)[0])
    score = float(bundle["model"].decision_function(x)[0])
    return {
        "label": "정상" if pred == 1 else "이상",
        "raw_prediction": pred,
        "decision_score": score,
    }


@tool
def detect_anomaly_oneclass(feature_0: float, feature_1: float) -> str:
    """2차원 센서값이 정상 범위인지 One-Class SVM(Day 4-3 / SVDD)으로 탐지합니다.

    Args:
        feature_0: 센서 축 0 값 (학습 데이터 컬럼 `0`)
        feature_1: 센서 축 1 값 (학습 데이터 컬럼 `1`)

    +1=정상, -1=이상. decision_score가 클수록 정상에 가깝습니다.
    """
    result = detect_anomaly_oneclass_raw(feature_0, feature_1)
    return (
        f"[One-Class SVM] 판정: {result['label']}\n"
        f"- decision_score: {result['decision_score']:.4f}\n"
        f"- raw_prediction: {result['raw_prediction']} (+1 정상 / -1 이상)"
    )


if __name__ == "__main__":
    bundle = train_oneclass_model()
    print("모델 저장:", MODEL_PATH)
    print("참고 F1:", bundle["metrics"]["f1"])
    print(detect_anomaly_oneclass.invoke({"feature_0": 1.0, "feature_1": 1.5}))
    print(detect_anomaly_oneclass.invoke({"feature_0": 5.0, "feature_1": -4.0}))
