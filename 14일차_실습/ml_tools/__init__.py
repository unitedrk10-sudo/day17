"""
Day 8~11 머신러닝 방법론 → LangChain Agent Tool

| 모듈 | 강의 | 알고리즘 | 용도 |
|------|------|----------|------|
| logistic_regression_tool | Day 9 (2-2) | LogisticRegression | 설비 고장 이진 분류 + 확률 |
| xgboost_classifier_tool | Day 10 (3-2) | XGBClassifier | 불균형 데이터 고장 예측 |
| oneclass_svm_tool | Day 11 (4-3) | OneClassSVM | 정상 기준 이상 탐지 |
| data_tools | Day 14-4 | 위 모델 + data/ CSV | 파일명·샘플 번호로 추론 |
"""

from .data_tools import (
    DATA_TOOLS,
    anomaly_detect_from_data,
    list_data_files,
    logistic_predict_from_data,
    preview_data_sample,
    xgboost_predict_from_data,
)
from .logistic_regression_tool import predict_failure_logistic
from .oneclass_svm_tool import detect_anomaly_oneclass
from .xgboost_classifier_tool import predict_failure_xgboost

ML_TOOLS = [
    predict_failure_logistic,
    predict_failure_xgboost,
    detect_anomaly_oneclass,
]

ALL_ML_TOOLS = ML_TOOLS + DATA_TOOLS

__all__ = [
    "ML_TOOLS",
    "DATA_TOOLS",
    "ALL_ML_TOOLS",
    "list_data_files",
    "preview_data_sample",
    "xgboost_predict_from_data",
    "logistic_predict_from_data",
    "anomaly_detect_from_data",
    "predict_failure_logistic",
    "predict_failure_xgboost",
    "detect_anomaly_oneclass",
]
