import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationResult,
    ClassificationMetrics,
    RegressionInput,
    RegressionResult,
)


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    features = np.asarray(data.train_features, dtype=float)
    target = np.asarray(data.train_target, dtype=float).reshape(-1)
    test_features = np.asarray(data.test_features, dtype=float)
    weights = np.zeros(features.shape[1])
    bias = 0.0
    loss_history = []
    for _ in range(data.epochs):
        errors = features @ weights + bias - target
        weights -= data.learning_rate * 2 * (features.T @ errors) / len(target)
        bias -= data.learning_rate * 2 * errors.mean()
        loss_history.append(float(np.mean((features @ weights + bias - target) ** 2)))
    return RegressionResult(weights, float(bias), loss_history, test_features @ weights + bias)


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    features = np.asarray(data.train_features, dtype=float)
    target = np.asarray(data.train_target, dtype=float).reshape(-1)
    test_features = np.asarray(data.test_features, dtype=float)
    weights = np.zeros(features.shape[1])
    bias = 0.0
    loss_history = []
    for _ in range(data.epochs):
        scores = features @ weights + bias
        probabilities = np.exp(-np.logaddexp(0, -scores))
        errors = probabilities - target
        weights -= data.learning_rate * (features.T @ errors) / len(target)
        bias -= data.learning_rate * errors.mean()
        scores = features @ weights + bias
        loss_history.append(float(np.mean(np.logaddexp(0, scores) - target * scores)))
    probabilities = np.exp(-np.logaddexp(0, -(test_features @ weights + bias)))
    predictions = (probabilities >= 0.5).astype(int)
    return ClassificationResult(weights, float(bias), loss_history, probabilities, predictions)


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split.

    После запуска запишите реальные результаты в experiment_results.json:
    {"linear_regression":{"test_mse": число},
     "classification":{"own_logistic": {"roc_auc": ..., "accuracy": ..., "precision": ..., "recall": ..., "f1": ...},
                       "logistic_regression": {...}, "decision_tree": {...}, "knn": {...}}}.
    """
    results = []
    for model in (
        LogisticRegression(max_iter=1000, random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        KNeighborsClassifier(n_neighbors=5),
    ):
        model.fit(data.train_features, np.asarray(data.train_target).reshape(-1))
        probabilities = model.predict_proba(data.test_features)[:, 1]
        predictions = model.predict(data.test_features)
        results.append(classification_metrics(data.test_target, probabilities, predictions))
    return BaselineComparisonResult(*results)


def classification_metrics(target, probabilities, predictions):
    return ClassificationMetrics(
        roc_auc=float(roc_auc_score(target, probabilities)),
        accuracy=float(accuracy_score(target, predictions)),
        precision=float(precision_score(target, predictions, zero_division=0)),
        recall=float(recall_score(target, predictions, zero_division=0)),
        f1=float(f1_score(target, predictions, zero_division=0)),
    )
