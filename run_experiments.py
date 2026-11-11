import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from grader_contracts.linear_models import ClassificationInput, RegressionInput
from linear_models import (
    classification_metrics,
    compare_classical_models,
    train_linear_regression,
    train_logistic_classifier,
)


ROOT = Path(__file__).resolve().parent


def regression_data():
    frame = pd.read_csv(ROOT / "Student_Performance.csv")
    features = frame.drop(columns="Performance Index")
    features["Extracurricular Activities"] = features["Extracurricular Activities"].map({"Yes": 1, "No": 0})
    target = frame["Performance Index"]
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.2, random_state=42,
    )
    scaler = StandardScaler()
    return RegressionInput(scaler.fit_transform(train_x), train_y.to_numpy(),
                           scaler.transform(test_x), test_y.to_numpy())


def classification_data():
    frame = pd.read_csv(ROOT / "german.csv", sep=";")
    features = frame.iloc[:, 1:]
    target = frame.iloc[:, 0]
    train_x, test_x, train_y, test_y = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=target,
    )
    numeric = [
        "Duration_of_Credit_monthly", "Credit_Amount", "Instalment_per_cent",
        "Duration_in_Current_address", "Age_years", "No_of_Credits_at_this_Bank",
        "No_of_dependents",
    ]
    categorical = [name for name in features if name not in numeric]
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
    ])
    return ClassificationInput(preprocessor.fit_transform(train_x), train_y.to_numpy(),
                               preprocessor.transform(test_x), test_y.to_numpy())


def main():
    regression = regression_data()
    classification = classification_data()
    own_linear = train_linear_regression(regression)
    own_logistic = train_logistic_classifier(classification)
    comparison = compare_classical_models(classification)
    results = {
        "linear_regression": {"test_mse": float(mean_squared_error(regression.test_target, own_linear.predictions))},
        "classification": {
            "own_logistic": asdict(classification_metrics(
                classification.test_target, own_logistic.probabilities, own_logistic.predictions,
            )),
            **asdict(comparison),
        },
    }
    (ROOT / "experiment_results.json").write_text(json.dumps(results, indent=2) + "\n")
    plots = ROOT / "plots"
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    experiment_log = {}
    for name, data, trainer, rates, axis in [
        ("linear", regression, train_linear_regression, [0.01, 0.05, 0.1], axes[0]),
        ("logistic", classification, train_logistic_classifier, [0.01, 0.1, 0.5], axes[1]),
    ]:
        experiments = []
        for rate in rates:
            run = trainer(type(data)(data.train_features, data.train_target,
                                     data.test_features, data.test_target, rate, data.epochs))
            initial_loss = (float(np.mean(np.asarray(data.train_target) ** 2))
                            if name == "linear" else float(np.log(2)))
            axis.plot(range(data.epochs + 1), [initial_loss] + run.loss_history, label=f"lr={rate}")
            experiments.append({"learning_rate": rate, "initial_loss": initial_loss,
                                "first_epoch_loss": run.loss_history[0],
                                "final_loss": run.loss_history[-1]})
        experiment_log[name] = experiments
        axis.set(xlabel="Epoch", ylabel="MSE" if name == "linear" else "Binary cross-entropy")
        axis.set_yscale("log")
        axis.legend()
        axis.grid(alpha=0.3)
    fig.savefig(plots / "training_loss.png", dpi=150)
    plt.close(fig)
    print(json.dumps(results, indent=2))
    print(json.dumps(experiment_log, indent=2))


if __name__ == "__main__":
    main()
