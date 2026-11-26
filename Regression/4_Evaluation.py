import importlib.util
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"
RESULTS_DIR = BASE_DIR / "Results"

RANDOM_STATE = 42


def load_training_module():
    path = BASE_DIR / "3_Model_Training.py"

    spec = importlib.util.spec_from_file_location(
        "training_module", path
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module


def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)

    return {
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted)
    }


def main():

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    training = load_training_module()

    # 1) Load preprocessed datasets
    train = pd.read_csv(DATA_DIR / "train_preprocessed.csv")
    test = pd.read_csv(DATA_DIR / "test_preprocessed.csv")

    # 2) Load best model selected during training
    validation_path = BASE_DIR / "validation_results.csv"

    if not validation_path.exists():
        raise SystemExit(
            "Run 3_Model_Training.py first."
        )

    results = pd.read_csv(validation_path)
    best = results.sort_values("Val_RMSE").iloc[0]

    best_name = best["Model"]
    use_weights = best["Weights"]

    if isinstance(use_weights, str):
        use_weights = use_weights.lower() == "true"
    else:
        use_weights = bool(use_weights)

    # 3) Prepare Features and Target
    features = train.drop(
        columns=training.DROP_COLS
    ).columns.tolist()

    X_train = train[features]
    X_test = test[features]

    y_train = train["Price_log"]
    y_test = test["Price_log"]

    # 4) Train the selected model on full training data
    model = training.get_models()[best_name]

    model.fit(
        X_train,
        y_train,
        sample_weight=(
            train["sample_weight"] if use_weights else None
        )
    )

    print("Best Model:", best_name)
    print("Used Sample Weights:", use_weights)

    # 5) Make Predictions
    train_pred = np.expm1(model.predict(X_train))
    test_pred = np.expm1(model.predict(X_test))

    actual_train = train["Price"]
    actual_test = test["Price"]

    # 6) Calculate Evaluation Metrics
    metrics = pd.DataFrame(
        [
            calculate_metrics(actual_train, train_pred),
            calculate_metrics(actual_test, test_pred)
        ],
        index=["Train", "Test"]
    )

    print("\n========== Evaluation Metrics ==========\n")
    print(
        metrics.to_string(
            float_format="{:,.4f}".format
        )
    )

    metrics.to_csv(
        RESULTS_DIR / "evaluation_metrics.csv",
        index_label="Dataset"
    )

    # 7) Actual vs Predicted
    comparison = pd.DataFrame({
        "Actual": actual_test,
        "Predicted": test_pred
    })

    comparison["Error"] = (
        comparison["Actual"] - comparison["Predicted"]
    )

    comparison.to_csv(
        RESULTS_DIR / "actual_vs_predicted.csv",
        index=False
    )

    print("\n========== Actual vs Predicted (First 10) ==========\n")

    print(
        comparison.head(10).to_string(
            index=False,
            float_format="{:,.0f}".format
        )
    )

    print("\nEvaluation completed successfully!")

    print("Results saved in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
