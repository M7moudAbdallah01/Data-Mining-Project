import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

DATA_PATH = DATA_DIR / "train_preprocessed.csv"

RANDOM_STATE = 42

DROP_COLS = ["Price", "Price_log", "sample_weight"]


def get_models():
    return {
        "Linear Regression": LinearRegression(),

        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.5,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
    }


def main():

    # Load preprocessed training data
    train = pd.read_csv(DATA_PATH)

    X = train.drop(columns=DROP_COLS)
    y = train["Price_log"]
    weights = train["sample_weight"]

    print("Training data:", X.shape)

    # Split training data into fitting and validation sets
    X_fit, X_val, y_fit, y_val, w_fit, _ = train_test_split(
        X,
        y,
        weights,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    results = []

    # Compare models with and without sample weights
    for use_weights in [False, True]:

        for name, model in get_models().items():

            model.fit(
                X_fit,
                y_fit,
                sample_weight=w_fit if use_weights else None
            )

            actual = np.expm1(y_val)
            predicted = np.expm1(model.predict(X_val))

            results.append({
                "Model": name,
                "Weights": use_weights,
                "Val_RMSE": np.sqrt(
                    mean_squared_error(actual, predicted)
                ),
                "Val_R2": r2_score(actual, predicted)
            })

    # Display comparison results
    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "Val_RMSE"
    ).reset_index(drop=True)

    print("\nModel Comparison:")
    print(results_df.to_string(index=False))

    # Save comparison results
    results_df.to_csv(
        BASE_DIR / "validation_results.csv",
        index=False
    )

    # Select the best model
    best = results_df.iloc[0]

    print("\nBest Model:", best["Model"])
    print("Using Sample Weights:", bool(best["Weights"]))
    print("Validation RMSE:", best["Val_RMSE"])
    print("Validation R2:", best["Val_R2"])

    print("\nModel comparison completed successfully!")


if __name__ == "__main__":
    main()
