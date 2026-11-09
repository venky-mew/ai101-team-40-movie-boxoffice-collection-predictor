import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def print_metrics(model, X_train, X_test, y_train, y_test):
    """Evaluate the model and print its performance metrics."""

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2,
    }

    print("\nModel Evaluation:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return predictions, metrics


def plot_results(correlations, y_test, predictions):
    """Plot feature correlations and actual vs predicted ratings."""

    plt.figure(figsize=(10, 7))
    sns.heatmap(
        correlations,
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, alpha=0.5)

    # Reference line for perfect predictions.
    lower = min(y_test.min(), predictions.min())
    upper = max(y_test.max(), predictions.max())
    plt.plot([lower, upper], [lower, upper], "r--")

    plt.xlabel("Actual Movie Rating")
    plt.ylabel("Predicted Movie Rating")
    plt.title("Actual vs Predicted Movie Ratings")
    plt.tight_layout()
    plt.show()