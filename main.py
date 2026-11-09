from src.data_preprocessing import (
    NUM_FEATURES,
    load_dataset,
    prepare_data,
    print_initial_analysis,
)
from src.model import train_model, get_coefficients
from src.evaluation import print_metrics, plot_results


def main():
    print("Loading TMDB movie metadata...")
    df = load_dataset()
    print(f"Merged dataset shape: {df.shape}")

    correlations = print_initial_analysis(df)

    X_train, X_test, y_train, y_test, scaler, clean_df = prepare_data(df)
    print(f"Dataset after preprocessing: {clean_df.shape}")

    model = train_model(X_train, y_train)

    predictions, metrics = print_metrics(
        model, X_train, X_test, y_train, y_test
    )