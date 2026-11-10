import json
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def main():
    # 1. Load & Merge Data
    path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    movies = pd.read_csv(f"{path}/tmdb_5000_movies.csv")
    credits = pd.read_csv(f"{path}/tmdb_5000_credits.csv").rename(columns={"movie_id": "id"})
    df = movies.merge(credits[["id", "cast", "crew"]], on="id")

    # 2. Extract Features & Clean
    for col in ["cast", "crew", "genres"]:
        df[f"{col}_size" if col != "genres" else "genres_count"] = df[col].apply(
            lambda x: len(json.loads(x)) if pd.notna(x) else 0
        )

    features = ["budget", "popularity", "runtime", "vote_count", "vote_average", "cast_size", "crew_size", "genres_count"]
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())
    clean_df = df[(df["revenue"] > 1e4) & (df["budget"] > 1e4)].reset_index(drop=True)

    X = clean_df[features]
    y = clean_df["revenue"]

    # 3. Train-Test Split & Scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Model Training & Evaluation
    model = LinearRegression().fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    print("--- Test Evaluation ---")
    print(f"MAE:  ${mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE: ${np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"R²:   {r2_score(y_test, y_pred):.4f}")

    # 5. Plot Results
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(clean_df[features + ["revenue"]].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
    axes[0].set_title("Correlation Heatmap")

    axes[1].scatter(y_test / 1e6, y_pred / 1e6, alpha=0.4, color="darkgreen")
    axes[1].plot([y_test.min() / 1e6, y_test.max() / 1e6], [y_test.min() / 1e6, y_test.max() / 1e6], "r--")
    axes[1].set(xlabel="Actual ($M)", ylabel="Predicted ($M)", title="Actual vs. Predicted Revenue")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()