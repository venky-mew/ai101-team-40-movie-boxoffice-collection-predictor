import json
import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def parse_json_count(json_str: str) -> int:
    """Safely extracts the element count from a JSON string representation of a list."""
    try:
        return len(json.loads(json_str))
    except (json.JSONDecodeError, TypeError):
        return 0


def load_and_preprocess_tmdb_data(
    num_features: list[str],
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Downloads TMDB movie dataset, merges credits, engineers JSON count features,

    filters invalid rows, splits data, and returns scaled features along with evaluation correlation matrix.
    """
    # 1. Download & Load Datasets
    dataset_path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")
    movies_df = pd.read_csv(f"{dataset_path}/tmdb_5000_movies.csv")
    credits_df = pd.read_csv(f"{dataset_path}/tmdb_5000_credits.csv")

    # 2. Merge Datasets
    credits_df.rename(columns={"movie_id": "id"}, inplace=True)
    df = movies_df.merge(credits_df[["id", "cast", "crew"]], on="id")

    # 3. Feature Engineering (JSON parsing)
    df["cast_size"] = df["cast"].apply(parse_json_count)
    df["crew_size"] = df["crew"].apply(parse_json_count)
    df["genres_count"] = df["genres"].apply(parse_json_count)

    # 4. Extract target dataset & clean missing/invalid records
    analysis_df = df[num_features + [target]].copy()
    analysis_df["runtime"] = analysis_df["runtime"].fillna(
        analysis_df["runtime"].median()
    )

    # Filter unreleased or non-revenue movies
    clean_df = (
        analysis_df[
            (analysis_df["revenue"] > 10000) & (analysis_df["budget"] > 10000)
        ]
        .copy()
        .reset_index(drop=True)
    )

    corr_matrix = clean_df.corr()

    # 5. Features & Target Split
    X = clean_df[num_features]
    y = clean_df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # 6. Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        corr_matrix,
    )