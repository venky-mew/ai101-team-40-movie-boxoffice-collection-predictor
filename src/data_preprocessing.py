import json
import pandas as pd

NUM_FEATURES = [
    "budget",
    "popularity",
    "runtime",
    "vote_count",
    "cast_size",
    "crew_size",
    "genre_count",
]


def load_dataset():
    """Load and merge the TMDB movies and credits datasets."""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError(
            "Install required packages using: "
            "pip install kagglehub pandas numpy scikit-learn matplotlib seaborn"
        ) from exc

    dataset_path = kagglehub.dataset_download(
        "tmdb/tmdb-movie-metadata"
    )

    movies = pd.read_csv(
        f"{dataset_path}/tmdb_5000_movies.csv"
    )
    credits = pd.read_csv(
        f"{dataset_path}/tmdb_5000_credits.csv"
    )

    return movies.merge(
        credits,
        left_on="id",
        right_on="movie_id",
        how="inner",
        suffixes=("", "_credits"),
    )


def print_initial_analysis(df):
    """Print basic statistics and return feature correlations."""
    print("\nDataset information:")
    print(df.info())

    print("\nDescriptive statistics:")
    print(df.describe().to_string())

    for column in ["cast", "crew", "genres"]:
        if column in df.columns:
            df[f"{column}_size"] = df[column].apply(
                lambda value: len(json.loads(value))
                if isinstance(value, str)
                else 0
            )

    feature_columns = [
        column for column in NUM_FEATURES
        if column in df.columns
    ]

    correlations = df[
        feature_columns + ["vote_average"]
    ].corr(numeric_only=True)

    print("\nCorrelation with vote_average:")
    print(correlations["vote_average"].sort_values(ascending=False))

    return correlations


def prepare_data(df):
    """Clean data, scale features, and split into train/test sets."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    df = df.copy()

    # Fill missing runtimes with the median.
    df["runtime"] = df["runtime"].fillna(df["runtime"].median())

    # Keep movies with at least 15 votes.
    df = df[df["vote_count"] >= 15].copy()

    # Create the features expected by the model.
    df["cast_size"] = df["cast"].apply(
        lambda value: len(json.loads(value))
        if isinstance(value, str) else 0
    )
    df["crew_size"] = df["crew"].apply(
        lambda value: len(json.loads(value))
        if isinstance(value, str) else 0
    )
    df["genre_count"] = df["genres"].apply(
        lambda value: len(json.loads(value))
        if isinstance(value, str) else 0
    )

    X = df[NUM_FEATURES].fillna(0)
    y = df["vote_average"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler, df