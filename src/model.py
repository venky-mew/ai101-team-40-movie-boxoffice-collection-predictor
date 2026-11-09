from sklearn.linear_model import LinearRegression
import pandas as pd


def train_model(X_train, y_train):
    """Train the multiple linear regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def get_coefficients(model, feature_names):
    """Return feature names and their model coefficients."""
    return pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_,
    }).sort_values(
        by="Coefficient",
        key=abs,
        ascending=False,
    )