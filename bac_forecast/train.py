"""Model training module."""

import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


def _build_models():
    return {
        "DecisionTree": DecisionTreeRegressor(max_depth=4, random_state=42),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsRegressor(n_neighbors=5)),
        ]),
        "SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("svr", SVR()),
        ]),
    }


def train_models(X_train, y_train):
    models = _build_models()
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def predict(fitted_models, X):
    preds = {}
    for name, model in fitted_models.items():
        preds[name] = model.predict(X)
    return pd.DataFrame(preds, index=X.index)