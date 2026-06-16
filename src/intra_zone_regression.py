"""Regression intra-zone : prediction des coordonnees (x, y) dans une zone.

Conforme au PDF IF23 P (page 5) : entree = vecteur RSSI par BSSID,
sortie = coordonnees (x, y) dans le repere de la zone.

Le dataset attendu (`data/regression/dataset_regression.csv`) a la forme :
    X, Y, <BSSID_1>, <BSSID_2>, ..., <BSSID_N>
avec une ligne par point de mesure (~30+ points), un RSSI moyen par BSSID,
et 0 pour les BSSID non captes a ce point.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              ExtraTreesRegressor)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
MISSING_VALUE = 0.0  # PDF: "0, -100... pour les BSSIDs inexistants" — le dataset utilise 0


def normalize_bssid(value: str) -> str:
    return str(value).strip().lower().rstrip(":")


def load_regression_dataset(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Charge le CSV (X, Y, BSSID_1, ..., BSSID_N).

    Retourne (X_features, y_xy, bssid_columns).
    Les BSSID sont normalises en minuscules sans `:` final.
    """
    df = pd.read_csv(csv_path)
    if "X" not in df.columns or "Y" not in df.columns:
        raise ValueError("Le CSV doit contenir les colonnes 'X' et 'Y'.")

    bssid_cols_raw = [c for c in df.columns if c not in ("X", "Y")]
    rename = {c: normalize_bssid(c) for c in bssid_cols_raw}
    df = df.rename(columns=rename)
    bssid_cols = [rename[c] for c in bssid_cols_raw]

    X = df[bssid_cols].astype(float).fillna(MISSING_VALUE)
    y = df[["X", "Y"]].astype(float).to_numpy()
    return X, y, bssid_cols


def build_model_zoo(rs: int = RANDOM_STATE) -> Dict[str, Pipeline]:
    """Zoo de modeles de regression multi-output."""
    return {
        "LinearRegression": Pipeline([
            ("sc", StandardScaler()),
            ("m", LinearRegression()),
        ]),
        "Ridge": Pipeline([
            ("sc", StandardScaler()),
            ("m", Ridge(alpha=1.0, random_state=rs)),
        ]),
        "RandomForest": Pipeline([
            ("m", RandomForestRegressor(
                n_estimators=300, max_depth=None, random_state=rs, n_jobs=-1)),
        ]),
        "ExtraTrees": Pipeline([
            ("m", ExtraTreesRegressor(
                n_estimators=400, max_depth=None, random_state=rs, n_jobs=-1)),
        ]),
        "GradientBoosting": Pipeline([
            ("m", MultiOutputRegressor(GradientBoostingRegressor(
                n_estimators=200, learning_rate=0.05, max_depth=4, random_state=rs))),
        ]),
        "KNN_3": Pipeline([
            ("sc", StandardScaler()),
            ("m", KNeighborsRegressor(n_neighbors=3, weights="distance")),
        ]),
        "KNN_5": Pipeline([
            ("sc", StandardScaler()),
            ("m", KNeighborsRegressor(n_neighbors=5, weights="distance")),
        ]),
        "MLP_64_32": Pipeline([
            ("sc", StandardScaler()),
            ("m", MLPRegressor(
                hidden_layer_sizes=(64, 32), alpha=8e-4,
                learning_rate_init=8e-4, batch_size=8,
                early_stopping=True, max_iter=2000, random_state=rs)),
        ]),
    }


def evaluate_models(
    X: pd.DataFrame,
    y: np.ndarray,
    models: Dict[str, Pipeline],
    test_size: float = 0.25,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, object], Dict[str, np.ndarray]]:
    """Holdout split + entrainement + metrics.

    Retourne (metrics_df, fitted_models, splits_dict).
    Metrics : MAE_x, MAE_y, MAE_euclid (m), RMSE_euclid, R2_x, R2_y.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)

    rows = []
    fitted: Dict[str, object] = {}
    for name, model in models.items():
        f = clone(model).fit(X_train, y_train)
        pred = f.predict(X_test)
        err = pred - y_test
        eucl = np.sqrt((err ** 2).sum(axis=1))
        rows.append({
            "model": name,
            "MAE_x": float(mean_absolute_error(y_test[:, 0], pred[:, 0])),
            "MAE_y": float(mean_absolute_error(y_test[:, 1], pred[:, 1])),
            "MAE_euclid_m": float(np.mean(eucl)),
            "median_euclid_m": float(np.median(eucl)),
            "RMSE_euclid_m": float(np.sqrt(mean_squared_error(y_test, pred))),
            "R2_x": float(r2_score(y_test[:, 0], pred[:, 0])),
            "R2_y": float(r2_score(y_test[:, 1], pred[:, 1])),
        })
        fitted[name] = f

    metrics = pd.DataFrame(rows).sort_values("MAE_euclid_m").reset_index(drop=True)
    splits = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    return metrics, fitted, splits


def select_top_bssids(
    X: pd.DataFrame,
    y: np.ndarray,
    k: int,
    random_state: int = RANDOM_STATE,
) -> List[str]:
    """Selectionne les k BSSID les plus informatifs via feature_importance d'un RF."""
    rf = RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1)
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=X.columns)
    return importances.sort_values(ascending=False).head(k).index.tolist()


@dataclass
class IntraZoneRegressor:
    """Wrapper qui maintient l'ordre des BSSID et offre la prediction depuis un scan dict."""
    model: object
    bssid_columns: List[str]
    bounds: Dict[str, Tuple[float, float]]
    missing_value: float = MISSING_VALUE

    def vectorize_scan(self, scan: Mapping[str, float]) -> np.ndarray:
        normalized = {normalize_bssid(b): float(r) for b, r in scan.items()}
        vec = np.full((1, len(self.bssid_columns)), self.missing_value, dtype=float)
        for i, b in enumerate(self.bssid_columns):
            if b in normalized:
                vec[0, i] = normalized[b]
        return vec

    def predict(self, scan: Mapping[str, float]) -> Tuple[float, float]:
        X = self.vectorize_scan(scan)
        pred = self.model.predict(X)[0]
        return float(pred[0]), float(pred[1])

    def visible_known_bssids(self, scan: Mapping[str, float]) -> int:
        normalized = {normalize_bssid(b) for b in scan.keys()}
        return sum(1 for b in self.bssid_columns if b in normalized)


def save_regression_artifacts(
    artifact_dir: Path,
    regressor: IntraZoneRegressor,
    metrics: pd.DataFrame,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, artifact_dir / "regressor.pkl")
    metrics.to_csv(artifact_dir / "metrics.csv", index=False)
    metadata = {
        "type": "IntraZoneRegressor",
        "n_bssid": len(regressor.bssid_columns),
        "bounds_x": list(regressor.bounds.get("x", (None, None))),
        "bounds_y": list(regressor.bounds.get("y", (None, None))),
        "missing_value": regressor.missing_value,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_regression_artifacts(artifact_dir: Path) -> IntraZoneRegressor:
    return joblib.load(Path(artifact_dir) / "regressor.pkl")
