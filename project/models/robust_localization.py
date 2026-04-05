from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

RANDOM_STATE = 42
SUMMARY_COLUMNS = [
    "visible_ssid_count",
    "rssi_max_overall",
    "rssi_mean_overall",
    "rssi_min_overall",
    "rssi_std_overall",
    "rssi_median_overall",
    "rssi_p25_overall",
    "rssi_p75_overall",
    "strongest_gap",
]


def normalize_bssid(value: str) -> str:
    return str(value).strip().lower().rstrip(":")


def normalize_ssid(value: str) -> str:
    ssid = str(value).strip().lower()
    return ssid if ssid else "<hidden>"


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)


def _colmap(df: pd.DataFrame) -> Dict[str, str]:
    out = {}
    for col in df.columns:
        key = col.lower().strip().replace(" ", "")
        out[key] = col
    return out


def _second_strongest(values: pd.Series) -> float:
    arr = np.sort(values.to_numpy(dtype=float))
    if arr.size == 0:
        return -100.0
    if arr.size == 1:
        return float(arr[-1])
    return float(arr[-2])


def _normalize_wifi_dataframe(df: pd.DataFrame, default_room: str = "unknown") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["room", "time", "bssid", "ssid", "rssi"])

    cmap = _colmap(df)
    bssid_col = cmap.get("bssid")
    ssid_col = cmap.get("ssid")
    rssi_col = cmap.get("rssi(dbm)") or cmap.get("rssi") or cmap.get("signal")
    time_col = cmap.get("time")

    if rssi_col is None:
        raise ValueError("Le dataset doit contenir RSSI(dBm) ou RSSI/signal.")

    out = pd.DataFrame()

    if bssid_col is not None:
        out["bssid"] = df[bssid_col].astype(str).map(normalize_bssid)
    else:
        out["bssid"] = ""

    if ssid_col is not None:
        out["ssid"] = df[ssid_col].astype(str).map(normalize_ssid)
    elif bssid_col is not None:
        # fallback degrade: on conserve un identifiant reseau meme sans SSID
        out["ssid"] = df[bssid_col].astype(str).map(normalize_bssid)
    else:
        raise ValueError("Le dataset doit contenir au minimum SSID (ou BSSID en fallback).")

    out["rssi"] = pd.to_numeric(df[rssi_col], errors="coerce")

    if time_col is None:
        out["time"] = pd.Timestamp.utcnow().floor("s")
    else:
        out["time"] = pd.to_datetime(df[time_col], errors="coerce")

    if "Room" in df.columns:
        out["room"] = df["Room"].astype(str)
    elif "room" in df.columns:
        out["room"] = df["room"].astype(str)
    else:
        out["room"] = default_room

    out = out.dropna(subset=["ssid", "rssi", "time"]).copy()
    out = out[out["ssid"].str.len() > 0].copy()
    out = out[(out["rssi"] >= -120) & (out["rssi"] <= -1)].copy()
    return out


def load_raw_wifi_data(raw_dir: Path) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    files = sorted(raw_dir.glob("wifi_*.csv"))
    if not files:
        raise FileNotFoundError(f"Aucun CSV wifi_*.csv trouve dans {raw_dir}")

    all_rows = []
    for csv_path in files:
        room = csv_path.stem.replace("wifi_", "")
        df = pd.read_csv(csv_path)
        norm = _normalize_wifi_dataframe(df, default_room=room)
        norm["room"] = room
        all_rows.append(norm)

    return pd.concat(all_rows, ignore_index=True)


def build_snapshot_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise ValueError("Dataset vide apres nettoyage.")

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["room", "time", "ssid", "rssi"]).copy()

    per_ssid = (
        df.groupby(["room", "time", "ssid"], as_index=False)
        .agg(
            rssi_mean=("rssi", "mean"),
            rssi_max=("rssi", "max"),
            rssi_std=("rssi", "std"),
            rssi_count=("rssi", "size"),
        )
        .fillna({"rssi_std": 0.0})
    )

    snapshot = (
        per_ssid.groupby(["room", "time"], as_index=False)
        .agg(
            visible_ssid_count=("ssid", "size"),
            rssi_max_overall=("rssi_mean", "max"),
            rssi_mean_overall=("rssi_mean", "mean"),
            rssi_min_overall=("rssi_mean", "min"),
            rssi_std_overall=("rssi_mean", "std"),
            rssi_median_overall=("rssi_mean", "median"),
            rssi_p25_overall=("rssi_mean", lambda x: float(np.percentile(x, 25))),
            rssi_p75_overall=("rssi_mean", lambda x: float(np.percentile(x, 75))),
            rssi_second_overall=("rssi_mean", _second_strongest),
        )
        .fillna({"rssi_std_overall": 0.0})
    )
    snapshot["strongest_gap"] = snapshot["rssi_max_overall"] - snapshot["rssi_second_overall"]
    snapshot = snapshot.drop(columns=["rssi_second_overall"])

    return per_ssid, snapshot


@dataclass
class RobustFeatureBuilder:
    max_ssids: int = 120
    min_ssid_frequency: int = 5
    fill_rssi: float = -100.0
    selected_ssids: List[str] | None = None
    feature_columns: List[str] | None = None

    def _ensure_compatible_state(self) -> None:
        # Compatibility with previously pickled builders (BSSID-based versions).
        if self.selected_ssids is None and hasattr(self, "selected_bssids"):
            legacy = getattr(self, "selected_bssids")
            if legacy is not None:
                self.selected_ssids = list(legacy)

        if self.selected_ssids is None and self.feature_columns is not None:
            inferred = []
            prefixes = ("presence__", "rssi_mean__", "rssi_max__", "rssi_std__", "rssi_count__")
            for col in self.feature_columns:
                for pref in prefixes:
                    if col.startswith(pref):
                        name = col[len(pref) :]
                        if name and name not in inferred:
                            inferred.append(name)
                        break
            if inferred:
                self.selected_ssids = inferred

        if self.feature_columns is None and self.selected_ssids is not None:
            cols = list(SUMMARY_COLUMNS)
            for stat in ["rssi_mean", "rssi_max", "rssi_std", "rssi_count", "presence"]:
                cols.extend([f"{stat}__{s}" for s in self.selected_ssids])
            self.feature_columns = cols

    def fit(self, per_ssid: pd.DataFrame) -> "RobustFeatureBuilder":
        freq = per_ssid["ssid"].value_counts()
        freq = freq[freq >= self.min_ssid_frequency]
        if self.max_ssids is not None:
            freq = freq.head(self.max_ssids)
        self.selected_ssids = freq.index.tolist()
        return self

    def _build_features(self, per_ssid: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
        self._ensure_compatible_state()
        if self.selected_ssids is None:
            raise RuntimeError("FeatureBuilder non entraine. Appelez fit() d'abord.")

        index_cols = ["room", "time"]
        base = snapshot[index_cols + SUMMARY_COLUMNS].set_index(index_cols).sort_index()

        filt = per_ssid[per_ssid["ssid"].isin(self.selected_ssids)].copy()

        blocks = []
        for stat in ["rssi_mean", "rssi_max", "rssi_std", "rssi_count"]:
            piv = filt.pivot_table(index=index_cols, columns="ssid", values=stat, aggfunc="mean")
            piv = piv.reindex(columns=self.selected_ssids)
            if stat in {"rssi_mean", "rssi_max"}:
                piv = piv.fillna(self.fill_rssi)
            else:
                piv = piv.fillna(0.0)
            piv.columns = [f"{stat}__{ssid}" for ssid in piv.columns]
            blocks.append(piv)

        presence = filt.assign(present=1).pivot_table(
            index=index_cols, columns="ssid", values="present", aggfunc="max"
        )
        presence = presence.reindex(columns=self.selected_ssids).fillna(0.0)
        presence.columns = [f"presence__{ssid}" for ssid in presence.columns]
        blocks.append(presence)

        X = pd.concat([base] + blocks, axis=1).fillna(0.0)

        if self.feature_columns is None:
            self.feature_columns = X.columns.tolist()
        else:
            X = X.reindex(columns=self.feature_columns, fill_value=0.0)

        return X

    def fit_transform(self, per_ssid: pd.DataFrame, snapshot: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        self.fit(per_ssid)
        X = self._build_features(per_ssid, snapshot)
        y = snapshot.set_index(["room", "time"]).loc[X.index].index.get_level_values("room")
        return X, pd.Series(y, index=X.index, name="room")

    def transform(self, per_ssid: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
        return self._build_features(per_ssid, snapshot)

    def vectorize_scan_dict(self, scan_dict: Mapping[str, float]) -> pd.DataFrame:
        self._ensure_compatible_state()
        if self.selected_ssids is None or self.feature_columns is None:
            raise RuntimeError("FeatureBuilder non entraine.")

        normalized = {normalize_ssid(k): float(v) for k, v in scan_dict.items() if str(k).strip()}
        values = np.array(list(normalized.values()), dtype=float)
        if values.size == 0:
            values = np.array([self.fill_rssi], dtype=float)

        sorted_values = np.sort(values)
        strongest = float(sorted_values[-1])
        second = float(sorted_values[-2]) if sorted_values.size > 1 else strongest

        row = {
            "visible_ssid_count": float(len(normalized)),
            "rssi_max_overall": strongest,
            "rssi_mean_overall": float(np.mean(values)),
            "rssi_min_overall": float(np.min(values)),
            "rssi_std_overall": float(np.std(values)),
            "rssi_median_overall": float(np.median(values)),
            "rssi_p25_overall": float(np.percentile(values, 25)),
            "rssi_p75_overall": float(np.percentile(values, 75)),
            "strongest_gap": strongest - second,
        }

        for ssid in self.selected_ssids:
            rssi = normalized.get(ssid)
            if rssi is None:
                row[f"rssi_mean__{ssid}"] = self.fill_rssi
                row[f"rssi_max__{ssid}"] = self.fill_rssi
                row[f"rssi_std__{ssid}"] = 0.0
                row[f"rssi_count__{ssid}"] = 0.0
                row[f"presence__{ssid}"] = 0.0
            else:
                row[f"rssi_mean__{ssid}"] = rssi
                row[f"rssi_max__{ssid}"] = rssi
                row[f"rssi_std__{ssid}"] = 0.0
                row[f"rssi_count__{ssid}"] = 1.0
                row[f"presence__{ssid}"] = 1.0

        X = pd.DataFrame([row])
        X = X.reindex(columns=self.feature_columns, fill_value=0.0)
        return X


def get_model_zoo(
    random_state: int = RANDOM_STATE,
    include_slow: bool = False,
    include_neural: bool = True,
) -> Dict[str, Pipeline]:
    models: Dict[str, Pipeline] = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        ),
        "RandomForest": Pipeline(
            steps=[
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=120,
                        max_depth=20,
                        random_state=random_state,
                        class_weight="balanced_subsample",
                        n_jobs=1,
                    ),
                )
            ]
        ),
        "ExtraTrees": Pipeline(
            steps=[
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=180,
                        max_depth=24,
                        random_state=random_state,
                        class_weight="balanced",
                        n_jobs=1,
                    ),
                )
            ]
        ),
    }

    if include_neural:
        models["MLP_32"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(32,),
                        activation="relu",
                        alpha=1e-3,
                        batch_size=64,
                        learning_rate_init=1e-3,
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=400,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        models["MLP_64_32"] = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=8e-4,
                        batch_size=64,
                        learning_rate_init=8e-4,
                        early_stopping=True,
                        validation_fraction=0.15,
                        max_iter=450,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    if include_slow:
        models["HistGradientBoosting"] = Pipeline(
            steps=[
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_iter=150,
                        max_depth=8,
                        random_state=random_state,
                    ),
                )
            ]
        )

    return models


def evaluate_models_cv(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    models: Dict[str, Pipeline],
    n_splits: int = 5,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []

    for name, model in models.items():
        acc_scores = []
        f1_scores = []

        for train_idx, valid_idx in skf.split(X, y_encoded):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y_encoded[train_idx], y_encoded[valid_idx]

            fitted = clone(model).fit(X_train, y_train)
            pred = fitted.predict(X_valid)
            acc_scores.append(accuracy_score(y_valid, pred))
            f1_scores.append(f1_score(y_valid, pred, average="macro"))

        rows.append(
            {
                "model": name,
                "cv_accuracy_mean": float(np.mean(acc_scores)),
                "cv_accuracy_std": float(np.std(acc_scores)),
                "cv_f1_macro_mean": float(np.mean(f1_scores)),
                "cv_f1_macro_std": float(np.std(f1_scores)),
            }
        )

    return pd.DataFrame(rows).sort_values(by="cv_f1_macro_mean", ascending=False).reset_index(drop=True)


def evaluate_models_holdout(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    models: Dict[str, Pipeline],
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, random_state=random_state
    )

    rows = []
    fitted_models: Dict[str, object] = {}

    for name, model in models.items():
        fitted = clone(model).fit(X_train, y_train)
        pred = fitted.predict(X_test)
        rows.append(
            {
                "model": name,
                "test_accuracy": float(accuracy_score(y_test, pred)),
                "test_f1_macro": float(f1_score(y_test, pred, average="macro")),
            }
        )
        fitted_models[name] = fitted

    metrics = pd.DataFrame(rows).sort_values(by="test_f1_macro", ascending=False).reset_index(drop=True)

    best_name = metrics.loc[0, "model"]
    diagnostics = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "best_model_name": best_name,
        "best_model_fitted": fitted_models[best_name],
        "all_fitted_models": fitted_models,
    }
    return metrics, diagnostics


def train_final_model(
    X: pd.DataFrame,
    y_encoded: np.ndarray,
    model_name: str,
    random_state: int = RANDOM_STATE,
    include_neural: bool = True,
):
    models = get_model_zoo(random_state=random_state, include_neural=include_neural)
    if model_name not in models:
        raise ValueError(f"Modele inconnu: {model_name}")
    return clone(models[model_name]).fit(X, y_encoded)


def save_artifacts(
    artifact_dir: Path,
    model,
    feature_builder: RobustFeatureBuilder,
    label_encoder: LabelEncoder,
    metadata: Dict[str, object],
) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, artifact_dir / "room_classifier_robust.pkl")
    joblib.dump(feature_builder, artifact_dir / "feature_builder.pkl")
    joblib.dump(label_encoder, artifact_dir / "label_encoder.pkl")

    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_artifacts(artifact_dir: Path) -> Dict[str, object]:
    artifact_dir = Path(artifact_dir)
    model = joblib.load(artifact_dir / "room_classifier_robust.pkl")
    feature_builder = joblib.load(artifact_dir / "feature_builder.pkl")
    label_encoder = joblib.load(artifact_dir / "label_encoder.pkl")
    if hasattr(feature_builder, "_ensure_compatible_state"):
        feature_builder._ensure_compatible_state()

    metadata_path = artifact_dir / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "model": model,
        "feature_builder": feature_builder,
        "label_encoder": label_encoder,
        "metadata": metadata,
    }


def predict_scan(
    scan_dict: Mapping[str, float],
    model,
    feature_builder: RobustFeatureBuilder,
    label_encoder: LabelEncoder,
    top_k: int = 3,
) -> Dict[str, object]:
    X = feature_builder.vectorize_scan_dict(scan_dict)

    pred_encoded = model.predict(X)[0]
    pred_room = label_encoder.inverse_transform([pred_encoded])[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        proba = _softmax(scores)[0]
    else:
        proba = np.zeros(len(label_encoder.classes_), dtype=float)
        proba[pred_encoded] = 1.0

    idx_sorted = np.argsort(proba)[::-1][:top_k]
    top_predictions = [
        {
            "room": str(label_encoder.classes_[i]),
            "probability": float(proba[i]),
        }
        for i in idx_sorted
    ]

    return {
        "predicted_room": str(pred_room),
        "confidence": float(np.max(proba)),
        "top_predictions": top_predictions,
    }


def predict_from_csv(
    csv_path: Path,
    model,
    feature_builder: RobustFeatureBuilder,
    label_encoder: LabelEncoder,
) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    norm = _normalize_wifi_dataframe(raw, default_room="unknown")
    per_ssid, snapshot = build_snapshot_tables(norm)
    X = feature_builder.transform(per_ssid, snapshot)

    pred_encoded = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        conf = np.max(proba, axis=1)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        proba = _softmax(scores)
        conf = np.max(proba, axis=1)
    else:
        conf = np.ones(len(pred_encoded))

    out = snapshot[["time"]].copy()
    out["predicted_room"] = label_encoder.inverse_transform(pred_encoded)
    out["confidence"] = conf
    return out.sort_values("time").reset_index(drop=True)


def predict_with_details_from_csv(
    csv_path: Path,
    model,
    feature_builder: RobustFeatureBuilder,
    label_encoder: LabelEncoder,
    expected_room: str | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    return predict_with_details_from_dataframe(
        raw_df=raw,
        model=model,
        feature_builder=feature_builder,
        label_encoder=label_encoder,
        expected_room=expected_room,
        top_k=top_k,
    )


def predict_with_details_from_dataframe(
    raw_df: pd.DataFrame,
    model,
    feature_builder: RobustFeatureBuilder,
    label_encoder: LabelEncoder,
    expected_room: str | None = None,
    top_k: int = 3,
) -> pd.DataFrame:
    norm = _normalize_wifi_dataframe(raw_df, default_room="unknown")
    per_ssid, snapshot = build_snapshot_tables(norm)
    X = feature_builder.transform(per_ssid, snapshot)

    pred_encoded = model.predict(X)
    classes = np.array(label_encoder.classes_)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        if scores.ndim == 1:
            scores = np.vstack([-scores, scores]).T
        proba = _softmax(scores)
    else:
        proba = np.zeros((len(pred_encoded), len(classes)), dtype=float)
        proba[np.arange(len(pred_encoded)), pred_encoded] = 1.0

    out = snapshot[["room", "time"]].copy()
    out = out.rename(columns={"room": "true_room"})
    out["predicted_room"] = label_encoder.inverse_transform(pred_encoded)
    out["confidence"] = np.max(proba, axis=1)

    if expected_room is not None:
        out["true_room"] = str(expected_room)

    top_k = max(1, min(int(top_k), proba.shape[1]))
    idx_sorted = np.argsort(proba, axis=1)[:, ::-1]
    for rank in range(top_k):
        idx = idx_sorted[:, rank]
        out[f"top{rank + 1}_room"] = classes[idx]
        out[f"top{rank + 1}_proba"] = proba[np.arange(len(out)), idx]

    out["is_correct"] = (out["predicted_room"] == out["true_room"]).astype(int)
    top_rooms = [f"top{i + 1}_room" for i in range(top_k)]
    out["topk_hit"] = out.apply(lambda row: int(row["true_room"] in {row[c] for c in top_rooms}), axis=1)

    return out.sort_values("time").reset_index(drop=True)


def build_training_dataset(
    raw_dir: Path,
    max_ssids: int = 120,
    min_ssid_frequency: int = 5,
) -> Tuple[pd.DataFrame, pd.Series, RobustFeatureBuilder, pd.DataFrame, pd.DataFrame]:
    raw_df = load_raw_wifi_data(raw_dir)
    per_ssid, snapshot = build_snapshot_tables(raw_df)

    builder = RobustFeatureBuilder(max_ssids=max_ssids, min_ssid_frequency=min_ssid_frequency)
    X, y = builder.fit_transform(per_ssid, snapshot)

    return X, y, builder, per_ssid, snapshot
