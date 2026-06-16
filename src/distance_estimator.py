"""Estimation de la distance a un acces point WiFi.

Approche principale : modele physique log-distance path loss.
    PL(d) = PL(d0) + 10 * n * log10(d / d0)
    => RSSI(d)   = RSSI(d0) - 10 * n * log10(d / d0)
    => d         = d0 * 10**((RSSI(d0) - RSSI(d)) / (10 * n))

Avec :
    - d0          = distance de reference (1 m par defaut)
    - RSSI(d0)    = RSSI mesure a la distance d0 (donnees `RSSI0.xlsx`)
    - n           = path loss exponent (2.0 espace libre, 3.0 interieur typique, 4.0 dense)

Calibration : si on dispose de paires (RSSI mesure, distance vraie) pour un BSSID,
on peut estimer n optimal en moindres carres.

Le module fournit aussi un cadre `train_supervised_regressor` pour entrainer un
RF / GradientBoosting / LinearRegression supervise quand l'utilisateur fournit un
dataset (rssi, distance) labellise.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import joblib
import numpy as np
import pandas as pd

DEFAULT_N = 3.0
DEFAULT_D0 = 1.0


def normalize_bssid(value: str) -> str:
    return str(value).strip().lower().rstrip(":")


@dataclass
class LogDistanceEstimator:
    """Estimateur de distance base sur le modele log-distance path loss."""
    rssi_ref: Dict[str, float] = field(default_factory=dict)
    ssid_by_bssid: Dict[str, str] = field(default_factory=dict)
    n: float = DEFAULT_N
    d0: float = DEFAULT_D0
    fallback_rssi_ref: float = -45.0

    @classmethod
    def from_reference_dataframe(
        cls,
        df: pd.DataFrame,
        n: float = DEFAULT_N,
        d0: float = DEFAULT_D0,
    ) -> "LogDistanceEstimator":
        """Construit un estimateur a partir d'un DataFrame de scan de reference.

        Le DataFrame doit contenir au moins les colonnes `BSSID` et `RSSI`
        (ou `RSSI(dBm)`). Si plusieurs lignes existent pour un meme BSSID, on
        prend le RSSI median (robuste aux outliers).
        """
        cmap = {c.lower().replace(" ", ""): c for c in df.columns}
        bssid_col = cmap["bssid"]
        rssi_col = cmap.get("rssi(dbm)") or cmap.get("rssi")
        if rssi_col is None:
            raise ValueError("Colonne RSSI introuvable dans le DataFrame de reference.")
        ssid_col = cmap.get("ssid")

        ref = df[[bssid_col, rssi_col]].copy()
        ref[bssid_col] = ref[bssid_col].astype(str).map(normalize_bssid)
        ref[rssi_col] = pd.to_numeric(ref[rssi_col], errors="coerce")
        ref = ref.dropna()

        rssi_ref = ref.groupby(bssid_col)[rssi_col].median().astype(float).to_dict()

        ssid_by_bssid: Dict[str, str] = {}
        if ssid_col is not None:
            tmp = df[[bssid_col, ssid_col]].copy()
            tmp[bssid_col] = tmp[bssid_col].astype(str).map(normalize_bssid)
            tmp[ssid_col] = tmp[ssid_col].astype(str).fillna("")
            tmp = tmp[tmp[ssid_col].str.len() > 0]
            for b, s in tmp.drop_duplicates(bssid_col).itertuples(index=False):
                ssid_by_bssid[b] = s

        return cls(rssi_ref=rssi_ref, ssid_by_bssid=ssid_by_bssid, n=n, d0=d0)

    def known_bssids(self) -> List[str]:
        return sorted(self.rssi_ref.keys())

    def label_for(self, bssid: str) -> str:
        b = normalize_bssid(bssid)
        ssid = self.ssid_by_bssid.get(b)
        return f"{ssid} ({b})" if ssid else b

    def estimate(self, bssid: str, rssi_measured: float) -> float:
        """Distance estimee en metres pour un BSSID et un RSSI mesure."""
        b = normalize_bssid(bssid)
        rssi_ref = self.rssi_ref.get(b, self.fallback_rssi_ref)
        return self._distance_from(rssi_ref, rssi_measured)

    def estimate_with_unknown(self, rssi_measured: float) -> float:
        return self._distance_from(self.fallback_rssi_ref, rssi_measured)

    def _distance_from(self, rssi_ref: float, rssi_measured: float) -> float:
        delta = float(rssi_ref) - float(rssi_measured)
        if self.n <= 0:
            return float("inf")
        return float(self.d0 * (10.0 ** (delta / (10.0 * self.n))))

    def estimate_many(self, scan: Mapping[str, float]) -> List[Dict[str, object]]:
        """Pour chaque (bssid, rssi) du scan, calcule la distance estimee."""
        out: List[Dict[str, object]] = []
        for bssid, rssi in scan.items():
            d = self.estimate(bssid, rssi)
            out.append({
                "bssid": normalize_bssid(bssid),
                "ssid": self.ssid_by_bssid.get(normalize_bssid(bssid), ""),
                "rssi_measured": float(rssi),
                "rssi_ref": float(self.rssi_ref.get(normalize_bssid(bssid), self.fallback_rssi_ref)),
                "distance_m": d,
                "known": normalize_bssid(bssid) in self.rssi_ref,
            })
        out.sort(key=lambda r: r["distance_m"])
        return out

    def calibrate_n(self, samples: List[Dict[str, float]]) -> float:
        """Estime n optimal via moindres carres sur des paires (rssi, distance, bssid_ref).

        Chaque sample est `{bssid: ..., rssi: ..., distance: ...}` (distance en metres).
        On resout n* qui minimise sum((delta_i - 10*n*log10(d_i/d0))**2)
        avec delta_i = rssi_ref[bssid] - rssi_i.
        """
        xs, ys = [], []
        for s in samples:
            b = normalize_bssid(s["bssid"])
            if b not in self.rssi_ref:
                continue
            d = float(s["distance"])
            if d <= 0:
                continue
            delta = self.rssi_ref[b] - float(s["rssi"])
            x = 10.0 * math.log10(d / self.d0)
            xs.append(x)
            ys.append(delta)
        if not xs:
            raise ValueError("Aucun echantillon calibrable (BSSID inconnus).")
        xs_arr = np.asarray(xs)
        ys_arr = np.asarray(ys)
        denom = float(np.dot(xs_arr, xs_arr))
        if denom <= 0:
            raise ValueError("Donnees degenerees (distances toutes egales a d0).")
        n_opt = float(np.dot(xs_arr, ys_arr) / denom)
        if not math.isfinite(n_opt) or n_opt <= 0:
            raise ValueError(f"n calibre invalide: {n_opt}")
        self.n = n_opt
        return n_opt


def save_distance_artifacts(
    artifact_dir: Path,
    estimator: LogDistanceEstimator,
    extra_metadata: Optional[Dict[str, object]] = None,
) -> None:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, artifact_dir / "estimator.pkl")
    metadata = {
        "type": "LogDistanceEstimator",
        "n": estimator.n,
        "d0": estimator.d0,
        "n_bssid_ref": len(estimator.rssi_ref),
        "fallback_rssi_ref": estimator.fallback_rssi_ref,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    with (artifact_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_distance_artifacts(artifact_dir: Path) -> LogDistanceEstimator:
    artifact_dir = Path(artifact_dir)
    estimator: LogDistanceEstimator = joblib.load(artifact_dir / "estimator.pkl")
    return estimator


def train_supervised_regressor(
    df: pd.DataFrame,
    rssi_col: str = "rssi",
    distance_col: str = "distance",
    bssid_col: Optional[str] = "bssid",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict[str, object]:
    """Entrainement de modeles ML supervises pour predire la distance.

    Retourne un dict avec les metriques par modele (MAE, RMSE, R²) et le best.
    Utilise comme features : RSSI (toujours), one-hot du BSSID si fourni.
    """
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split as tts

    df = df.dropna(subset=[rssi_col, distance_col]).copy()
    y = pd.to_numeric(df[distance_col], errors="coerce").astype(float).values
    feature_frames = [pd.DataFrame({"rssi": pd.to_numeric(df[rssi_col], errors="coerce")})]
    if bssid_col and bssid_col in df.columns:
        feature_frames.append(pd.get_dummies(df[bssid_col].astype(str).map(normalize_bssid),
                                              prefix="bssid"))
    X = pd.concat(feature_frames, axis=1).fillna(0.0)

    X_train, X_test, y_train, y_test = tts(X, y, test_size=test_size, random_state=random_state)
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=random_state, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(n_estimators=200, random_state=random_state),
    }
    rows: List[Dict[str, object]] = []
    fitted: Dict[str, object] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            "model": name,
            "mae": float(mean_absolute_error(y_test, pred)),
            "rmse": float(math.sqrt(mean_squared_error(y_test, pred))),
            "r2": float(r2_score(y_test, pred)),
        })
        fitted[name] = model
    metrics = pd.DataFrame(rows).sort_values("mae").reset_index(drop=True)
    best = metrics.iloc[0]["model"]
    return {"metrics": metrics, "fitted_models": fitted, "best": best,
            "X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
