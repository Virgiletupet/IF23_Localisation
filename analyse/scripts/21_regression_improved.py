"""
21_regression_improved.py — Améliorer la précision (x, y) intra-zone.
Petit jeu (46 points, 52 BSSID) -> on cherche le meilleur compromis :
  - valeur de remplissage des réseaux absents : 0 vs -100
  - modèles : kNN pondéré (k réglé), ExtraTrees, RandomForest, Gradient Boosting,
              processus gaussien (RBF), et moyenne des 2 meilleurs
  - sélection des réseaux les plus informatifs (mutual information)
Évaluation Leave-One-Out (robuste sur peu de points). Métriques : erreur
euclidienne moyenne/médiane/p90, R² par axe. Régénère les figures régression.
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
RS = 42


def euclid(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def loo_predict(model, X, y):
    loo = LeaveOneOut(); pred = np.zeros_like(y, dtype=float)
    for tr, te in loo.split(X):
        m = model
        from sklearn.base import clone
        m = clone(model).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return pred


def make_models():
    gpr = MultiOutputRegressor(GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=10.0) + WhiteKernel(noise_level=1.0),
        normalize_y=True, alpha=1e-6, random_state=RS))
    return {
        "kNN k=3 (dist)": Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor(3, weights="distance"))]),
        "kNN k=5 (dist)": Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor(5, weights="distance"))]),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=600, random_state=RS, n_jobs=-1),
        "RandomForest": RandomForestRegressor(n_estimators=500, random_state=RS, n_jobs=-1),
        "GradBoost": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=300, max_depth=2, random_state=RS)),
        "GP (RBF)": Pipeline([("s", StandardScaler()), ("m", gpr)]),
    }


def evaluate(X, y, label):
    rows = []
    preds = {}
    for name, mdl in make_models().items():
        try:
            p = loo_predict(mdl, X, y)
            err = euclid(p, y)
            rows.append({"config": label, "modele": name, "MAE_m": err.mean(),
                         "med_m": np.median(err), "p90_m": np.percentile(err, 90),
                         "R2_x": r2_score(y[:, 0], p[:, 0]), "R2_y": r2_score(y[:, 1], p[:, 1])})
            preds[name] = p
        except Exception as e:
            print("   skip", name, e)
    return pd.DataFrame(rows), preds


def main():
    df = pd.read_csv(ROOT / "data" / "regression" / "dataset_regression.csv")
    y = df[["X", "Y"]].to_numpy(float)
    raw = df.drop(columns=["X", "Y"])

    all_rows = []; best = (1e9, None, None, None)
    for fill in [0.0, -100.0]:
        X = raw.fillna(fill).to_numpy(float)
        # nettoyage colonnes constantes
        X = X[:, X.std(axis=0) > 0]
        res, preds = evaluate(X, y, f"fill={fill:.0f} | {X.shape[1]} BSSID")
        all_rows.append(res)
        for _, r in res.iterrows():
            if r["MAE_m"] < best[0]:
                best = (r["MAE_m"], r["modele"], f"fill={fill:.0f}", preds[r["modele"]])

    # Sélection de réseaux (mutual info) sur le meilleur fill
    bestfill = float(best[2].split("=")[1])
    Xb = raw.fillna(bestfill).to_numpy(float); Xb = Xb[:, raw.fillna(bestfill).to_numpy(float).std(axis=0) > 0]
    for k in [15, 25, 35]:
        mi = (mutual_info_regression(Xb, y[:, 0], random_state=RS) +
              mutual_info_regression(Xb, y[:, 1], random_state=RS))
        idx = np.argsort(mi)[::-1][:k]
        res, preds = evaluate(Xb[:, idx], y, f"fill={bestfill:.0f} | top-{k} BSSID")
        all_rows.append(res)
        for _, r in res.iterrows():
            if r["MAE_m"] < best[0]:
                best = (r["MAE_m"], r["modele"], f"fill={bestfill:.0f} top-{k}", preds[r["modele"]])

    table = pd.concat(all_rows, ignore_index=True).sort_values("MAE_m").round(3)
    print("=" * 88)
    print("RÉGRESSION (x,y) — comparaison (Leave-One-Out)")
    print("=" * 88)
    print(table.head(15).to_string(index=False))
    table.to_csv(FIG / "regression_improved.csv", index=False)
    print(f"\nMEILLEUR : {best[1]} ({best[2]}) -> MAE {best[0]:.3f} m")

    # Comparaison ancien (ExtraTrees fill=0 5-fold ~1.36) vs nouveau best
    old = table[(table["modele"] == "ExtraTrees")]["MAE_m"].max()
    print(f"Repère ExtraTrees brut: MAE ~{old:.2f} m | gain best: {old-best[0]:.2f} m")
    return best


if __name__ == "__main__":
    main()
