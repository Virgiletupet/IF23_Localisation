"""
17_regression.py — Régression intra-zone (position x,y), consigne page 5.
Entrée : dataset_regression.csv (X, Y + RSSI par BSSID). Sortie : (x, y).
Évaluation par validation croisée (peu de points -> KFold), métriques du cours :
erreur euclidienne moyenne/médiane/p90 et CDF des erreurs.
Produit fig19_regression_cdf.png, fig20_regression_scatter.png, regression_results.csv
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
RS = 42
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})


def euclid(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))


def main():
    df = pd.read_csv(ROOT / "data" / "regression" / "dataset_regression.csv").fillna(0.0)
    y = df[["X", "Y"]].to_numpy(float)
    X = df.drop(columns=["X", "Y"]).to_numpy(float)
    print(f"Régression : {X.shape[0]} points, {X.shape[1]} BSSID | zone {y[:,0].min():.0f}..{y[:,0].max():.0f} x "
          f"{y[:,1].min():.0f}..{y[:,1].max():.0f} m")

    models = {
        "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=RS, n_jobs=-1),
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=RS, n_jobs=-1),
        "k-NN (dist)": Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor(n_neighbors=3, weights="distance"))]),
        "Ridge": Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=10.0))]),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RS)
    rows = {}
    best_name, best_pred, best_mae = None, None, 1e9
    for name, mdl in models.items():
        pred = cross_val_predict(mdl, X, y, cv=cv)
        err = euclid(pred, y)
        mae, med, p90 = err.mean(), np.median(err), np.percentile(err, 90)
        rows[name] = {"modele": name, "MAE_m": round(mae, 3), "mediane_m": round(med, 3),
                      "p90_m": round(p90, 3), "max_m": round(err.max(), 3)}
        print(f"  {name:14s} MAE={mae:.2f} m | médiane={med:.2f} m | p90={p90:.2f} m")
        if mae < best_mae:
            best_mae, best_name, best_pred = mae, name, pred
    pd.DataFrame(rows.values()).to_csv(FIG / "regression_results.csv", index=False)
    print(f"Meilleur : {best_name} (MAE {best_mae:.2f} m)")

    # CDF des erreurs (toutes méthodes)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name, mdl in models.items():
        pred = cross_val_predict(mdl, X, y, cv=cv)
        err = np.sort(euclid(pred, y))
        ax.plot(err, np.linspace(0, 1, len(err)), label=name, linewidth=1.8)
    ax.set_xlabel("Erreur de position (m)"); ax.set_ylabel("Fraction des points (CDF)")
    ax.set_title("Régression intra-zone : CDF de l'erreur de position\n"
                 "Médiane ≈ 1,1 m sur une zone de 5×7 m")
    ax.legend(); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(FIG / "fig19_regression_cdf.png", bbox_inches="tight"); plt.close(fig)
    print("  -> fig19_regression_cdf.png")

    # Scatter vrai vs prédit (meilleur modèle)
    fig, ax = plt.subplots(figsize=(6.5, 7))
    ax.scatter(y[:, 0], y[:, 1], c="#27ae60", label="Vraie position", s=45, zorder=3)
    ax.scatter(best_pred[:, 0], best_pred[:, 1], c="#d9534f", marker="x", label="Prédiction", s=45, zorder=3)
    for i in range(len(y)):
        ax.plot([y[i, 0], best_pred[i, 0]], [y[i, 1], best_pred[i, 1]], color="#999", linewidth=0.6, zorder=1)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal")
    ax.set_title(f"Position vraie vs prédite — {best_name}\n(les traits relient vraie position et estimation)")
    ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig20_regression_scatter.png", bbox_inches="tight"); plt.close(fig)
    print("  -> fig20_regression_scatter.png")


if __name__ == "__main__":
    main()
