"""
22_regression_final.py — Régression intra-zone, modèle final HYBRIDE par axe.
Constat : X et Y ont des structures différentes. On prédit X par processus
gaussien (meilleur sur l'axe difficile) et Y par ExtraTrees (meilleur sur Y).
Évaluation Leave-One-Out. Régénère fig19/fig20 et regression_final.csv.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
RS = 42
B_DARK = "#0B3C77"; B_MID = "#1F77B4"; RED = "#C0392B"; GREY = "#B6BCC2"; NAVY = "#001E62"
BLUES = ["#0B3C77", "#2171B5", "#6BAED6", "#9ECAE1"]
fams = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Segoe UI" if "Segoe UI" in fams else "DejaVu Sans"
plt.rcParams.update({"figure.dpi": 150, "font.family": FONT, "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.2, "axes.spines.top": False,
                     "axes.spines.right": False})


def loo_axis(model, X, ya):
    p = np.zeros(len(ya))
    for tr, te in LeaveOneOut().split(X):
        p[te] = clone(model).fit(X[tr], ya[tr]).predict(X[te])
    return p


def main():
    df = pd.read_csv(ROOT / "data" / "regression" / "dataset_regression.csv")
    y = df[["X", "Y"]].to_numpy(float)
    X = df.drop(columns=["X", "Y"]).fillna(0.0).to_numpy(float); X = X[:, X.std(0) > 0]
    print(f"{len(y)} points | {X.shape[1]} BSSID")

    et = ExtraTreesRegressor(n_estimators=600, random_state=RS, n_jobs=-1)
    rf = RandomForestRegressor(n_estimators=500, random_state=RS, n_jobs=-1)
    gp = Pipeline([("s", StandardScaler()),
                   ("m", GaussianProcessRegressor(kernel=C(1.0) * RBF(10) + WhiteKernel(1.0),
                                                  normalize_y=True, random_state=RS))])
    knn = Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor(4, weights="distance"))])

    def preds_multi(m):
        return np.column_stack([loo_axis(m, X, y[:, 0]), loo_axis(m, X, y[:, 1])])

    configs = {
        "k-NN pondéré": preds_multi(knn),
        "RandomForest": preds_multi(rf),
        "ExtraTrees": preds_multi(et),
        "Processus gaussien": preds_multi(gp),
        "Hybride (X=GP, Y=ET)": np.column_stack([loo_axis(gp, X, y[:, 0]), loo_axis(et, X, y[:, 1])]),
    }
    rows = []
    for name, p in configs.items():
        err = np.sqrt(((p - y) ** 2).sum(1))
        rows.append({"modele": name, "MAE_m": round(err.mean(), 3), "med_m": round(np.median(err), 3),
                     "p90_m": round(np.percentile(err, 90), 3),
                     "R2_x": round(r2_score(y[:, 0], p[:, 0]), 2), "R2_y": round(r2_score(y[:, 1], p[:, 1]), 2)})
    tab = pd.DataFrame(rows).sort_values("MAE_m")
    tab.to_csv(FIG / "regression_final.csv", index=False)
    print(tab.to_string(index=False))
    best = "Hybride (X=GP, Y=ET)"; bp = configs[best]

    # CDF
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    order = ["k-NN pondéré", "ExtraTrees", "Processus gaussien", "Hybride (X=GP, Y=ET)"]
    cols = [GREY, BLUES[2], BLUES[1], RED]
    for name, c in zip(order, cols):
        err = np.sort(np.sqrt(((configs[name] - y) ** 2).sum(1)))
        lw = 2.4 if name == best else 1.5
        ax.plot(err, np.linspace(0, 1, len(err)), label=name, color=c, linewidth=lw)
    ax.set_xlabel("Erreur de position (m)"); ax.set_ylabel("CDF"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, framealpha=0.9)
    fig.tight_layout(pad=0.6); fig.savefig(FIG / "fig19_regression_cdf.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Scatter hybride
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.scatter(y[:, 0], y[:, 1], c=B_MID, label="Vraie", s=42, zorder=3, edgecolor="white", linewidth=0.5)
    ax.scatter(bp[:, 0], bp[:, 1], c=RED, marker="x", label="Prédite", s=42, zorder=3)
    for i in range(len(y)):
        ax.plot([y[i, 0], bp[i, 0]], [y[i, 1], bp[i, 1]], color=GREY, linewidth=0.6, zorder=1)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal"); ax.legend(fontsize=8)
    fig.tight_layout(pad=0.6); fig.savefig(FIG / "fig20_regression_scatter.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> fig19_regression_cdf.png, fig20_regression_scatter.png, regression_final.csv")


if __name__ == "__main__":
    main()
