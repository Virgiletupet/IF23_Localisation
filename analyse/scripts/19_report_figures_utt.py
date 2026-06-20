"""
19_report_figures_utt.py — Figures du rapport, palette UTT SOBRE et compacte.
Palette resserrée : bleu UTT dominant + rouge/vert en accents uniquement.
Pas de titre dans les figures (les légendes Word décrivent) -> plus propre.
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
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
sys.path.insert(0, str(ROOT / "src"))

# Palette sobre : famille bleue + 2 accents
B_DARK = "#0B3C77"; B_MID = "#1F77B4"; B_LIGHT = "#8FBEE0"; B_PALE = "#CFE0F0"
RED = "#C0392B"; GREEN = "#2E8B57"; GREY = "#B6BCC2"; NAVY = "#001E62"
BLUES4 = ["#0B3C77", "#2171B5", "#6BAED6", "#BDD7E7"]

fams = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Segoe UI" if "Segoe UI" in fams else "DejaVu Sans"
plt.rcParams.update({
    "figure.dpi": 150, "font.family": FONT, "font.size": 10,
    "axes.grid": True, "grid.alpha": 0.20, "grid.color": "#cccccc", "axes.axisbelow": True,
    "axes.edgecolor": "#bbbbbb", "axes.labelcolor": "#333", "axes.labelsize": 9.5,
    "xtick.color": "#333", "ytick.color": "#333", "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(fig, name):
    fig.tight_layout(pad=0.6)
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  ->", name)


def fig_escalier():
    sb = pd.read_csv(FIG / "eval_summary_bssid.csv"); et = sb[sb["model"] == "ExtraTrees"].iloc[0]
    head = pd.read_csv(FIG / "honest_headline_bssid.csv").iloc[0]
    ev = pd.read_csv(FIG / "evolution_real.csv")
    s1 = float(ev.loc[ev["strategie"].str.startswith("S1"), "accuracy"].iloc[0])
    s2 = float(ev.loc[ev["strategie"].str.startswith("S2"), "accuracy"].iloc[0])
    labels = ["Test\naléatoire", "+ séparation\ntemporelle", "+ holdout\ntemporel",
              "Test réel\nALT2 (avril)", "Après ré-\napprentissage"]
    vals = [et["A_aleatoire"] * 100, et["C2_groupkfold"] * 100, head["accuracy"] * 100, s1 * 100, s2 * 100]
    cols = [B_LIGHT, B_MID, B_DARK, RED, GREEN]
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    bars = ax.bar(range(5), vals, color=cols, width=0.68, zorder=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center",
                fontweight="bold", color=NAVY, fontsize=9)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 110)
    save(fig, "fig13_escalier.png")


def fig_evolution():
    pz = pd.read_csv(FIG / "evolution_real_perzone.csv")
    ev = pd.read_csv(FIG / "evolution_real.csv")
    s1 = float(ev.loc[ev["strategie"].str.startswith("S1"), "accuracy"].iloc[0])
    s2 = float(ev.loc[ev["strategie"].str.startswith("S2"), "accuracy"].iloc[0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.6, 3.5), gridspec_kw={"width_ratios": [2, 1]})
    cols = [RED if a < 0.5 else B_MID for a in pz["acc_S1"]]
    a1.bar(pz["zone"], pz["acc_S1"] * 100, color=cols, width=0.7)
    for i, a in enumerate(pz["acc_S1"]):
        a1.text(i, a * 100 + 2, f"{a*100:.0f}", ha="center", fontsize=8, color=NAVY)
    a1.set_ylabel("Accuracy (%)"); a1.set_ylim(0, 108); a1.set_title("Par zone (S1)", fontsize=9, color=NAVY)
    a2.bar(["S1", "S2"], [s1 * 100, s2 * 100], color=[RED, GREEN], width=0.6)
    for i, v in enumerate([s1 * 100, s2 * 100]):
        a2.text(i, v + 2, f"{v:.0f}%", ha="center", fontweight="bold", color=NAVY)
    a2.set_ylim(0, 108); a2.set_title("Global", fontsize=9, color=NAVY)
    save(fig, "fig18_evolution.png")


def fig_importance():
    fi = pd.read_csv(FIG / "feature_importance_bssid.csv").head(12).iloc[::-1]
    cmap = {"stable": B_MID, "ephemere": RED, "autre": GREY}
    clean = lambda t: (str(t).encode("ascii", "ignore").decode().strip() or "<masque>")
    labels = [clean(r["ssid"]) if isinstance(r.get("ssid"), str) and r.get("ssid") else clean(r["network"])
              for _, r in fi.iterrows()]
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ax.barh(range(len(fi)), fi["importance"], color=[cmap.get(t, GREY) for t in fi["type"]])
    ax.set_yticks(range(len(fi))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Importance")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=B_MID, label="institutionnel"),
                       Patch(color=RED, label="personnel (éphémère)"),
                       Patch(color=GREY, label="autre")], loc="lower right", fontsize=7.5, framealpha=0.9)
    save(fig, "fig4_importance_reseaux.png")


def fig_hierarchique():
    hh = pd.read_csv(FIG / "hierarchical_bssid.csv")
    short = {"Plat (24 salles)": "Plat (salle directe)",
             "Classification d'étage": "Détection d'étage",
             "Hiérarchique (étage prédit -> salle)": "Hiérarchique (étage prédit)",
             "Hiérarchique (étage oracle -> salle)": "Hiérarchique (étage connu)"}
    hh = hh.copy(); hh["lab"] = hh["approche"].map(lambda x: short.get(x, x))
    cols = [GREEN if "étage" in a and "Hiér" not in a else B_MID for a in hh["approche"]]
    fig, ax = plt.subplots(figsize=(6.8, 3.0))
    bars = ax.barh(hh["lab"][::-1], hh["accuracy"][::-1] * 100, color=cols[::-1])
    for b, v in zip(bars, hh["accuracy"][::-1] * 100):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v:.0f}%", va="center", fontsize=8, color=NAVY)
    ax.set_xlabel("Accuracy (%)"); ax.set_xlim(0, 105)
    save(fig, "fig12_hierarchique.png")


def fig_couverture():
    from eval_utils import build_dataset
    X, _, _, _ = build_dataset(scope="bssid")
    pres = [c for c in X.columns if c.startswith("presence__")]
    cov = X[pres].mean().sort_values(ascending=False).values * 100
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.plot(range(len(cov)), cov, color=B_DARK, linewidth=1.8)
    ax.fill_between(range(len(cov)), cov, alpha=0.15, color=B_MID)
    ax.set_xlabel("Réseaux (triés par couverture)"); ax.set_ylabel("Présence (%)")
    save(fig, "fig17_couverture.png")


def fig_regression():
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv(ROOT / "data" / "regression" / "dataset_regression.csv").fillna(0.0)
    y = df[["X", "Y"]].to_numpy(float); X = df.drop(columns=["X", "Y"]).to_numpy(float)
    models = [("ExtraTrees", ExtraTreesRegressor(n_estimators=400, random_state=42, n_jobs=-1)),
              ("RandomForest", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
              ("k-NN", Pipeline([("s", StandardScaler()), ("m", KNeighborsRegressor(3, weights="distance"))])),
              ("Ridge", Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=10.0))]))]
    cv = KFold(5, shuffle=True, random_state=42)
    eu = lambda a, b: np.sqrt(((a - b) ** 2).sum(axis=1))
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    best = (1e9, None, None)
    for (name, mdl), c in zip(models, BLUES4):
        pred = cross_val_predict(mdl, X, y, cv=cv); err = np.sort(eu(pred, y))
        ax.plot(err, np.linspace(0, 1, len(err)), label=name, linewidth=1.8, color=c)
        if err.mean() < best[0]:
            best = (err.mean(), name, pred)
    ax.set_xlabel("Erreur de position (m)"); ax.set_ylabel("CDF"); ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, framealpha=0.9)
    save(fig, "fig19_regression_cdf.png")
    _, bname, bpred = best
    fig, ax = plt.subplots(figsize=(4.4, 4.4))
    ax.scatter(y[:, 0], y[:, 1], c=B_MID, label="Vraie", s=42, zorder=3, edgecolor="white", linewidth=0.5)
    ax.scatter(bpred[:, 0], bpred[:, 1], c=RED, marker="x", label="Prédite", s=42, zorder=3)
    for i in range(len(y)):
        ax.plot([y[i, 0], bpred[i, 0]], [y[i, 1], bpred[i, 1]], color=GREY, linewidth=0.6, zorder=1)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_aspect("equal"); ax.legend(fontsize=8)
    save(fig, "fig20_regression_scatter.png")


def main():
    print(f"Police: {FONT} | figures sobres ->", FIG)
    for f in [fig_escalier, fig_evolution, fig_importance, fig_hierarchique, fig_couverture, fig_regression]:
        try:
            f()
        except Exception as e:
            print("  ERREUR", f.__name__, ":", e)
    print("OK")


if __name__ == "__main__":
    main()
