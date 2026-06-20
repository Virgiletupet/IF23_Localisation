"""
05_figures.py
=============
Génère les figures du rapport à partir des CSV produits par les scripts 02/03/04.
Sorties PNG dans reports/figures/.
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

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
sys.path.insert(0, str(ROOT / "src"))

plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
C_BIAS, C_LEAK, C_HON = "#d9534f", "#8e44ad", "#27ae60"


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print("  ->", name)


def fig_protocoles():
    s = pd.read_csv(FIG / "eval_summary_bssid.csv")
    r = s[s["model"] == "ExtraTrees"].iloc[0]
    labels = ["A. Aléatoire\n(biaisé)", "B. In-sample\n(leakage)",
              "C1. Holdout\ntemporel", "C2. GroupKFold\ntemporel"]
    vals = [r["A_aleatoire"], r["B_insample"], r["C1_holdout_temporel"], r["C2_groupkfold"]]
    colors = [C_BIAS, C_LEAK, C_HON, C_HON]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, [v*100 for v in vals], color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, v*100+1, f"{v*100:.1f}%", ha="center", fontweight="bold")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_title("Performance selon le protocole d'évaluation (ExtraTrees, scope BSSID)\n"
                 "Le « 99 % » et le « 100 % live » s'effondrent en évaluation honnête")
    ax.axhspan(0, 0, color=C_BIAS, label="biaisé")
    save(fig, "fig1_protocoles_extratrees.png")


def fig_modeles_honnete():
    c2 = pd.read_csv(FIG / "eval_C2_group_cv_bssid.csv").sort_values("accuracy_mean")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(c2["model"], c2["accuracy_mean"]*100, xerr=c2["accuracy_std"]*100,
            color="#2c7fb8", edgecolor="black", linewidth=0.6, capsize=4)
    for y, (v, s) in enumerate(zip(c2["accuracy_mean"], c2["accuracy_std"])):
        ax.text(v*100+1.5, y, f"{v*100:.1f}%", va="center")
    ax.set_xlabel("Accuracy (%) — StratifiedGroupKFold (honnête)")
    ax.set_xlim(0, 100)
    ax.set_title("Comparaison honnête des modèles vus en cours (scope BSSID)")
    save(fig, "fig2_modeles_honnete.png")


def fig_biaise_vs_honnete():
    s = pd.read_csv(FIG / "eval_summary_bssid.csv")
    x = np.arange(len(s)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w/2, s["A_aleatoire"]*100, w, label="A. Aléatoire (biaisé)", color=C_BIAS, edgecolor="black", linewidth=0.5)
    ax.bar(x+w/2, s["C2_groupkfold"]*100, w, label="C2. GroupKFold (honnête)", color=C_HON, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(s["model"], rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108); ax.legend()
    ax.set_title("Écart entre évaluation biaisée et honnête, par modèle (scope BSSID)")
    save(fig, "fig3_biaise_vs_honnete.png")


def fig_importance():
    fi = pd.read_csv(FIG / "feature_importance_bssid.csv").head(15).iloc[::-1]
    cmap = {"stable": "#27ae60", "ephemere": "#d9534f", "autre": "#999999"}
    def clean(t):
        return (str(t).encode("ascii", "ignore").decode().strip() or "<masqué>")
    labels = [clean(row['ssid']) if isinstance(row.get("ssid"), str) and row.get("ssid") else clean(row["network"])
              for _, row in fi.iterrows()]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(len(fi)), fi["importance"], color=[cmap.get(t, "#999") for t in fi["type"]],
            edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(fi))); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Importance (somme sur les statistiques)")
    ax.set_title("Top 15 des réseaux les plus discriminants\n"
                 "Les 2 premiers sont des téléphones personnels des collecteurs")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=cmap["stable"], label="institutionnel (stable)"),
                       Patch(color=cmap["ephemere"], label="personnel (éphémère)"),
                       Patch(color=cmap["autre"], label="autre")], loc="lower right")
    save(fig, "fig4_importance_reseaux.png")


def fig_f1_par_zone():
    rep = pd.read_csv(FIG / "honest_classification_report_bssid.csv", index_col=0)
    zones = rep.drop(index=[i for i in ["accuracy", "macro avg", "weighted avg"] if i in rep.index])
    zones = zones.sort_values("f1-score")
    colors = ["#d9534f" if v < 0.5 else ("#f0ad4e" if v < 0.75 else "#27ae60") for v in zones["f1-score"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(zones.index, zones["f1-score"], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("F1-score (protocole honnête, holdout temporel)")
    ax.set_xlim(0, 1.05)
    ax.set_title("F1-score par zone en évaluation honnête\n"
                 "Forte hétérogénéité : certaines zones s'effondrent (confusions d'adjacence)")
    save(fig, "fig5_f1_par_zone.png")


def fig_snapshots_par_zone():
    from robust_localization import load_raw_wifi_data, build_snapshot_tables
    raw = load_raw_wifi_data(ROOT / "data" / "raw")
    _, snap = build_snapshot_tables(raw)
    cnt = snap.groupby("room").size().sort_values()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(cnt.index, cnt.values, color="#2c7fb8", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Nombre de snapshots")
    ax.set_title(f"Distribution des snapshots par zone (total = {cnt.sum()})\n"
                 "Classes déséquilibrées -> importance du macro-F1 / balanced accuracy")
    save(fig, "fig6_snapshots_par_zone.png")


def fig_assaini():
    p = FIG / "sanitized_comparison.csv"
    if not p.exists():
        print("  (sanitized_comparison.csv absent, fig7 ignorée)"); return
    s = pd.read_csv(p)
    name = {"all": "Tous réseaux", "institution": "Institutionnels\nseuls (assaini)", "perso": "Personnels\nseuls"}
    s["lab"] = s["jeu_features"].map(name)
    x = np.arange(len(s)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x-w/2, s["holdout_temp_acc"]*100, w, label="Holdout temporel", color="#2c7fb8", edgecolor="black", linewidth=0.5)
    ax.bar(x+w/2, s["groupkfold_acc"]*100, w, label="GroupKFold", color="#7fcdbb", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(s["lab"]); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100); ax.legend()
    for i, row in s.iterrows():
        ax.text(i-w/2, row["holdout_temp_acc"]*100+1, f"{row['holdout_temp_acc']*100:.0f}", ha="center", fontsize=8)
        ax.text(i+w/2, row["groupkfold_acc"]*100+1, f"{row['groupkfold_acc']*100:.0f}", ha="center", fontsize=8)
    ax.set_title("Modèle assaini : impact du retrait des réseaux personnels (honnête, BSSID)")
    save(fig, "fig7_assaini.png")


def fig_scope():
    b = pd.read_csv(FIG / "eval_summary_bssid.csv"); s = pd.read_csv(FIG / "eval_summary_ssid.csv")
    models = b["model"]; x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x-w/2, b["C2_groupkfold"]*100, w, label="BSSID (par point d'accès)", color="#225ea8", edgecolor="black", linewidth=0.5)
    ax.bar(x+w/2, s["C2_groupkfold"]*100, w, label="SSID (par nom de réseau)", color="#41b6c4", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%) — GroupKFold honnête"); ax.set_ylim(0, 100); ax.legend()
    ax.set_title("BSSID vs SSID en évaluation honnête : l'agrégation par SSID généralise un peu mieux")
    save(fig, "fig8_bssid_vs_ssid.png")


def main():
    print("Génération des figures ->", FIG)
    for f in [fig_protocoles, fig_modeles_honnete, fig_biaise_vs_honnete, fig_importance,
              fig_f1_par_zone, fig_snapshots_par_zone, fig_assaini, fig_scope]:
        try:
            f()
        except Exception as e:
            print("  ERREUR", f.__name__, ":", e)
    print("Terminé.")


if __name__ == "__main__":
    main()
