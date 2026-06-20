"""
11_figures_alt3.py — Figures spécifiques aux apports ALT3
(progression honnête, bayésien, tuning, hiérarchique).
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
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})
C_BIAS, C_HON = "#d9534f", "#27ae60"


def save(fig, name):
    fig.tight_layout(); fig.savefig(FIG / name, bbox_inches="tight"); plt.close(fig)
    print("  ->", name)


def fig_progression():
    alt1 = pd.read_csv(FIG / "alt1_reproduction.csv").iloc[0]
    sb = pd.read_csv(FIG / "eval_summary_bssid.csv")
    et = sb[sb["model"] == "ExtraTrees"].iloc[0]
    labels = ["ALT1\n(6 BSSID, 16 zones)", "ALT2/3\n(104 BSSID, 24 zones)"]
    biais = [alt1["biaise_aleatoire"]*100, et["A_aleatoire"]*100]
    honn = [alt1["honnete_groupkfold"]*100, et["C2_groupkfold"]*100]
    x = np.arange(2); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x-w/2, biais, w, label="Annoncé (biaisé, aléatoire)", color=C_BIAS, edgecolor="black", linewidth=0.5)
    ax.bar(x+w/2, honn, w, label="Réel (honnête, GroupKFold)", color=C_HON, edgecolor="black", linewidth=0.5)
    for i in range(2):
        ax.text(i-w/2, biais[i]+1, f"{biais[i]:.0f}%", ha="center", fontsize=9)
        ax.text(i+w/2, honn[i]+1, f"{honn[i]:.0f}%", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
    ax.legend()
    ax.set_title("Progression réelle entre jalons (honnête) : ~40 % -> ~80 %\n"
                 "Le gain d'ALT2 est réel ; seul le chiffre annoncé (99 %) était surestimé")
    save(fig, "fig9_progression_honnete.png")


def fig_bayesien():
    b = pd.read_csv(FIG / "bayesian_bssid.csv")
    x = np.arange(len(b)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x-w/2, b["holdout_acc"]*100, w, label="Holdout temporel", color="#2c7fb8", edgecolor="black", linewidth=0.5)
    ax.bar(x+w/2, b["groupkfold_acc"]*100, w, label="GroupKFold", color="#7fcdbb", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(b["modele"], fontsize=8)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.legend()
    for i, row in b.iterrows():
        ax.text(i-w/2, row["holdout_acc"]*100+1, f"{row['holdout_acc']*100:.0f}", ha="center", fontsize=8)
        ax.text(i+w/2, row["groupkfold_acc"]*100+1, f"{row['groupkfold_acc']*100:.0f}", ha="center", fontsize=8)
    ax.set_title("Modèle bayésien (cours) : la modélisation explicite de la présence\n"
                 "des AP surpasse nettement le GaussianNB brut (scope BSSID)")
    save(fig, "fig10_bayesien.png")


def fig_tuning():
    t = pd.read_csv(FIG / "hyperparam_tuning_bssid.csv")
    x = np.arange(len(t)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x-w/2, t["defaut"]*100, w, label="Par défaut", color="#bdbdbd", edgecolor="black", linewidth=0.5)
    ax.bar(x+w/2, t["optimise"]*100, w, label="Optimisé", color="#2c7fb8", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(t["modele"]); ax.set_ylabel("Accuracy (%) — GroupKFold")
    ax.set_ylim(0, 100); ax.legend()
    for i, row in t.iterrows():
        ax.text(i+w/2, row["optimise"]*100+1, f"+{row['gain']*100:.1f}", ha="center", fontsize=8, color="#27ae60")
    ax.set_title("Optimisation des hyperparamètres (CV group-aware)\nGains réels mais modestes")
    save(fig, "fig11_tuning.png")


def fig_hierarchique():
    hh = pd.read_csv(FIG / "hierarchical_bssid.csv")
    colors = ["#2c7fb8", "#27ae60", "#f0ad4e", "#9b59b6"]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.barh(hh["approche"][::-1], hh["accuracy"][::-1]*100, color=colors[::-1],
                   edgecolor="black", linewidth=0.5)
    for b, v in zip(bars, hh["accuracy"][::-1]*100):
        ax.text(v+1, b.get_y()+b.get_height()/2, f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlabel("Accuracy (%) — holdout temporel honnête"); ax.set_xlim(0, 105)
    ax.set_title("Approche hybride hiérarchique : la détection d'étage est très fiable (94 %),\n"
                 "base solide pour la cartographie")
    save(fig, "fig12_hierarchique.png")


def main():
    print("Figures ALT3 ->", FIG)
    for f in [fig_progression, fig_bayesien, fig_tuning, fig_hierarchique]:
        try:
            f()
        except Exception as e:
            print("  ERREUR", f.__name__, ":", e)
    print("OK")


if __name__ == "__main__":
    main()
