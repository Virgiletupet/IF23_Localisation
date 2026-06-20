"""
18_staircase_final.py — Escalier de réalisme DÉFINITIF (avec données ALT2 réelles).
Du test aléatoire (ALT1) jusqu'au vrai test sur une nouvelle session (ALT2, avril),
puis récupération après ré-apprentissage. Remplace la version simulée.
Produit fig13_escalier.png (écrasé).
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.axisbelow": True})


def main():
    sb = pd.read_csv(FIG / "eval_summary_bssid.csv")
    et = sb[sb["model"] == "ExtraTrees"].iloc[0]
    head = pd.read_csv(FIG / "honest_headline_bssid.csv").iloc[0]
    ev = pd.read_csv(FIG / "evolution_real.csv")
    s1 = float(ev.loc[ev["strategie"].str.startswith("S1"), "accuracy"].iloc[0])
    s2 = float(ev.loc[ev["strategie"].str.startswith("S2"), "accuracy"].iloc[0])

    steps = [
        ("1. Test aléatoire\n(même session ALT1)", et["A_aleatoire"], "#d9534f"),
        ("2. + séparation\ntemporelle (CV)", et["C2_groupkfold"], "#e67e22"),
        ("3. + holdout temporel\nstrict (fin de session)", head["accuracy"], "#f0ad4e"),
        ("4. Test RÉEL sur ALT2\n(avril, +2,5 mois)", s1, "#c0392b"),
    ]
    labels = [s[0] for s in steps]; vals = [s[1] * 100 for s in steps]; cols = [s[2] for s in steps]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    xs = list(range(len(steps)))
    bars = ax.bar(xs, vals, color=cols, edgecolor="black", linewidth=0.6, width=0.72, zorder=2)
    ax.step([x - 0.5 for x in xs] + [xs[-1] + 0.5], vals + [vals[-1]], where="post",
            color="#555", linewidth=1.1, zorder=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontweight="bold")
    # Récupération après ré-apprentissage
    xr = len(steps)
    ax.bar([xr], [s2 * 100], color="#27ae60", edgecolor="black", linewidth=0.6, width=0.72, zorder=2)
    ax.text(xr, s2 * 100 + 1.5, f"{s2*100:.0f}%", ha="center", fontweight="bold", color="#1e7e34")
    labels.append("5. Après ré-apprentissage\n(+50% ALT2)")
    ax.annotate("", xy=(xr, s2 * 100), xytext=(xr - 1, s1 * 100 + 3),
                arrowprops=dict(arrowstyle="->", color="#1e7e34", lw=1.6))
    ax.text(xr - 0.5, (s2 * 100 + s1 * 100) / 2 + 6, "ré-apprentissage", color="#1e7e34",
            fontsize=8.5, style="italic", ha="center")

    ax.set_xticks(list(range(len(labels)))); ax.set_xticklabels(labels, fontsize=8.3)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 108)
    ax.set_title("L'escalier de réalisme : du test de base (99 %) au vrai test sur une nouvelle\n"
                 "session (ALT2, 47 %), puis récupération par ré-apprentissage (100 %)")
    fig.tight_layout(); fig.savefig(FIG / "fig13_escalier.png", bbox_inches="tight"); plt.close(fig)
    print(f"S1(ALT2)={s1:.3f}  S2={s2:.3f}  -> fig13_escalier.png réécrit")


if __name__ == "__main__":
    main()
