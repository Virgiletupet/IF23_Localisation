"""
24_floor_sketch.py — Croquis schématique d'étage (cartographie, Brique 3).
Agencement (donné par l'utilisateur) : par section (S / P) et par étage (1er
chiffre du numéro), salles 01 et 02 côte à côte d'un côté, 03 et 04 en face,
un palier entre les deux, et un couloir central. On surligne la zone prédite et
on place, le cas échéant, la position (x, y) estimée dans la salle.
Produit fig22_plan_etage.png (exemple : zone prédite S102).
"""
from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
NAVY = "#001E62"; BLUE = "#1F77B4"; LIGHT = "#D6E6F5"; GREEN = "#2E8B57"
RED = "#C0392B"; GREY = "#E9ECEF"; CORR = "#F4D58D"
fams = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Segoe UI" if "Segoe UI" in fams else "DejaVu Sans"
plt.rcParams.update({"figure.dpi": 150, "font.family": FONT})


def room(ax, x, y, w, hh, label, highlight=False, pos=None):
    fc = "#FBE3DE" if highlight else LIGHT
    ec = RED if highlight else BLUE
    ax.add_patch(FancyBboxPatch((x, y), w, hh, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=2 if highlight else 1.2))
    ax.text(x + w / 2, y + hh / 2 + (0.12 if pos is not None else 0), label,
            ha="center", va="center", fontsize=10, fontweight="bold",
            color=RED if highlight else NAVY)
    if pos is not None:  # position (x,y) normalisée [0,1] dans la salle
        px, py = pos
        ax.plot(x + 0.12 + px * (w - 0.24), y + 0.12 + py * (hh - 0.24), "o",
                color=GREEN, ms=10, mec="white", mew=1.5, zorder=5)
        ax.text(x + w / 2, y + hh / 2 - 0.22, "position estimée", ha="center",
                fontsize=7, color=GREEN, style="italic")


def draw_floor(ax, section, floor, highlight=None, pos=None):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    ax.set_title(f"Section {section} — étage {floor}", color=NAVY, fontsize=11, fontweight="bold")
    # couloir central
    ax.add_patch(Rectangle((0.3, 2.5), 9.4, 1.0, fc=CORR, ec="#C9A227", lw=1))
    ax.text(5, 3.0, "couloir", ha="center", va="center", fontsize=8, color="#8a6d1b")
    f = floor
    top = [f"{section}{f}01", f"{section}{f}02"]
    bot = [f"{section}{f}03", f"{section}{f}04"]
    xs = [0.6, 4.0]
    for lbl, x in zip(top, xs):
        room(ax, x, 3.7, 3.0, 1.9, lbl, highlight=(lbl == highlight),
             pos=pos if lbl == highlight else None)
    for lbl, x in zip(bot, xs):
        room(ax, x, 0.4, 3.0, 1.9, lbl, highlight=(lbl == highlight),
             pos=pos if lbl == highlight else None)
    # palier à droite
    ax.add_patch(FancyBboxPatch((7.7, 0.4), 1.9, 5.2, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=GREY, ec="#9AA0A6", lw=1.2))
    ax.text(8.65, 3.0, f"PALIER\n{floor}", ha="center", va="center", fontsize=8, color="#555")


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # exemple : zone prédite = S102 (étage 1), position estimée dans la salle
    draw_floor(axes[0, 0], "S", 1, highlight="S102", pos=(0.35, 0.6))
    draw_floor(axes[0, 1], "P", 1)
    draw_floor(axes[1, 0], "S", 2)
    draw_floor(axes[1, 1], "P", 2)
    fig.suptitle("Croquis schématique des étages — exemple : zone prédite S102 + position (x, y)",
                 color=NAVY, fontsize=12, fontweight="bold")
    # légende
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig.legend(handles=[Patch(fc="#FBE3DE", ec=RED, label="zone prédite"),
                        Patch(fc=LIGHT, ec=BLUE, label="autres salles"),
                        Patch(fc=CORR, ec="#C9A227", label="couloir"),
                        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN,
                               markersize=9, label="position (x, y)")],
               loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(FIG / "fig22_plan_etage.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> fig22_plan_etage.png")


if __name__ == "__main__":
    main()
