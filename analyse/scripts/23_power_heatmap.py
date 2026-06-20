"""
23_power_heatmap.py — Cartographie en puissance (consigne A2).
À partir du jeu de régression (points x,y + RSSI par borne), on interpole la
puissance de chaque borne sur la zone 5×7 m. Donne les cartes de chaleur RSSI
qui révèlent la structure spatiale du signal (et expliquent pourquoi X est plus
dur à prédire que Y).
Produit fig21_heatmaps.png.
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
from scipy.interpolate import griddata

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
NAVY = "#001E62"
fams = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Segoe UI" if "Segoe UI" in fams else "DejaVu Sans"
plt.rcParams.update({"figure.dpi": 150, "font.family": FONT, "font.size": 9,
                     "axes.titlecolor": NAVY})


def main():
    df = pd.read_csv(ROOT / "data" / "regression" / "dataset_regression.csv")
    xy = df[["X", "Y"]].to_numpy(float)
    aps = df.drop(columns=["X", "Y"])
    # bornes les mieux couvertes (présentes, RSSI != 0) et de bon niveau
    cov = (aps != 0).sum(0)
    cand = cov[cov >= 30].index.tolist()
    strength = {a: aps.loc[aps[a] != 0, a].mean() for a in cand}
    top = sorted(cand, key=lambda a: strength[a], reverse=True)[:4]
    print("Bornes cartographiées:", top)

    gx = np.linspace(xy[:, 0].min(), xy[:, 0].max(), 120)
    gy = np.linspace(xy[:, 1].min(), xy[:, 1].max(), 120)
    GX, GY = np.meshgrid(gx, gy)

    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for ax, ap in zip(axes.ravel(), top):
        mask = aps[ap].to_numpy() != 0
        pts = xy[mask]; vals = aps[ap].to_numpy()[mask]
        grid = griddata(pts, vals, (GX, GY), method="cubic")
        grid_lin = griddata(pts, vals, (GX, GY), method="linear")
        grid = np.where(np.isnan(grid), grid_lin, grid)
        im = ax.imshow(grid, extent=[gx.min(), gx.max(), gy.min(), gy.max()],
                       origin="lower", cmap="viridis", aspect="auto")
        ax.scatter(pts[:, 0], pts[:, 1], c="white", edgecolor="black", s=18, linewidth=0.5, zorder=3)
        ax.set_title(f"{ap}", fontsize=9)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("RSSI (dBm)", fontsize=8)
    fig.suptitle("Cartographie en puissance — RSSI interpolé par borne sur la zone (points blancs = mesures)",
                 color=NAVY, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIG / "fig21_heatmaps.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> fig21_heatmaps.png")


if __name__ == "__main__":
    main()
