"""
25_map_assist.py — Aide cartographique DÉSACTIVABLE (exploitation de la mobilité).
Idée (consigne, optionnelle) : on ne peut pas « téléporter » d'une zone à une
autre. En suivant une trajectoire, on contraint les prédictions successives par
un graphe de transition entre zones voisines (lissage de type Viterbi).

  aide_carto = OFF -> prédiction brute (argmax à chaque instant)
  aide_carto = ON  -> décodage Viterbi (émissions = proba du modèle,
                      transitions = graphe d'adjacence des zones)

On MESURE le gain sur des trajectoires simulées respectant l'adjacence. Par
défaut l'aide reste OFF ; on ne l'active que si elle améliore.
Produit fig23_map_assist.png + map_assist.csv
"""
from __future__ import annotations

import re
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
sys.path.insert(0, str(ROOT / "src"))
from robust_localization import RobustFeatureBuilder, build_snapshot_tables, load_raw_wifi_data  # noqa

RS = 42
NAVY = "#001E62"; BLUE = "#1F77B4"; GREEN = "#2E8B57"
fams = {f.name for f in font_manager.fontManager.ttflist}
FONT = "Segoe UI" if "Segoe UI" in fams else "DejaVu Sans"
plt.rcParams.update({"figure.dpi": 150, "font.family": FONT, "font.size": 10,
                     "axes.titlecolor": NAVY, "axes.spines.top": False, "axes.spines.right": False})


def merge_zone(z):
    return z[:-1] if re.match(r"^[PS]\d{3}[AB]$", str(z)) else str(z)


def floor_of(z):
    m = re.match(r"^[PS](\d)\d\d$", z)
    if m: return int(m.group(1))
    if "1ER" in z or z == "couloirB": return 1
    if "2E" in z or z == "couloirB(2e)": return 2
    return 0


def section_of(z):
    return z[0] if z and z[0] in "SP" else "C"


def adjacency(zones):
    """Deux zones sont reliées si même étage (couloir commun) ; les paliers
    relient les étages consécutifs."""
    n = len(zones); A = np.zeros((n, n))
    for i, zi in enumerate(zones):
        for j, zj in enumerate(zones):
            if i == j:
                continue
            fi, fj = floor_of(zi), floor_of(zj)
            same_floor = fi == fj and fi != 0
            palier_link = ("PALIER" in zi or "PALIER" in zj) and abs(fi - fj) == 1
            if same_floor or palier_link:
                A[i, j] = 1
    return A


def transition_matrix(zones, p_stay=0.5):
    A = adjacency(zones); n = len(zones)
    T = np.full((n, n), 1e-4)
    for i in range(n):
        T[i, i] = p_stay
        nb = A[i].sum()
        if nb > 0:
            T[i, A[i] > 0] += (1 - p_stay) / nb
        T[i] /= T[i].sum()
    return T


def viterbi(log_em, log_T):
    n_steps, n = log_em.shape
    dp = np.full((n_steps, n), -np.inf); back = np.zeros((n_steps, n), int)
    dp[0] = log_em[0] + np.log(np.full(n, 1.0 / n))
    for t in range(1, n_steps):
        for j in range(n):
            scores = dp[t - 1] + log_T[:, j]
            back[t, j] = np.argmax(scores)
            dp[t, j] = scores[back[t, j]] + log_em[t, j]
    path = [int(np.argmax(dp[-1]))]
    for t in range(n_steps - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def main():
    raw = load_raw_wifi_data(ROOT / "data" / "raw"); raw = raw.copy()
    raw["room"] = raw["room"].map(merge_zone); raw["ssid"] = raw["bssid"]
    per, snap = build_snapshot_tables(raw)
    builder = RobustFeatureBuilder(max_ssids=200, min_ssid_frequency=5)
    X, y = builder.fit_transform(per, snap)
    y = y.to_numpy()
    times = pd.to_datetime(X.index.get_level_values("time"), errors="coerce")

    # split temporel par zone : train 70% / test 30%
    df = pd.DataFrame({"room": y, "t": times}).reset_index(drop=True)
    train_mask = np.zeros(len(df), bool)
    for room, sub in df.groupby("room"):
        order = sub.sort_values("t").index.to_numpy()
        train_mask[order[:int(round(len(order) * 0.7))]] = True
    le = LabelEncoder(); ye = le.fit_transform(y); zones = list(le.classes_)
    clf = ExtraTreesClassifier(n_estimators=300, max_depth=24, class_weight="balanced",
                               random_state=RS, n_jobs=-1).fit(X[train_mask], ye[train_mask])

    Xte, yte = X[~train_mask], ye[~train_mask]
    proba_all = clf.predict_proba(Xte)
    # index des snapshots de test par zone
    by_zone = {z: np.where(yte == i)[0] for i, z in enumerate(zones)}
    by_zone = {z: idx for z, idx in by_zone.items() if len(idx) > 0}
    valid_zones = list(by_zone.keys())

    T = transition_matrix(zones, p_stay=0.5)
    logT = np.log(T)
    rng = np.random.default_rng(RS)
    A = adjacency(zones)

    n_traj, steps = 200, 12
    acc_off, acc_on = [], []
    for _ in range(n_traj):
        # marche aléatoire respectant l'adjacence
        cur = rng.choice([zones.index(z) for z in valid_zones])
        path_true, em_idx = [], []
        for _ in range(steps):
            if zones[cur] in by_zone:
                path_true.append(cur); em_idx.append(rng.choice(by_zone[zones[cur]]))
            nxts = np.where((A[cur] > 0))[0]
            nxts = [z for z in nxts if zones[z] in by_zone]
            cur = rng.choice(nxts) if (len(nxts) and rng.random() > 0.4) else cur
        if len(path_true) < 3:
            continue
        em = proba_all[em_idx]                      # (steps, n_zones)
        pred_off = np.argmax(em, axis=1)
        log_em = np.log(np.clip(em, 1e-9, 1))
        pred_on = viterbi(log_em, logT)
        yt = np.array(path_true)
        acc_off.append(np.mean(pred_off == yt))
        acc_on.append(np.mean(np.array(pred_on) == yt))

    off, on = float(np.mean(acc_off)), float(np.mean(acc_on))
    res = pd.DataFrame([{"aide_carto": "OFF (brut)", "accuracy_trajectoire": round(off, 3)},
                        {"aide_carto": "ON (mobilité)", "accuracy_trajectoire": round(on, 3)}])
    res.to_csv(FIG / "map_assist.csv", index=False)
    print(res.to_string(index=False))
    print(f"Gain de l'aide carto : {(on-off)*100:+.1f} points (sur {len(acc_off)} trajectoires simulées)")

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    bars = ax.bar(["Aide OFF\n(prédiction brute)", "Aide ON\n(mobilité)"], [off * 100, on * 100],
                  color=[BLUE, GREEN], width=0.55)
    for b, v in zip(bars, [off * 100, on * 100]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontweight="bold", color=NAVY)
    ax.set_ylabel("Accuracy sur trajectoire (%)"); ax.set_ylim(0, 105)
    ax.set_title("Aide cartographique (mobilité) : impact mesuré")
    fig.tight_layout(pad=0.6); fig.savefig(FIG / "fig23_map_assist.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  -> fig23_map_assist.png, map_assist.csv")


if __name__ == "__main__":
    main()
