"""
14_temporal_strategies.py — Évolution temporelle ALT1/ALT2/ALT3 (consigne, obligatoire).

Teste les 3 stratégies demandées :
  S1 : train ALT1                  -> test ALT2 et ALT3
  S2 : train ALT1 + 50% ALT2       -> test reste ALT2 et ALT3
  S3 : train ALT1 + ALT2           -> test ALT3

Les datasets sont attendus dans :
  data/raw        (ALT1, déjà présent)
  data/raw_alt2   (ALT2 — à déposer)
  data/raw_alt3   (ALT3 — à déposer)
Si ALT2/ALT3 sont absents, le script l'indique et s'arrête proprement.
Le builder de features est ajusté sur ALT1 puis appliqué tel quel aux autres
jeux (mêmes réseaux, mêmes colonnes), comme l'exige une évaluation cohérente.
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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from robust_localization import (  # noqa: E402
    RobustFeatureBuilder, build_snapshot_tables, load_raw_wifi_data,
)

FIG = ROOT / "reports" / "figures"
RS = 42
DIRS = {"ALT1": ROOT / "data" / "raw",
        "ALT2": ROOT / "data" / "raw_alt2",
        "ALT3": ROOT / "data" / "raw_alt3"}


def load_scope_bssid(raw_dir: Path):
    raw = load_raw_wifi_data(raw_dir)
    raw = raw.copy(); raw["ssid"] = raw["bssid"]
    return build_snapshot_tables(raw)


def dataset_present(d: Path) -> bool:
    return d.exists() and any(d.glob("wifi_*.csv"))


def et():
    return ExtraTreesClassifier(n_estimators=300, max_depth=24, class_weight="balanced",
                               random_state=RS, n_jobs=-1)


def score(model, X, y):
    p = model.predict(X)
    return accuracy_score(y, p), f1_score(y, p, average="macro")


def main():
    # ALT1 : ajustement du builder
    per1, snap1 = load_scope_bssid(DIRS["ALT1"])
    builder = RobustFeatureBuilder(max_ssids=120, min_ssid_frequency=5)
    X1, y1 = builder.fit_transform(per1, snap1)
    print(f"ALT1 : {X1.shape[0]} snapshots x {X1.shape[1]} features")

    missing = [k for k in ("ALT2", "ALT3") if not dataset_present(DIRS[k])]
    if missing:
        print("\n*** Datasets manquants : " + ", ".join(missing))
        print("    Déposez les CSV 'wifi_<zone>.csv' dans :")
        for k in missing:
            print(f"      {DIRS[k]}")
        print("    Puis relancez ce script : python scripts/14_temporal_strategies.py")
        return

    per2, snap2 = load_scope_bssid(DIRS["ALT2"])
    X2 = builder.transform(per2, snap2)
    y2 = snap2.set_index(["room", "time"]).loc[X2.index].index.get_level_values("room").to_numpy()
    per3, snap3 = load_scope_bssid(DIRS["ALT3"])
    X3 = builder.transform(per3, snap3)
    y3 = snap3.set_index(["room", "time"]).loc[X3.index].index.get_level_values("room").to_numpy()
    print(f"ALT2 : {X2.shape[0]} snapshots | ALT3 : {X3.shape[0]} snapshots")

    rows = []
    # S1 : train ALT1
    m = et().fit(X1, y1.to_numpy())
    a2, f2 = score(m, X2, y2); a3, f3 = score(m, X3, y3)
    rows += [{"strategie": "S1: ALT1 -> ALT2", "accuracy": a2, "f1_macro": f2},
             {"strategie": "S1: ALT1 -> ALT3", "accuracy": a3, "f1_macro": f3}]

    # S2 : train ALT1 + 50% ALT2 -> reste ALT2, ALT3
    rng = np.random.default_rng(RS)
    idx = rng.permutation(len(X2)); half = len(X2) // 2
    tr2, te2 = idx[:half], idx[half:]
    Xtr = pd.concat([X1, X2.iloc[tr2]]); ytr = np.concatenate([y1.to_numpy(), y2[tr2]])
    m = et().fit(Xtr, ytr)
    a2, f2 = score(m, X2.iloc[te2], y2[te2]); a3, f3 = score(m, X3, y3)
    rows += [{"strategie": "S2: ALT1+50%ALT2 -> reste ALT2", "accuracy": a2, "f1_macro": f2},
             {"strategie": "S2: ALT1+50%ALT2 -> ALT3", "accuracy": a3, "f1_macro": f3}]

    # S3 : train ALT1 + ALT2 -> ALT3
    Xtr = pd.concat([X1, X2]); ytr = np.concatenate([y1.to_numpy(), y2])
    m = et().fit(Xtr, ytr)
    a3, f3 = score(m, X3, y3)
    rows += [{"strategie": "S3: ALT1+ALT2 -> ALT3", "accuracy": a3, "f1_macro": f3}]

    df = pd.DataFrame(rows)
    df["accuracy"] = df["accuracy"].round(3); df["f1_macro"] = df["f1_macro"].round(3)
    print("\n=== Évolution temporelle (ExtraTrees, scope BSSID) ===")
    print(df.to_string(index=False))
    df.to_csv(FIG / "temporal_strategies.csv", index=False)
    print(f"\nCSV -> {FIG / 'temporal_strategies.csv'}")


if __name__ == "__main__":
    main()
