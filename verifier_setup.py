"""
verifier_setup.py — vérifie (sans WiFi) que l'environnement et les modèles
fonctionnent : charge le modèle combiné et prédit sur des scans réels (ALT2).
Reproduit le chemin de prédiction "live" (identifiant composé SSID|BSSID
normalisé), seul chemin cohérent avec l'entraînement.

Usage :  python verifier_setup.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd


def main():
    try:
        import sklearn
        print(f"scikit-learn {sklearn.__version__}")
    except Exception as e:
        print("scikit-learn manquant :", e); return

    from robust_localization import (load_artifacts, predict_scan,
                                     normalize_bssid, normalize_ssid)

    model_dir = ROOT / "models" / "artifacts_combined"
    if not model_dir.exists():
        print("Modèle introuvable :", model_dir); return
    print("Chargement du modèle :", model_dir.name)
    try:
        art = load_artifacts(model_dir)
    except AttributeError as e:
        print("\n[ERREUR DE VERSION]", e,
              "\n=> Corrige avec :  python -m pip install scikit-learn==1.8.0"); return

    fb, model, le = art["feature_builder"], art["model"], art["label_encoder"]
    salles = ["S102", "P202", "S103", "P104"]
    ok = 0
    print("\nPrédiction sur des scans réels (campagne ALT2) :")
    for room in salles:
        f = ROOT / "data" / "raw_alt2" / f"wifi_{room}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        tcol = [c for c in df.columns if c.lower().strip() == "time"][0]
        one = df[df[tcol] == df[tcol].iloc[len(df) // 2]]   # un instant = un scan
        scan = {}
        for _, r in one.iterrows():
            key = f"{normalize_ssid(str(r['SSID']))}|{normalize_bssid(str(r['BSSID']))}"
            scan[key] = max(scan.get(key, -200.0), float(r["RSSI(dBm)"]))
        res = predict_scan(scan, model, fb, le)
        good = res["predicted_room"] == room
        ok += good
        print(f"  salle {room:5s} -> prédit {res['predicted_room']:5s} "
              f"(confiance {res['confidence']*100:.0f} %)  {'OK' if good else 'X'}")

    print(f"\n{ok}/{len(salles)} correctes.")
    print("Environnement et modèles fonctionnels — tu peux lancer la version live "
          "(python app\\live_app.py)." if ok >= 3 else
          "Chargement OK mais prédictions faibles (drift possible ou données différentes).")


if __name__ == "__main__":
    main()
