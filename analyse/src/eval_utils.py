"""
eval_utils.py — utilitaires partagés d'évaluation honnête (ALT3).
Centralise la construction des jeux de données et les protocoles de
séparation temporelle (anti-fuite) utilisés par les scripts 02-09.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from robust_localization import (
    RobustFeatureBuilder,
    build_snapshot_tables,
    load_raw_wifi_data,
)

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
UNIFIED = ROOT / "data" / "cleaned" / "dataset_unified.csv"


def build_dataset(scope: str = "bssid", max_ids: int = 120, min_freq: int = 5):
    """scope='bssid' (par AP physique) ou 'ssid' (par nom de réseau)."""
    raw = load_raw_wifi_data(RAW_DIR)
    bssid_to_ssid = (raw.groupby("bssid")["ssid"]
                     .agg(lambda s: s.value_counts().index[0]).to_dict())
    if scope == "bssid":
        raw = raw.copy()
        raw["ssid"] = raw["bssid"]
    per_ssid, snapshot = build_snapshot_tables(raw)
    builder = RobustFeatureBuilder(max_ssids=max_ids, min_ssid_frequency=min_freq)
    X, y = builder.fit_transform(per_ssid, snapshot)
    return X, y, builder, bssid_to_ssid


def build_alt1_dataset():
    """Pipeline ALT1 : dataset_unified.csv, 6 BSSID communs, pivot (Room,Time)."""
    df = pd.read_csv(UNIFIED)
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    piv = df.pivot_table(index=["Room", "Time"], columns="BSSID",
                         values="RSSI(dBm)", aggfunc="mean").fillna(-100.0)
    piv.index = piv.index.set_names(["room", "time"])
    y = piv.index.get_level_values("room")
    return piv, pd.Series(y, index=piv.index, name="room")


def time_blocks(index, n_blocks=5):
    """Identifiant de bloc temporel 'room__bk' par snapshot (groupes group-aware)."""
    df = pd.DataFrame({"room": index.get_level_values("room"),
                       "time": pd.to_datetime(index.get_level_values("time"), errors="coerce")})
    groups = np.empty(len(df), dtype=object)
    for room, sub in df.groupby("room"):
        order = sub.sort_values("time").index.to_numpy()
        for b, idx in enumerate(np.array_split(order, min(n_blocks, len(order)))):
            for pos in idx:
                groups[pos] = f"{room}__b{b}"
    return groups


def temporal_mask(index, frac=0.7):
    """Masque booléen train : les frac% premiers snapshots (temps) par zone."""
    df = pd.DataFrame({"room": index.get_level_values("room"),
                       "time": pd.to_datetime(index.get_level_values("time"), errors="coerce")})
    df = df.reset_index(drop=True)
    mask = np.zeros(len(df), dtype=bool)
    for room, sub in df.groupby("room"):
        order = sub.sort_values("time").index.to_numpy()
        mask[order[:int(round(len(order) * frac))]] = True
    return mask


def floor_of(zone: str) -> str:
    """Étage déduit du nom de zone (pour le pipeline hybride hiérarchique)."""
    z = str(zone)
    if "Pbas" in z or "pbas" in z or "Pbas" in z or "pallierPbas" in z:
        return "RDC"
    if z.startswith(("S1", "P1")) or z == "PALIER1ER":
        return "Etage1"
    if z.startswith(("S2", "P2")) or z == "PALIER2E":
        return "Etage2"
    if z in ("couloirB",):
        return "Etage1"
    if z == "couloirB(2e)":
        return "Etage2"
    return "Autre"
