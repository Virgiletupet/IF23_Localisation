"""Scanner WiFi partage par les apps GUI.

Encapsule pywifi. Le format du scan dict envoye au modele depend de l'encoding
des features utilise a l'entrainement :
- modeles SSID-only       (`UTTetudiants`)              : utiliser `to_ssid_dict`
- modeles BSSID-only      (`ac:2a:a1:7e:df:61`)         : utiliser `to_bssid_dict`
- modeles SSID|BSSID      (`utt|ac:2a:a1:7e:df:61`)     : utiliser `to_compound_dict`

`detect_feature_format(feature_builder)` infere automatiquement le bon format
depuis `feature_builder.selected_ssids` (regarde si les cles contiennent `|`,
si elles ressemblent a des MAC, ou si elles sont des SSID nus).
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import pywifi

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from robust_localization import normalize_ssid
from distance_estimator import normalize_bssid


_BSSID_RE = re.compile(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$")


def detect_feature_format(feature_builder) -> str:
    """Retourne 'ssid', 'bssid' ou 'compound' selon les cles connues du builder.

    On regarde un echantillon de `selected_ssids` :
    - s'il contient '|' → compound 'ssid|bssid'
    - s'il matche une MAC → bssid pur
    - sinon → ssid nu
    """
    keys = list(getattr(feature_builder, "selected_ssids", None) or [])
    if not keys:
        return "ssid"
    has_pipe = sum(1 for k in keys[:20] if "|" in k)
    if has_pipe > 0:
        return "compound"
    looks_mac = sum(1 for k in keys[:20] if _BSSID_RE.match(k))
    if looks_mac > len(keys[:20]) // 2:
        return "bssid"
    return "ssid"


class WiFiScanner:
    def __init__(self) -> None:
        wifi = pywifi.PyWiFi()
        ifaces = wifi.interfaces()
        if not ifaces:
            raise RuntimeError("Aucune interface WiFi detectee. Activez le WiFi.")
        self.iface = ifaces[0]

    def _raw_results(self, wait: float):
        self.iface.scan()
        time.sleep(wait)
        return self.iface.scan_results()

    def scan_once_by_bssid(self, wait: float = 2.5) -> list[dict]:
        """Scan brut : une entree par AP avec ssid + bssid + rssi."""
        rows = []
        for net in self._raw_results(wait):
            bssid = normalize_bssid(net.bssid)
            if not bssid:
                continue
            ssid = normalize_ssid(net.ssid)
            rows.append({
                "bssid": bssid,
                "ssid": "" if ssid == "<hidden>" else ssid,
                "rssi": float(net.signal),
            })
        rows.sort(key=lambda r: r["rssi"], reverse=True)
        return rows

    def scan_once_by_ssid(self, wait: float = 2.5) -> dict[str, float]:
        """Agrege par SSID, garde le plus fort RSSI."""
        rows = self.scan_once_by_bssid(wait=wait)
        return to_ssid_dict(rows)


def to_ssid_dict(rows: list[dict]) -> dict[str, float]:
    """Plus fort RSSI par SSID (ssid '<hidden>' / vide ignore)."""
    scan: dict[str, float] = {}
    for r in rows:
        ssid = r.get("ssid") or "<hidden>"
        if ssid in ("", "<hidden>"):
            continue
        rssi = float(r["rssi"])
        if ssid not in scan or rssi > scan[ssid]:
            scan[ssid] = rssi
    return scan


def to_bssid_dict(rows: list[dict]) -> dict[str, float]:
    """Plus fort RSSI par BSSID."""
    scan: dict[str, float] = {}
    for r in rows:
        bssid = r.get("bssid") or ""
        if not bssid:
            continue
        rssi = float(r["rssi"])
        if bssid not in scan or rssi > scan[bssid]:
            scan[bssid] = rssi
    return scan


def to_compound_dict(rows: list[dict]) -> dict[str, float]:
    """Cle = 'ssid|bssid', plus fort RSSI par paire. SSID vide → '<hidden>|bssid'."""
    scan: dict[str, float] = {}
    for r in rows:
        bssid = r.get("bssid") or ""
        if not bssid:
            continue
        ssid = r.get("ssid") or "<hidden>"
        ssid = ssid if ssid else "<hidden>"
        key = f"{ssid}|{bssid}"
        rssi = float(r["rssi"])
        if key not in scan or rssi > scan[key]:
            scan[key] = rssi
    return scan


def scan_for_model(rows: list[dict], feature_builder) -> dict[str, float]:
    """Vectorise un scan brut au format attendu par le feature_builder."""
    fmt = detect_feature_format(feature_builder)
    if fmt == "compound":
        return to_compound_dict(rows)
    if fmt == "bssid":
        return to_bssid_dict(rows)
    return to_ssid_dict(rows)


def confidence_color(conf: float) -> str:
    if conf >= 0.75:
        return "#22c55e"
    if conf >= 0.45:
        return "#f59e0b"
    return "#ef4444"


def rssi_color(rssi: float) -> str:
    if rssi >= -55:
        return "#22c55e"
    if rssi >= -75:
        return "#f59e0b"
    return "#ef4444"


def distance_color(d_meters: float) -> str:
    if d_meters <= 5.0:
        return "#22c55e"
    if d_meters <= 15.0:
        return "#f59e0b"
    return "#ef4444"
