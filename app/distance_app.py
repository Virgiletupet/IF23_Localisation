"""Live GUI — estimation de distance a un AP WiFi.

Lance: `python app/distance_app.py`
Affiche en temps reel :
- distance estimee a un AP cible (gros chiffre + barre)
- tableau des distances par AP visible
- salle predite (live, modele de classification)
- slider pour ajuster `n` (path loss exponent)
- bouton de calibration : entrer la vraie distance pour ajuster `n`
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, simpledialog

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import customtkinter as ctk
import numpy as np

from distance_estimator import LogDistanceEstimator, load_distance_artifacts
from robust_localization import load_artifacts, predict_scan, normalize_ssid

from wifi_scan import WiFiScanner, confidence_color, distance_color, rssi_color

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

DISTANCE_DIR = ROOT / "models" / "artifacts_distance"

CLASSIFIER_OPTIONS = {
    "Robust SSID-only (74 features)": ROOT / "models" / "artifacts_robust_ssid",
    "Robust full (529 features)": ROOT / "models" / "artifacts_robust",
    "Advanced v2 (best CV+holdout)": ROOT / "models" / "artifacts_v2",
}


class DistanceApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("IF23 — Distance & Salle Live")
        self.geometry("1200x780")
        self.minsize(1000, 660)

        self.estimator: LogDistanceEstimator | None = None
        self.classifier_assets = None
        self.scanner: WiFiScanner | None = None
        self.running = False
        self.scan_history: deque[dict[str, float]] = deque(maxlen=4)
        self.last_rows: list[dict] = []
        self.last_scan_ssid: dict[str, float] = {}

        self._build_ui()
        self._init_scanner()
        self._load_estimator()
        self._on_classifier_change(self.classifier_var.get())
        self._refresh_target_options()

    # ------------- UI -------------
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0, fg_color="#111827", height=72)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(8, weight=1)
        header.grid_propagate(False)

        ctk.CTkLabel(
            header, text="IF23 — Distance & Salle Live",
            font=("Segoe UI", 19, "bold"),
        ).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        ctk.CTkLabel(header, text="Classifieur:").grid(row=0, column=1, padx=(20, 6))
        self.classifier_var = ctk.StringVar(value=list(CLASSIFIER_OPTIONS.keys())[0])
        self.classifier_menu = ctk.CTkOptionMenu(
            header, values=list(CLASSIFIER_OPTIONS.keys()),
            variable=self.classifier_var, width=260,
            command=self._on_classifier_change,
        )
        self.classifier_menu.grid(row=0, column=2, padx=4)

        ctk.CTkLabel(header, text="n =").grid(row=0, column=3, padx=(16, 4))
        self.n_var = ctk.DoubleVar(value=3.0)
        self.n_slider = ctk.CTkSlider(
            header, from_=1.5, to=5.0, number_of_steps=35,
            variable=self.n_var, width=160,
            command=lambda v: self._update_n_label(),
        )
        self.n_slider.grid(row=0, column=4, padx=4)
        self.n_label = ctk.CTkLabel(header, text="3.00", width=40,
                                     font=("Segoe UI", 12, "bold"))
        self.n_label.grid(row=0, column=5, padx=(0, 12))

        self.start_btn = ctk.CTkButton(
            header, text="Demarrer", command=self.start,
            fg_color="#16a34a", hover_color="#15803d", width=110,
        )
        self.start_btn.grid(row=0, column=6, padx=8)
        self.stop_btn = ctk.CTkButton(
            header, text="Arreter", command=self.stop,
            fg_color="#dc2626", hover_color="#b91c1c", width=110,
            state="disabled",
        )
        self.stop_btn.grid(row=0, column=7, padx=8)

        # Body
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        body.grid_columnconfigure(0, weight=2)
        body.grid_columnconfigure(1, weight=3)
        body.grid_rowconfigure(0, weight=2)
        body.grid_rowconfigure(1, weight=3)

        # ---- Card distance principale ----
        target_card = ctk.CTkFrame(body, corner_radius=14)
        target_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        target_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            target_card, text="AP cible", font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, pady=(14, 4))

        self.target_var = ctk.StringVar(value="(aucun)")
        self.target_menu = ctk.CTkOptionMenu(
            target_card, values=["(aucun)"], variable=self.target_var,
            width=420,
        )
        self.target_menu.grid(row=1, column=0, padx=18, pady=4)

        ctk.CTkLabel(
            target_card, text="Distance estimee",
            font=("Segoe UI", 12), text_color="#9ca3af",
        ).grid(row=2, column=0, pady=(14, 0))
        self.dist_label = ctk.CTkLabel(
            target_card, text="—",
            font=("Segoe UI", 48, "bold"), text_color="#3b82f6",
        )
        self.dist_label.grid(row=3, column=0, pady=(0, 6))

        self.dist_bar = ctk.CTkProgressBar(
            target_card, width=380, height=18,
        )
        self.dist_bar.set(0)
        self.dist_bar.grid(row=4, column=0, padx=18)

        self.target_info = ctk.CTkLabel(
            target_card, text="RSSI: —   |   ref: —",
            font=("Segoe UI", 11), text_color="#d1d5db",
        )
        self.target_info.grid(row=5, column=0, pady=(8, 12))

        ctk.CTkButton(
            target_card, text="Calibrer n a partir d'une distance vraie",
            command=self._calibrate_n,
        ).grid(row=6, column=0, padx=18, pady=(0, 14), sticky="ew")

        # ---- Card salle predite (compact) ----
        room_card = ctk.CTkFrame(body, corner_radius=14)
        room_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        room_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            room_card, text="Salle predite (modele de classification)",
            font=("Segoe UI", 13, "bold"),
        ).grid(row=0, column=0, pady=(14, 6))
        self.room_label = ctk.CTkLabel(
            room_card, text="—",
            font=("Segoe UI", 36, "bold"), text_color="#3b82f6",
        )
        self.room_label.grid(row=1, column=0, pady=(0, 4))
        self.room_conf_bar = ctk.CTkProgressBar(room_card, width=320, height=14)
        self.room_conf_bar.set(0)
        self.room_conf_bar.grid(row=2, column=0, padx=24)
        self.room_conf_label = ctk.CTkLabel(
            room_card, text="Confiance: —", font=("Segoe UI", 12),
        )
        self.room_conf_label.grid(row=3, column=0, pady=(6, 6))

        self.room_top3 = ctk.CTkLabel(
            room_card, text="Top-3: —", font=("Segoe UI", 11), text_color="#9ca3af",
            justify="left", anchor="w",
        )
        self.room_top3.grid(row=4, column=0, padx=18, pady=(0, 14), sticky="ew")

        # ---- Card distances par AP ----
        ap_card = ctk.CTkFrame(body, corner_radius=14)
        ap_card.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(8, 0))
        ap_card.grid_columnconfigure(0, weight=1)
        ap_card.grid_rowconfigure(1, weight=1)

        head_row = ctk.CTkFrame(ap_card, fg_color="transparent")
        head_row.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        head_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            head_row, text="Distances estimees par AP visible",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.ap_count_label = ctk.CTkLabel(
            head_row, text="0 AP", font=("Segoe UI", 12), text_color="#9ca3af",
        )
        self.ap_count_label.grid(row=0, column=1, sticky="e")

        # Header line
        col_header = ctk.CTkFrame(ap_card, fg_color="#0b1220")
        col_header.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 0))
        for i, (txt, w) in enumerate([("SSID", 130), ("BSSID", 160),
                                       ("RSSI", 70), ("ref", 70), ("Distance", 90)]):
            col_header.grid_columnconfigure(i, weight=1)
            ctk.CTkLabel(
                col_header, text=txt, anchor="w", width=w,
                font=("Segoe UI", 11, "bold"), text_color="#9ca3af",
            ).grid(row=0, column=i, padx=(8 if i == 0 else 4), pady=4, sticky="w")

        self.ap_box = ctk.CTkScrollableFrame(ap_card, fg_color="#0b1220")
        self.ap_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.ap_box.grid_columnconfigure(0, weight=1)

        # Footer
        self.status = ctk.CTkLabel(
            self, text="Pret.", anchor="w",
            font=("Segoe UI", 11), text_color="#9ca3af",
            fg_color="#0b1220", height=24,
        )
        self.status.grid(row=2, column=0, sticky="ew")

    # ------------- Init -------------
    def _init_scanner(self) -> None:
        try:
            self.scanner = WiFiScanner()
            self._set_status("Interface WiFi detectee.")
        except Exception as e:
            self._set_status(f"WiFi indispo: {e}")
            messagebox.showerror("WiFi", str(e))

    def _load_estimator(self) -> None:
        try:
            self.estimator = load_distance_artifacts(DISTANCE_DIR)
            self.n_var.set(float(self.estimator.n))
            self._update_n_label()
            self._set_status(
                f"Estimateur charge ({len(self.estimator.known_bssids())} BSSID, n={self.estimator.n:.2f}).",
            )
        except Exception as e:
            self._set_status(f"Echec chargement estimateur: {e}")
            messagebox.showerror(
                "Estimateur",
                f"Impossible de charger {DISTANCE_DIR}.\nExecutez d'abord 09_distance_regression.ipynb.\n\n{e}",
            )

    def _on_classifier_change(self, choice: str) -> None:
        path = CLASSIFIER_OPTIONS[choice]
        try:
            self.classifier_assets = load_artifacts(path)
            self._set_status(f"Classifieur charge: {choice}")
        except Exception as e:
            self.classifier_assets = None
            self._set_status(f"Classifieur indispo: {e}")

    def _refresh_target_options(self) -> None:
        if self.estimator is None:
            return
        opts = []
        for b in self.estimator.known_bssids():
            ssid = self.estimator.ssid_by_bssid.get(b, "")
            opts.append(f"{ssid or '(no ssid)'} | {b}")
        if not opts:
            opts = ["(aucun)"]
        self.target_menu.configure(values=opts)
        self.target_var.set(opts[0])

    def _update_n_label(self) -> None:
        n = float(self.n_var.get())
        self.n_label.configure(text=f"{n:.2f}")
        if self.estimator is not None:
            self.estimator.n = n

    # ------------- Run -------------
    def start(self) -> None:
        if self.scanner is None or self.estimator is None:
            messagebox.showwarning("Indispo", "Scanner ou estimateur non pret.")
            return
        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        threading.Thread(target=self._scan_loop, daemon=True).start()
        self._set_status("Scan en cours...")

    def stop(self) -> None:
        self.running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self._set_status("Arrete.")

    def _scan_loop(self) -> None:
        while self.running:
            try:
                rows = self.scanner.scan_once_by_bssid(wait=2.5)
                # Pour la classification (par SSID), on agrege le top RSSI par SSID
                ssid_scan: dict[str, float] = {}
                for r in rows:
                    s = r["ssid"] or "<hidden>"
                    if s == "<hidden>":
                        continue
                    if s not in ssid_scan or r["rssi"] > ssid_scan[s]:
                        ssid_scan[s] = r["rssi"]
                self.scan_history.append(ssid_scan)
                merged_ssid = self._merge_ssid_history()
                self.last_rows = rows
                self.last_scan_ssid = ssid_scan
                self.after(0, self._refresh_ui, rows, merged_ssid)
            except Exception as e:
                self.after(0, self._set_status, f"Erreur scan: {e}")
                self.running = False
                self.after(0, self.stop)
                return

    def _merge_ssid_history(self) -> dict[str, float]:
        if not self.scan_history:
            return {}
        keys = set()
        for d in self.scan_history:
            keys.update(d.keys())
        return {
            k: float(np.mean([d[k] for d in self.scan_history if k in d]))
            for k in keys
        }

    # ------------- Refresh -------------
    def _refresh_ui(self, rows: list[dict], merged_ssid: dict[str, float]) -> None:
        # Distance par AP
        scan_dict = {r["bssid"]: r["rssi"] for r in rows}
        estimates = self.estimator.estimate_many(scan_dict) if self.estimator else []

        for w in list(self.ap_box.winfo_children()):
            w.destroy()
        for i, e in enumerate(estimates):
            row = ctk.CTkFrame(self.ap_box, fg_color="transparent")
            row.grid(row=i, column=0, sticky="ew", padx=4, pady=1)
            for c in range(5):
                row.grid_columnconfigure(c, weight=(2 if c == 0 else 1))
            ssid_lbl = e["ssid"] or "(no ssid)"
            tag = "" if e["known"] else " ?"
            ctk.CTkLabel(row, text=ssid_lbl + tag, anchor="w", width=130,
                         font=("Segoe UI", 11)).grid(row=0, column=0, padx=(8, 4), sticky="w")
            ctk.CTkLabel(row, text=e["bssid"], anchor="w", width=160,
                         font=("Segoe UI", 10), text_color="#9ca3af").grid(
                row=0, column=1, padx=4, sticky="w")
            ctk.CTkLabel(row, text=f"{int(e['rssi_measured'])}",
                         anchor="e", width=70, font=("Segoe UI", 11, "bold"),
                         text_color=rssi_color(e["rssi_measured"])).grid(
                row=0, column=2, padx=4, sticky="e")
            ctk.CTkLabel(row, text=f"{int(e['rssi_ref'])}", anchor="e", width=70,
                         font=("Segoe UI", 11), text_color="#9ca3af").grid(
                row=0, column=3, padx=4, sticky="e")
            ctk.CTkLabel(row, text=f"{e['distance_m']:.1f} m",
                         anchor="e", width=90, font=("Segoe UI", 11, "bold"),
                         text_color=distance_color(e["distance_m"])).grid(
                row=0, column=4, padx=(4, 8), sticky="e")
        self.ap_count_label.configure(text=f"{len(estimates)} AP")

        # AP cible
        target_label = self.target_var.get()
        if target_label and "|" in target_label:
            target_bssid = target_label.split("|", 1)[1].strip()
            est_for_target = next((e for e in estimates if e["bssid"] == target_bssid), None)
            if est_for_target is not None:
                d = est_for_target["distance_m"]
                self.dist_label.configure(text=f"{d:.1f} m",
                                           text_color=distance_color(d))
                self.dist_bar.set(min(1.0, d / 30.0))
                self.dist_bar.configure(progress_color=distance_color(d))
                self.target_info.configure(
                    text=f"RSSI: {int(est_for_target['rssi_measured'])} dBm   |   "
                         f"ref: {int(est_for_target['rssi_ref'])} dBm",
                )
            else:
                self.dist_label.configure(text="—", text_color="#9ca3af")
                self.dist_bar.set(0)
                self.target_info.configure(text="AP cible non visible dans ce scan")

        # Salle predite
        if self.classifier_assets is not None and merged_ssid:
            try:
                pred = predict_scan(
                    merged_ssid,
                    self.classifier_assets["model"],
                    self.classifier_assets["feature_builder"],
                    self.classifier_assets["label_encoder"],
                    top_k=3,
                )
                room = pred["predicted_room"]
                conf = float(pred["confidence"])
                col = confidence_color(conf)
                self.room_label.configure(text=str(room), text_color=col)
                self.room_conf_bar.set(conf)
                self.room_conf_bar.configure(progress_color=col)
                self.room_conf_label.configure(text=f"Confiance: {conf:.1%}")
                top3_text = "  |  ".join(
                    f"{p['room']} {p['probability']:.0%}" for p in pred["top_predictions"]
                )
                self.room_top3.configure(text=f"Top-3: {top3_text}")
            except Exception as e:
                self.room_label.configure(text="—")
                self.room_conf_label.configure(text=f"Erreur: {e}")

        self._set_status(f"Dernier scan OK — {datetime.now():%H:%M:%S}")

    # ------------- Calibration -------------
    def _calibrate_n(self) -> None:
        if not self.last_rows:
            messagebox.showinfo("Calibration", "Effectuez d'abord un scan.")
            return
        target_label = self.target_var.get()
        if "|" not in target_label:
            messagebox.showinfo("Calibration", "Selectionnez un AP cible.")
            return
        target_bssid = target_label.split("|", 1)[1].strip()
        row = next((r for r in self.last_rows if r["bssid"] == target_bssid), None)
        if row is None:
            messagebox.showinfo("Calibration", "AP cible non visible dans le dernier scan.")
            return
        d_str = simpledialog.askstring(
            "Calibration n",
            f"Distance reelle a l'AP {target_label.split('|')[0].strip()} (en metres) ?",
            parent=self,
        )
        if not d_str:
            return
        try:
            d_true = float(d_str.replace(",", "."))
        except ValueError:
            messagebox.showerror("Calibration", "Valeur invalide.")
            return
        try:
            n_opt = self.estimator.calibrate_n([{
                "bssid": target_bssid, "rssi": row["rssi"], "distance": d_true,
            }])
            self.n_var.set(float(n_opt))
            self._update_n_label()
            self._set_status(f"n calibre : {n_opt:.3f} (1 mesure)")
        except Exception as e:
            messagebox.showerror("Calibration", str(e))

    def _set_status(self, msg: str) -> None:
        self.status.configure(text=msg)


def main() -> None:
    app = DistanceApp()
    app.mainloop()


if __name__ == "__main__":
    main()
