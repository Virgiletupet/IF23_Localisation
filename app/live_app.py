"""Live GUI — localisation indoor en temps reel via scan WiFi.

Lance: `python app/live_app.py`
Dependances: customtkinter, pywifi, joblib, scikit-learn, pandas, numpy.
"""
from __future__ import annotations

import csv
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import customtkinter as ctk
import numpy as np

from robust_localization import load_artifacts, predict_scan
from wifi_scan import (WiFiScanner, confidence_color, rssi_color,
                        scan_for_model, detect_feature_format)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


def _build_artifact_options() -> dict:
    opts = {
        "Combined ALT1+ALT2 (recommande)": ROOT / "models" / "artifacts_combined",
        "ALT2 zones recentes (Avr)": ROOT / "models" / "artifacts_alt2",
        "ALT1 zones fusionnees (Feb)": ROOT / "models" / "artifacts_alt1",
        "Advanced v2 (sous-zones A/B, ALT1)": ROOT / "models" / "artifacts_v2",
        "Robust full (legacy BSSID)": ROOT / "models" / "artifacts_robust",
        "Robust SSID-only (limite, debug)": ROOT / "models" / "artifacts_robust_ssid",
    }
    return {k: v for k, v in opts.items() if (v / "metadata.json").exists()}


ARTIFACT_OPTIONS = _build_artifact_options()


class LiveApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("IF23 — Localisation Live")
        self.geometry("1180x740")
        self.minsize(980, 640)

        self.assets = None
        self.scanner: WiFiScanner | None = None
        self.running = False
        self.scan_history: deque[dict[str, float]] = deque(maxlen=5)
        self.stats = {"total": 0, "correct": 0}
        self.last_scan: dict[str, float] = {}
        self.last_result = None
        self._last_truth_scan_id = -1
        self._scan_id = 0

        self._build_ui()
        self._init_scanner()
        self._load_default_model()

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(self, corner_radius=0, fg_color="#111827", height=70)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(7, weight=1)
        header.grid_propagate(False)

        ctk.CTkLabel(
            header, text="IF23 — Localisation Live",
            font=("Segoe UI", 20, "bold"),
        ).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        ctk.CTkLabel(header, text="Modele:").grid(row=0, column=1, padx=(20, 6))
        self.model_var = ctk.StringVar(value=list(ARTIFACT_OPTIONS.keys())[0])
        self.model_menu = ctk.CTkOptionMenu(
            header, values=list(ARTIFACT_OPTIONS.keys()),
            variable=self.model_var, command=self._on_model_change, width=320,
        )
        self.model_menu.grid(row=0, column=2, padx=6)

        ctk.CTkLabel(header, text="Intervalle (s):").grid(row=0, column=3, padx=(16, 6))
        self.interval_var = ctk.DoubleVar(value=2.5)
        self.interval_slider = ctk.CTkSlider(
            header, from_=1.0, to=8.0, number_of_steps=14,
            variable=self.interval_var, width=140,
            command=lambda v: self.interval_label.configure(text=f"{float(v):.1f}"),
        )
        self.interval_slider.grid(row=0, column=4, padx=6)
        self.interval_label = ctk.CTkLabel(header, text="2.5", width=30)
        self.interval_label.grid(row=0, column=5, padx=(0, 12))

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
        self.stop_btn.grid(row=0, column=7, padx=(0, 20), sticky="e")

        # Body grid
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=2)
        body.grid_rowconfigure(1, weight=3)

        # ---- Card 1 : prediction ----
        pred_card = ctk.CTkFrame(body, corner_radius=14)
        pred_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))
        pred_card.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            pred_card, text="Salle predite",
            font=("Segoe UI", 12), text_color="#9ca3af",
        ).grid(row=0, column=0, pady=(16, 0))
        self.pred_label = ctk.CTkLabel(
            pred_card, text="—",
            font=("Segoe UI", 52, "bold"), text_color="#3b82f6",
        )
        self.pred_label.grid(row=1, column=0, pady=(0, 8))

        self.conf_bar = ctk.CTkProgressBar(
            pred_card, width=440, height=20, progress_color="#22c55e",
        )
        self.conf_bar.set(0)
        self.conf_bar.grid(row=2, column=0, padx=24)
        self.conf_label = ctk.CTkLabel(
            pred_card, text="Confiance: —", font=("Segoe UI", 13),
        )
        self.conf_label.grid(row=3, column=0, pady=(6, 14))

        # Ground truth row
        gt = ctk.CTkFrame(pred_card, fg_color="#0b1220", corner_radius=10)
        gt.grid(row=4, column=0, padx=20, pady=(0, 16), sticky="ew")
        gt.grid_columnconfigure(2, weight=1)
        ctk.CTkLabel(gt, text="Verite terrain:", font=("Segoe UI", 12)).grid(
            row=0, column=0, padx=10, pady=10,
        )
        self.truth_var = ctk.StringVar(value="(non defini)")
        self.truth_menu = ctk.CTkOptionMenu(
            gt, values=["(non defini)"], variable=self.truth_var,
            width=160, command=self._on_truth_change,
        )
        self.truth_menu.grid(row=0, column=1, padx=6, pady=10)
        self.truth_indicator = ctk.CTkLabel(
            gt, text="", font=("Segoe UI", 14, "bold"), width=60,
        )
        self.truth_indicator.grid(row=0, column=2, padx=10)
        self.acc_label = ctk.CTkLabel(
            gt, text="Accuracy live: —", font=("Segoe UI", 12), text_color="#d1d5db",
        )
        self.acc_label.grid(row=0, column=3, padx=10)
        ctk.CTkButton(gt, text="Reset", width=70, command=self._reset_stats).grid(
            row=0, column=4, padx=10, pady=10,
        )

        # ---- Card 2 : top-K ----
        topk_card = ctk.CTkFrame(body, corner_radius=14)
        topk_card.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=(8, 0))
        topk_card.grid_columnconfigure(0, weight=1)
        topk_card.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            topk_card, text="Top-5 predictions", font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, pady=(14, 8))
        topk_inner = ctk.CTkFrame(topk_card, fg_color="transparent")
        topk_inner.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        topk_inner.grid_columnconfigure(1, weight=1)
        self.topk_rows = []
        for i in range(5):
            name = ctk.CTkLabel(
                topk_inner, text="—", anchor="w", width=140,
                font=("Segoe UI", 13, "bold"),
            )
            name.grid(row=i, column=0, padx=(0, 8), pady=8, sticky="w")
            bar = ctk.CTkProgressBar(topk_inner, height=14)
            bar.set(0)
            bar.grid(row=i, column=1, padx=4, pady=8, sticky="ew")
            pct = ctk.CTkLabel(
                topk_inner, text="—", width=60,
                font=("Segoe UI", 12), text_color="#9ca3af",
            )
            pct.grid(row=i, column=2, padx=4, pady=8, sticky="e")
            self.topk_rows.append((name, bar, pct))

        # ---- Card 3 : SSID list ----
        ssid_card = ctk.CTkFrame(body, corner_radius=14)
        ssid_card.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        ssid_card.grid_columnconfigure(0, weight=1)
        ssid_card.grid_rowconfigure(1, weight=1)

        head_row = ctk.CTkFrame(ssid_card, fg_color="transparent")
        head_row.grid(row=0, column=0, sticky="ew", padx=14, pady=(14, 6))
        head_row.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            head_row, text="SSID visibles (scan courant)",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, sticky="w")
        self.ssid_count_label = ctk.CTkLabel(
            head_row, text="0 SSID", font=("Segoe UI", 12), text_color="#9ca3af",
        )
        self.ssid_count_label.grid(row=0, column=1, sticky="e", padx=10)
        ctk.CTkButton(
            head_row, text="Snapshot CSV", width=140, command=self._save_snapshot,
        ).grid(row=0, column=2, padx=(10, 0))

        self.ssid_box = ctk.CTkScrollableFrame(ssid_card, fg_color="#0b1220")
        self.ssid_box.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.ssid_box.grid_columnconfigure(0, weight=1)

        # Footer
        self.status = ctk.CTkLabel(
            self, text="Pret.", anchor="w",
            font=("Segoe UI", 11), text_color="#9ca3af",
            fg_color="#0b1220", height=24,
        )
        self.status.grid(row=2, column=0, sticky="ew")

    # ---------------- Init ----------------
    def _init_scanner(self) -> None:
        try:
            self.scanner = WiFiScanner()
            self._set_status("Interface WiFi detectee.")
        except Exception as e:
            self._set_status(f"WiFi indispo: {e}")
            messagebox.showerror("WiFi", str(e))

    def _load_default_model(self) -> None:
        self._on_model_change(self.model_var.get())

    def _on_model_change(self, choice: str) -> None:
        path = ARTIFACT_OPTIONS[choice]
        try:
            self.assets = load_artifacts(path)
            classes = list(self.assets["label_encoder"].classes_)
            fmt = detect_feature_format(self.assets["feature_builder"])
            self.feature_format = fmt
            self.truth_menu.configure(values=["(non defini)"] + classes)
            self._set_status(
                f"Modele charge: {choice} | {len(classes)} classes | format scan: {fmt}",
            )
        except Exception as e:
            self._set_status(f"Echec chargement modele: {e}")
            messagebox.showerror("Modele", str(e))

    def _on_truth_change(self, _value: str) -> None:
        self._reset_stats()

    # ---------------- Scan loop ----------------
    def start(self) -> None:
        if self.scanner is None or self.assets is None:
            messagebox.showwarning("Indispo", "Scanner ou modele non pret.")
            return
        self.running = True
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.model_menu.configure(state="disabled")
        threading.Thread(target=self._scan_loop, daemon=True).start()
        self._set_status("Scan en cours...")

    def stop(self) -> None:
        self.running = False
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.model_menu.configure(state="normal")
        self._set_status("Arrete.")

    def _scan_loop(self) -> None:
        while self.running:
            try:
                interval = float(self.interval_var.get())
                wait = max(1.0, interval - 0.2)
                rows = self.scanner.scan_once_by_bssid(wait=wait)
                # Format scan dict pour le modele en cours (ssid / bssid / compound)
                scan = scan_for_model(rows, self.assets["feature_builder"])
                # Pour l'affichage des SSID visibles : agrege par SSID
                ssid_for_display = {}
                for r in rows:
                    s = r.get("ssid") or "<hidden>"
                    if s in ("", "<hidden>"):
                        continue
                    if s not in ssid_for_display or r["rssi"] > ssid_for_display[s]:
                        ssid_for_display[s] = float(r["rssi"])
                self.scan_history.append(scan)
                merged = self._merge_history()
                self.last_scan = ssid_for_display
                if not merged:
                    time.sleep(0.4)
                    continue
                result = predict_scan(
                    merged,
                    self.assets["model"],
                    self.assets["feature_builder"],
                    self.assets["label_encoder"],
                    top_k=5,
                )
                # Compute coverage : combien de cles connues du modele sont vues
                known = set(self.assets["feature_builder"].selected_ssids or [])
                seen_known = sum(1 for k in scan if k in known)
                result["_coverage"] = (seen_known, len(known))
                self.last_result = result
                self._scan_id += 1
                self.after(0, self._refresh_ui, ssid_for_display, result, self._scan_id)
            except Exception as e:
                self.after(0, self._set_status, f"Erreur scan: {e}")
                self.running = False
                self.after(0, self.stop)
                return

    def _merge_history(self) -> dict[str, float]:
        if not self.scan_history:
            return {}
        keys: set[str] = set()
        for d in self.scan_history:
            keys.update(d.keys())
        return {
            k: float(np.mean([d[k] for d in self.scan_history if k in d]))
            for k in keys
        }

    # ---------------- UI refresh ----------------
    def _refresh_ui(self, scan: dict[str, float], result, scan_id: int) -> None:
        room = result["predicted_room"]
        conf = float(result["confidence"])

        col = confidence_color(conf)
        self.pred_label.configure(text=str(room), text_color=col)
        self.conf_bar.set(conf)
        self.conf_bar.configure(progress_color=col)
        self.conf_label.configure(text=f"Confiance: {conf:.1%}")

        # Top-5
        top = result["top_predictions"]
        for (name_lbl, bar, pct_lbl), item in zip(self.topk_rows, top):
            name_lbl.configure(text=str(item["room"]))
            p = float(item["probability"])
            bar.set(p)
            bar.configure(progress_color=confidence_color(p))
            pct_lbl.configure(text=f"{p:.1%}")
        for j in range(len(top), 5):
            n, b, p = self.topk_rows[j]
            n.configure(text="—")
            b.set(0)
            p.configure(text="—")

        # Ground truth (compte une fois par scan)
        truth = self.truth_var.get()
        if truth and truth != "(non defini)" and scan_id != self._last_truth_scan_id:
            self._last_truth_scan_id = scan_id
            self.stats["total"] += 1
            if truth == room:
                self.stats["correct"] += 1
                self.truth_indicator.configure(text="OK", text_color="#22c55e")
            else:
                self.truth_indicator.configure(text="KO", text_color="#ef4444")
            acc = self.stats["correct"] / self.stats["total"]
            self.acc_label.configure(
                text=f"Accuracy live: {acc:.1%}  ({self.stats['correct']}/{self.stats['total']})",
            )
        elif not truth or truth == "(non defini)":
            self.truth_indicator.configure(text="")
            self.acc_label.configure(text="Accuracy live: —")

        # SSID list
        for w in list(self.ssid_box.winfo_children()):
            w.destroy()
        sorted_ssids = sorted(scan.items(), key=lambda kv: kv[1], reverse=True)
        for i, (ssid, rssi) in enumerate(sorted_ssids):
            row = ctk.CTkFrame(self.ssid_box, fg_color="transparent")
            row.grid(row=i, column=0, sticky="ew", padx=4, pady=1)
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(
                row, text=ssid, anchor="w", font=("Segoe UI", 11),
            ).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(
                row, text=f"{int(rssi)} dBm", anchor="e", width=80,
                font=("Segoe UI", 11, "bold"), text_color=rssi_color(rssi),
            ).grid(row=0, column=1, sticky="e")
        self.ssid_count_label.configure(text=f"{len(sorted_ssids)} SSID")

        self._set_status(f"Dernier scan OK — {datetime.now():%H:%M:%S}")

    # ---------------- Actions ----------------
    def _reset_stats(self) -> None:
        self.stats = {"total": 0, "correct": 0}
        self.acc_label.configure(text="Accuracy live: —")
        self.truth_indicator.configure(text="")
        self._last_truth_scan_id = -1

    def _save_snapshot(self) -> None:
        if not self.last_scan:
            messagebox.showinfo("Snapshot", "Aucun scan disponible.")
            return
        room = self.truth_var.get()
        default = (
            "snapshot.csv"
            if room in (None, "(non defini)")
            else f"wifi_{room}_{datetime.now():%Y%m%d_%H%M%S}.csv"
        )
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=default,
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["BSSID", "SSID", "RSSI(dBm)", "Time"])
            for ssid, rssi in sorted(self.last_scan.items(), key=lambda kv: kv[1], reverse=True):
                w.writerow(["", ssid, int(rssi), ts])
        self._set_status(f"Snapshot enregistre: {path}")

    def _set_status(self, msg: str) -> None:
        self.status.configure(text=msg)


def main() -> None:
    app = LiveApp()
    app.mainloop()


if __name__ == "__main__":
    main()
