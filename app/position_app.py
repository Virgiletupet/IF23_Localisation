"""Live GUI — prediction des coordonnees (x, y) dans une zone.

Conforme au PDF IF23 P (page 5) : entree = vecteur RSSI par BSSID,
sortie = position (x, y) dans le repere de la zone.

Lance: `python app/position_app.py`
"""
from __future__ import annotations

import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from tkinter import messagebox

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

import customtkinter as ctk
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from intra_zone_regression import load_regression_artifacts, normalize_bssid
from wifi_scan import WiFiScanner, rssi_color

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ARTIFACT_DIR = ROOT / "models" / "artifacts_regression"


class PositionApp(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("IF23 — Position (x, y) intra-zone")
        self.geometry("1200x780")
        self.minsize(1000, 660)

        self.regressor = None
        self.scanner: WiFiScanner | None = None
        self.running = False
        self.scan_history: deque[dict[str, float]] = deque(maxlen=4)
        self.last_pred: tuple[float, float] | None = None
        self.last_scan: list[dict] = []

        self._build_ui()
        self._init_scanner()
        self._load_regressor()
        self._draw_zone()

    # ---------- UI ----------
    def _build_ui(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(self, corner_radius=0, fg_color="#111827", height=72)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(5, weight=1)
        header.grid_propagate(False)

        ctk.CTkLabel(
            header, text="IF23 — Position (x, y) intra-zone",
            font=("Segoe UI", 19, "bold"),
        ).grid(row=0, column=0, padx=20, pady=18, sticky="w")

        ctk.CTkLabel(header, text="Intervalle (s):").grid(row=0, column=1, padx=(20, 6))
        self.interval_var = ctk.DoubleVar(value=2.5)
        self.interval_slider = ctk.CTkSlider(
            header, from_=1.0, to=8.0, number_of_steps=14,
            variable=self.interval_var, width=160,
            command=lambda v: self.interval_label.configure(text=f"{float(v):.1f}"),
        )
        self.interval_slider.grid(row=0, column=2, padx=4)
        self.interval_label = ctk.CTkLabel(header, text="2.5", width=30)
        self.interval_label.grid(row=0, column=3, padx=(0, 12))

        self.start_btn = ctk.CTkButton(
            header, text="Demarrer", command=self.start,
            fg_color="#16a34a", hover_color="#15803d", width=110,
        )
        self.start_btn.grid(row=0, column=4, padx=8)
        self.stop_btn = ctk.CTkButton(
            header, text="Arreter", command=self.stop,
            fg_color="#dc2626", hover_color="#b91c1c", width=110,
            state="disabled",
        )
        self.stop_btn.grid(row=0, column=5, padx=(0, 20), sticky="e")

        # Body
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.grid(row=1, column=0, sticky="nsew", padx=12, pady=12)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=2)
        body.grid_rowconfigure(0, weight=1)

        # ---- Card carte ----
        map_card = ctk.CTkFrame(body, corner_radius=14)
        map_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        map_card.grid_columnconfigure(0, weight=1)
        map_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            map_card, text="Position dans la zone",
            font=("Segoe UI", 14, "bold"),
        ).grid(row=0, column=0, pady=(14, 4))

        self.fig = Figure(figsize=(6, 5.5), dpi=100, facecolor="#1f2937")
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=map_card)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew",
                                          padx=14, pady=(0, 14))

        # ---- Card details ----
        info_card = ctk.CTkFrame(body, corner_radius=14)
        info_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        info_card.grid_columnconfigure(0, weight=1)
        info_card.grid_rowconfigure(5, weight=1)

        ctk.CTkLabel(info_card, text="Coordonnees predites",
                     font=("Segoe UI", 12), text_color="#9ca3af").grid(
            row=0, column=0, pady=(14, 0))
        self.coord_label = ctk.CTkLabel(
            info_card, text="(—, —)",
            font=("Consolas", 36, "bold"), text_color="#3b82f6",
        )
        self.coord_label.grid(row=1, column=0, pady=(0, 8))

        self.bounds_label = ctk.CTkLabel(
            info_card, text="Zone: —", font=("Segoe UI", 11), text_color="#9ca3af",
        )
        self.bounds_label.grid(row=2, column=0, pady=(0, 12))

        self.coverage_label = ctk.CTkLabel(
            info_card, text="BSSID connus visibles: —",
            font=("Segoe UI", 12),
        )
        self.coverage_label.grid(row=3, column=0, pady=(0, 8))

        ctk.CTkLabel(info_card, text="BSSID visibles (top RSSI)",
                     font=("Segoe UI", 13, "bold")).grid(row=4, column=0,
                                                          pady=(8, 6))

        self.bssid_box = ctk.CTkScrollableFrame(info_card, fg_color="#0b1220")
        self.bssid_box.grid(row=5, column=0, sticky="nsew", padx=14, pady=(0, 14))
        self.bssid_box.grid_columnconfigure(0, weight=1)

        # Footer
        self.status = ctk.CTkLabel(
            self, text="Pret.", anchor="w",
            font=("Segoe UI", 11), text_color="#9ca3af",
            fg_color="#0b1220", height=24,
        )
        self.status.grid(row=2, column=0, sticky="ew")

    # ---------- Init ----------
    def _init_scanner(self) -> None:
        try:
            self.scanner = WiFiScanner()
            self._set_status("Interface WiFi detectee.")
        except Exception as e:
            self._set_status(f"WiFi indispo: {e}")
            messagebox.showerror("WiFi", str(e))

    def _load_regressor(self) -> None:
        try:
            self.regressor = load_regression_artifacts(ARTIFACT_DIR)
            xb = self.regressor.bounds.get("x", (None, None))
            yb = self.regressor.bounds.get("y", (None, None))
            self.bounds_label.configure(
                text=f"Zone: X ∈ [{xb[0]:.1f}, {xb[1]:.1f}] m, "
                     f"Y ∈ [{yb[0]:.1f}, {yb[1]:.1f}] m  |  "
                     f"{len(self.regressor.bssid_columns)} BSSID",
            )
            self._set_status(f"Modele charge ({len(self.regressor.bssid_columns)} BSSID, "
                              f"zone {xb[1]-xb[0]:.1f}×{yb[1]-yb[0]:.1f} m).")
        except Exception as e:
            self._set_status(f"Echec chargement: {e}")
            messagebox.showerror(
                "Modele",
                f"Impossible de charger {ARTIFACT_DIR}.\n\n"
                f"Executez d'abord 10_intra_zone_regression.ipynb.\n\n{e}",
            )

    # ---------- Map ----------
    def _draw_zone(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#1f2937")
        if self.regressor is None:
            self.ax.text(0.5, 0.5, "Modele non charge",
                         ha="center", va="center",
                         transform=self.ax.transAxes, color="#9ca3af")
            self.canvas.draw()
            return
        xb = self.regressor.bounds.get("x", (0, 1))
        yb = self.regressor.bounds.get("y", (0, 1))
        self.ax.set_xlim(xb[0] - 0.5, xb[1] + 0.5)
        self.ax.set_ylim(yb[0] - 0.5, yb[1] + 0.5)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.2, color="#9ca3af")

        from matplotlib.patches import Rectangle
        zone_rect = Rectangle((xb[0], yb[0]), xb[1] - xb[0], yb[1] - yb[0],
                              facecolor="#3b82f6", alpha=0.08,
                              edgecolor="#3b82f6", linewidth=1.5)
        self.ax.add_patch(zone_rect)

        if self.last_pred is not None:
            x, y = self.last_pred
            self.ax.scatter([x], [y], s=360, marker="o",
                            facecolor="#22c55e", edgecolor="white", linewidth=2.5,
                            zorder=5)
            self.ax.annotate(f"({x:.2f}, {y:.2f})",
                             (x, y), fontsize=11, color="white",
                             xytext=(10, 10), textcoords="offset points",
                             bbox=dict(boxstyle="round", facecolor="#22c55e",
                                       edgecolor="none", alpha=0.85))

        self.ax.set_xlabel("X (m)", color="#d1d5db")
        self.ax.set_ylabel("Y (m)", color="#d1d5db")
        self.ax.tick_params(colors="#d1d5db")
        for spine in self.ax.spines.values():
            spine.set_color("#374151")
        self.ax.set_title("Position predite dans la zone", color="white")
        self.fig.tight_layout()
        self.canvas.draw()

    # ---------- Run ----------
    def start(self) -> None:
        if self.scanner is None or self.regressor is None:
            messagebox.showwarning("Indispo", "Scanner ou modele non pret.")
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
                interval = float(self.interval_var.get())
                wait = max(1.0, interval - 0.2)
                rows = self.scanner.scan_once_by_bssid(wait=wait)
                self.last_scan = rows
                # Agrege par BSSID en gardant le RSSI max
                scan_dict: dict[str, float] = {}
                for r in rows:
                    b = normalize_bssid(r["bssid"])
                    if b not in scan_dict or r["rssi"] > scan_dict[b]:
                        scan_dict[b] = r["rssi"]
                self.scan_history.append(scan_dict)
                merged = self._merge_history()
                self.after(0, self._refresh_ui, rows, merged)
            except Exception as e:
                self.after(0, self._set_status, f"Erreur scan: {e}")
                self.running = False
                self.after(0, self.stop)
                return

    def _merge_history(self) -> dict[str, float]:
        if not self.scan_history:
            return {}
        keys = set()
        for d in self.scan_history:
            keys.update(d.keys())
        return {k: float(np.mean([d[k] for d in self.scan_history if k in d]))
                for k in keys}

    # ---------- Refresh ----------
    def _refresh_ui(self, rows: list[dict], merged: dict[str, float]) -> None:
        if not merged or self.regressor is None:
            return
        try:
            x, y = self.regressor.predict(merged)
            self.last_pred = (x, y)
            visible_known = self.regressor.visible_known_bssids(merged)
        except Exception as e:
            self._set_status(f"Erreur prediction: {e}")
            return

        self.coord_label.configure(text=f"({x:+.2f}, {y:+.2f}) m")
        coverage_pct = visible_known / max(1, len(self.regressor.bssid_columns))
        col = "#22c55e" if coverage_pct >= 0.5 else ("#f59e0b" if coverage_pct >= 0.25 else "#ef4444")
        self.coverage_label.configure(
            text=f"BSSID connus visibles: {visible_known} / {len(self.regressor.bssid_columns)} "
                 f"({coverage_pct:.0%})",
            text_color=col,
        )

        # BSSID list (top par RSSI)
        for w in list(self.bssid_box.winfo_children()):
            w.destroy()
        sorted_rows = sorted(rows, key=lambda r: r["rssi"], reverse=True)[:25]
        known_set = set(self.regressor.bssid_columns)
        for i, r in enumerate(sorted_rows):
            row = ctk.CTkFrame(self.bssid_box, fg_color="transparent")
            row.grid(row=i, column=0, sticky="ew", padx=4, pady=1)
            row.grid_columnconfigure(0, weight=1)
            tag = " ✓" if r["bssid"] in known_set else ""
            label = (r["ssid"] or "(no ssid)") + tag
            ctk.CTkLabel(
                row, text=f"{label:30.30s}", anchor="w",
                font=("Segoe UI", 10),
                text_color="#22c55e" if tag else "#d1d5db",
            ).grid(row=0, column=0, sticky="w", padx=(8, 4))
            ctk.CTkLabel(
                row, text=r["bssid"], anchor="w", width=140,
                font=("Consolas", 9), text_color="#9ca3af",
            ).grid(row=0, column=1, sticky="w", padx=4)
            ctk.CTkLabel(
                row, text=f"{int(r['rssi'])} dBm", anchor="e", width=80,
                font=("Segoe UI", 10, "bold"),
                text_color=rssi_color(r["rssi"]),
            ).grid(row=0, column=2, sticky="e", padx=(4, 8))

        self._draw_zone()
        self._set_status(f"Dernier scan OK — {datetime.now():%H:%M:%S}")

    def _set_status(self, msg: str) -> None:
        self.status.configure(text=msg)


def main() -> None:
    app = PositionApp()
    app.mainloop()


if __name__ == "__main__":
    main()
