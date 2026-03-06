"""
ui_detector.py
──────────────────────────────────────────────────────────────────
Premium real-time motor anomaly detection dashboard.
Runs the same detection logic as demo_real_time_detector.py in a
background thread and visualises everything live in a sleek dark GUI.

Dependencies (all already in the project .venv):
    numpy, pandas, joblib, tf_keras, matplotlib
No extra installs needed — tkinter ships with Python.
──────────────────────────────────────────────────────────────────
"""

import os, sys, warnings, threading, time, queue
from pathlib import Path
from collections import deque
import tkinter as tk
from tkinter import ttk, font as tkfont

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).parent

# ── Colours / Theme ───────────────────────────────────────────────────────────
BG        = "#0d1117"
PANEL     = "#161b22"
BORDER    = "#21262d"
ACCENT    = "#58a6ff"
GREEN     = "#3fb950"
RED       = "#f85149"
YELLOW    = "#d29922"
TEXT      = "#e6edf3"
SUBTEXT   = "#8b949e"
CHART_BG  = "#0d1117"

# ── Constants ─────────────────────────────────────────────────────────────────
THRESHOLD       = 0.0628
SEQUENCE_LENGTH = 30
DELAY           = 0.08          # simulated sensor step delay (s)
HISTORY         = 120           # data points kept on charts
ANOMALY_START   = 100
ANOMALY_LEN     = 80

SENSOR_LABELS  = ["AX", "AY", "AZ", "GX", "GY", "GZ", "Temp", "Flame"]
SENSOR_COLORS  = ["#58a6ff", "#3fb950", "#f0883e", "#bc8cff",
                  "#ff7b72", "#79c0ff", "#ffa657", "#d2a8ff"]

# ── Global state shared between detector thread and GUI ───────────────────────
data_q: "queue.Queue[dict]" = queue.Queue()   # detector → GUI updates


# ══════════════════════════════════════════════════════════════════════════════
#  DETECTOR THREAD
# ══════════════════════════════════════════════════════════════════════════════
def run_detector(stop_event: threading.Event):
    """Runs demo detection loop; pushes results into data_q."""
    try:
        import tf_keras as keras
    except ImportError:
        data_q.put({"error": "tf_keras not found. Activate the project venv."})
        return

    data_q.put({"log": "Loading model…"})
    model  = keras.models.load_model(_HERE / "motor_anomaly_lstm.keras")
    scaler = joblib.load(_HERE / "scaler.save")
    data_q.put({"log": "Model loaded ✓"})

    df = pd.read_csv(_HERE / "motor_normal_data.csv")
    cols = ["ax", "ay", "az", "gx", "gy", "gz", "temp", "flame"]
    normal_rows = df[cols].values.tolist()

    anomaly_row = [32767.0, -32768.0, 32767.0, 16000.0, -16000.0, 16000.0, 500.0, 0.0]
    sequence    = []
    announced   = False

    for step, row in enumerate(normal_rows):
        if stop_event.is_set():
            break

        in_anomaly = ANOMALY_START <= step < ANOMALY_START + ANOMALY_LEN
        use_row = anomaly_row if in_anomaly else row

        if in_anomaly and not announced:
            data_q.put({"log": f">>> Injecting motor fault at step {step} <<<"})
            announced = True

        values_scaled = scaler.transform(np.array(use_row, dtype=float).reshape(1, -1))[0]
        if in_anomaly:
            values_scaled = np.array([3.5, -2.5, 3.5, 2.5, -2.5, 2.5, 6.0, -3.0])

        sequence.append(values_scaled)
        if len(sequence) > SEQUENCE_LENGTH:
            sequence.pop(0)

        if len(sequence) == SEQUENCE_LENGTH:
            X             = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 8)
            reconstruction = model.predict(X, verbose=0)
            mse            = float(np.mean(np.square(X - reconstruction)))
            is_anomaly     = mse > THRESHOLD

            data_q.put({
                "step"      : step,
                "mse"       : mse,
                "anomaly"   : is_anomaly,
                "sensors"   : list(use_row),
                "log"       : (f"[{step:>4}] MSE={mse:.6f}  "
                               f"{'⚠ ANOMALY' if is_anomaly else '✓ Normal'}"),
            })

        time.sleep(DELAY)

    data_q.put({"log": "── Demo finished ──", "done": True})


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN GUI CLASS
# ══════════════════════════════════════════════════════════════════════════════
class AnomalyDashboard(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("Motor Anomaly Detection Dashboard")
        self.configure(bg=BG)
        self.geometry("1280x760")
        self.minsize(1100, 650)

        self._stop_event  = threading.Event()
        self._thread: threading.Thread | None = None
        self._running     = False

        # Rolling history buffers
        self.mse_history     = deque([0.0] * HISTORY, maxlen=HISTORY)
        self.sensor_history  = [deque([0.0] * HISTORY, maxlen=HISTORY)
                                for _ in range(8)]
        self.step_counter    = 0
        self.anomaly_count   = 0
        self.normal_count    = 0

        # Widget stubs — assigned inside _build_* helpers (declared here for Pyright)
        self._btn:         tk.Button
        self._status_var:  tk.StringVar
        self._state_label: tk.Label
        self._mse_var:     tk.StringVar
        self._step_var:    tk.StringVar
        self._norm_var:    tk.StringVar
        self._anom_var:    tk.StringVar
        self._sensor_vars: list[tk.StringVar]
        self._log:         tk.Text
        self._ax_mse:      plt.Axes                  # type: ignore[name-defined]
        self._mse_line:    plt.Line2D                # type: ignore[name-defined]
        self._mse_fill:    object
        self._thresh_line: plt.Line2D                # type: ignore[name-defined]
        self._mse_canvas:  FigureCanvasTkAgg
        self._ax_sen:      plt.Axes                  # type: ignore[name-defined]
        self._sen_lines:   list[plt.Line2D]          # type: ignore[name-defined]
        self._sen_canvas:  FigureCanvasTkAgg

        self._build_ui()
        self._start_animation()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI layout ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        # ---- top header bar --------------------------------------------------
        hdr = tk.Frame(self, bg=PANEL, height=56)
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⚙  Motor Health Monitoring",
                 bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 18, "bold")).pack(side="left", padx=20, pady=10)

        tk.Label(hdr, text="LSTM Anomaly Detection  ·  Real-Time Demo",
                 bg=PANEL, fg=SUBTEXT,
                 font=("Segoe UI", 11)).pack(side="left", padx=8)

        # start / stop button
        self._btn = tk.Button(hdr, text="▶  Start",
                              bg=GREEN, fg="#0d1117",
                              activebackground="#2ea043",
                              font=("Segoe UI", 11, "bold"),
                              bd=0, padx=18, pady=6, cursor="hand2",
                              command=self._toggle_run)
        self._btn.pack(side="right", padx=16, pady=10)

        # ---- status ticker ---------------------------------------------------
        self._status_var = tk.StringVar(value="Idle — press Start")
        ticker = tk.Label(self, textvariable=self._status_var,
                          bg="#0d1117", fg=SUBTEXT,
                          font=("Segoe UI", 10), anchor="w")
        ticker.pack(fill="x", side="top", padx=18, pady=(6, 0))

        # ---- main body -------------------------------------------------------
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=8)

        # left column — status card + sensor gauges + log
        left = tk.Frame(body, bg=BG, width=290)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)

        self._build_status_card(left)
        self._build_sensor_grid(left)
        self._build_log_panel(left)

        # right column — charts
        right = tk.Frame(body, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        self._build_mse_chart(right)
        self._build_sensor_chart(right)

    # ── Status card ───────────────────────────────────────────────────────────
    def _build_status_card(self, parent):
        card = tk.Frame(parent, bg=PANEL, bd=0,
                        highlightbackground=BORDER, highlightthickness=1)
        card.pack(fill="x", pady=(0, 8))

        self._state_label = tk.Label(card, text="● IDLE",
                                     bg=PANEL, fg=SUBTEXT,
                                     font=("Segoe UI", 20, "bold"))
        self._state_label.pack(pady=(14, 4))

        row = tk.Frame(card, bg=PANEL)
        row.pack(pady=(0, 12))

        self._mse_var = tk.StringVar(value="MSE: —")
        tk.Label(row, textvariable=self._mse_var,
                 bg=PANEL, fg=TEXT,
                 font=("Segoe UI", 12)).pack(side="left", padx=14)

        self._step_var = tk.StringVar(value="Step: —")
        tk.Label(row, textvariable=self._step_var,
                 bg=PANEL, fg=SUBTEXT,
                 font=("Segoe UI", 11)).pack(side="left")

        # stats row
        stats = tk.Frame(card, bg=PANEL)
        stats.pack(fill="x", padx=10, pady=(0, 12))

        self._norm_var = tk.StringVar(value="0")
        self._anom_var = tk.StringVar(value="0")

        for label, var, color in [("Normal", self._norm_var, GREEN),
                                   ("Anomalies", self._anom_var, RED)]:
            col = tk.Frame(stats, bg=BORDER)
            col.pack(side="left", expand=True, fill="x", padx=3)
            tk.Label(col, textvariable=var,
                     bg=BORDER, fg=color,
                     font=("Segoe UI", 16, "bold")).pack(pady=(8, 2))
            tk.Label(col, text=label,
                     bg=BORDER, fg=SUBTEXT,
                     font=("Segoe UI", 9)).pack(pady=(0, 8))

    # ── Sensor gauge grid ─────────────────────────────────────────────────────
    def _build_sensor_grid(self, parent):
        outer = tk.Frame(parent, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        outer.pack(fill="x", pady=(0, 8))

        tk.Label(outer, text="Live Sensor Values",
                 bg=PANEL, fg=SUBTEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(8, 4))

        grid = tk.Frame(outer, bg=PANEL)
        grid.pack(fill="x", padx=6, pady=(0, 8))

        self._sensor_vars = []
        for i, (lbl, col) in enumerate(zip(SENSOR_LABELS, SENSOR_COLORS)):
            r, c = divmod(i, 2)
            cell = tk.Frame(grid, bg=BORDER)
            cell.grid(row=r, column=c, padx=3, pady=3, sticky="ew")
            grid.columnconfigure(c, weight=1)

            tk.Label(cell, text=lbl, bg=BORDER, fg=col,
                     font=("Segoe UI", 8, "bold")).pack(anchor="w", padx=6, pady=(5, 0))
            var = tk.StringVar(value="—")
            tk.Label(cell, textvariable=var, bg=BORDER, fg=TEXT,
                     font=("Consolas", 11, "bold")).pack(anchor="w", padx=6, pady=(0, 5))
            self._sensor_vars.append(var)

    # ── Log panel ─────────────────────────────────────────────────────────────
    def _build_log_panel(self, parent):
        outer = tk.Frame(parent, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        outer.pack(fill="both", expand=True)

        tk.Label(outer, text="Event Log",
                 bg=PANEL, fg=SUBTEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=8, pady=(8, 2))

        self._log = tk.Text(outer, bg=CHART_BG, fg=TEXT,
                            font=("Consolas", 9),
                            relief="flat", bd=0,
                            state="disabled", wrap="word")
        sb = tk.Scrollbar(outer, command=self._log.yview,
                          bg=PANEL, troughcolor=BORDER,
                          activebackground=ACCENT)
        self._log.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y", padx=(0, 4))
        self._log.pack(fill="both", expand=True, padx=6, pady=(0, 8))

        # text tags for colour-coded log lines
        self._log.tag_configure("normal",  foreground=GREEN)
        self._log.tag_configure("anomaly", foreground=RED)
        self._log.tag_configure("info",    foreground=ACCENT)
        self._log.tag_configure("warn",    foreground=YELLOW)

    # ── MSE reconstruction error chart ────────────────────────────────────────
    def _build_mse_chart(self, parent):
        frame = tk.Frame(parent, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="x", pady=(0, 6))

        tk.Label(frame, text="Reconstruction Error (MSE)",
                 bg=PANEL, fg=SUBTEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 0))

        fig, self._ax_mse = plt.subplots(figsize=(6, 2.2), facecolor=CHART_BG)
        fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18)
        self._ax_mse.set_facecolor(CHART_BG)
        self._ax_mse.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in self._ax_mse.spines.values():
            spine.set_edgecolor(BORDER)

        self._mse_line, = self._ax_mse.plot([], [], color=ACCENT, lw=1.5, alpha=0.9)
        self._mse_fill  = None
        self._thresh_line = self._ax_mse.axhline(THRESHOLD, color=RED,
                                                  ls="--", lw=1.2, alpha=0.8,
                                                  label=f"Threshold {THRESHOLD}")
        self._ax_mse.legend(loc="upper left", facecolor=PANEL,
                            edgecolor=BORDER, labelcolor=SUBTEXT, fontsize=8)
        self._ax_mse.set_xlim(0, HISTORY)
        self._ax_mse.set_ylim(0, THRESHOLD * 4)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="x", padx=6, pady=(2, 6))
        self._mse_canvas = canvas

    # ── 8-channel sensor chart ─────────────────────────────────────────────────
    def _build_sensor_chart(self, parent):
        frame = tk.Frame(parent, bg=PANEL,
                         highlightbackground=BORDER, highlightthickness=1)
        frame.pack(fill="both", expand=True)

        tk.Label(frame, text="Sensor Time Series",
                 bg=PANEL, fg=SUBTEXT,
                 font=("Segoe UI", 9, "bold")).pack(anchor="w", padx=10, pady=(8, 0))

        fig, self._ax_sen = plt.subplots(figsize=(6, 3.2), facecolor=CHART_BG)
        fig.subplots_adjust(left=0.08, right=0.78, top=0.92, bottom=0.14)
        self._ax_sen.set_facecolor(CHART_BG)
        self._ax_sen.tick_params(colors=SUBTEXT, labelsize=8)
        for spine in self._ax_sen.spines.values():
            spine.set_edgecolor(BORDER)

        self._sen_lines = []
        for i in range(8):
            line, = self._ax_sen.plot([], [], color=SENSOR_COLORS[i],
                                      lw=1.2, alpha=0.85,
                                      label=SENSOR_LABELS[i])
            self._sen_lines.append(line)

        self._ax_sen.legend(loc="upper left",
                            bbox_to_anchor=(1.01, 1),
                            facecolor=PANEL, edgecolor=BORDER,
                            labelcolor=TEXT, fontsize=8, ncol=1)
        self._ax_sen.set_xlim(0, HISTORY)
        self._ax_sen.set_ylim(-35000, 35000)

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=(2, 6))
        self._sen_canvas = canvas

    # ── Animation (poll queue, refresh charts) ────────────────────────────────
    def _start_animation(self):
        self._poll_queue()

    def _poll_queue(self):
        """Drain the data queue and update the UI every ~50 ms."""
        try:
            while True:
                item = data_q.get_nowait()
                self._handle_item(item)
        except queue.Empty:
            pass

        self._update_charts()
        self.after(50, self._poll_queue)

    def _handle_item(self, item: dict):
        if "error" in item:
            self._log_msg(f"ERROR: {item['error']}", "anomaly")
            return

        if "log" in item:
            text = item["log"]
            tag  = ("anomaly" if "ANOMALY" in text or "fault" in text.lower()
                    else "warn" if ">>>" in text
                    else "info"  if "✓" not in text and "Normal" not in text and "loaded" in text.lower()
                    else "normal" if "Normal" in text or "✓" in text
                    else "info")
            self._log_msg(text, tag)

        if "mse" in item:
            mse       = item["mse"]
            is_anom   = item["anomaly"]
            sensors   = item["sensors"]
            step      = item["step"]

            self.step_counter = step
            if is_anom:
                self.anomaly_count += 1
            else:
                self.normal_count += 1

            self.mse_history.append(mse)
            for i, v in enumerate(sensors):
                self.sensor_history[i].append(v)

            # update text labels
            self._mse_var.set(f"MSE: {mse:.6f}")
            self._step_var.set(f"Step: {step}")
            self._norm_var.set(str(self.normal_count))
            self._anom_var.set(str(self.anomaly_count))
            self._status_var.set(
                f"Step {step}  |  MSE {mse:.6f}  |  "
                f"{'⚠ ANOMALY DETECTED' if is_anom else '✓ Normal'}  "
                f"|  Anomaly count: {self.anomaly_count}"
            )

            for i, v in enumerate(sensors):
                self._sensor_vars[i].set(f"{v:>10.1f}")

            # update state card
            if is_anom:
                self._state_label.config(text="⚠  ANOMALY", fg=RED)
                self.configure(bg="#1a0d0d")
                self._flush_bg("#1a0d0d")
            else:
                self._state_label.config(text="●  NORMAL", fg=GREEN)
                self.configure(bg=BG)
                self._flush_bg(BG)

        if item.get("done"):
            self._running = False
            self._btn.config(text="▶  Restart", bg=GREEN)
            self._state_label.config(text="Done", fg=SUBTEXT)
            self._status_var.set("Demo finished.")

    def _flush_bg(self, color):
        """Flash the window background to draw attention on state changes."""
        pass  # subtle — handled via label colour already

    def _update_charts(self):
        xs = list(range(HISTORY))

        # MSE chart
        ys_mse = list(self.mse_history)
        self._mse_line.set_data(xs, ys_mse)

        # colour fill above/below threshold
        ax = self._ax_mse
        for coll in ax.collections:
            coll.remove()
        ax.fill_between(xs, ys_mse, THRESHOLD,
                        where=[y > THRESHOLD for y in ys_mse],
                        color=RED, alpha=0.25, interpolate=True)
        ax.fill_between(xs, ys_mse, 0,
                        where=[y <= THRESHOLD for y in ys_mse],
                        color=ACCENT, alpha=0.12, interpolate=True)

        cur_max = max(max(ys_mse, default=0), THRESHOLD * 1.5)
        ax.set_ylim(0, cur_max * 1.15)
        self._mse_canvas.draw_idle()

        # Sensor chart
        for i, line in enumerate(self._sen_lines):
            line.set_data(xs, list(self.sensor_history[i]))

        visible = [v for hist in self.sensor_history for v in hist if v != 0]
        if visible:
            mn, mx = min(visible), max(visible)
            pad = max(abs(mx - mn) * 0.1, 1)
            self._ax_sen.set_ylim(mn - pad, mx + pad)

        self._sen_canvas.draw_idle()

    # ── Controls ──────────────────────────────────────────────────────────────
    def _toggle_run(self):
        if self._running:
            self._stop_event.set()
            self._running = False
            self._btn.config(text="▶  Start", bg=GREEN)
            self._state_label.config(text="Stopped", fg=YELLOW)
            self._log_msg("Stopped by user.", "warn")
        else:
            # reset state
            self._stop_event.clear()
            self.mse_history     = deque([0.0] * HISTORY, maxlen=HISTORY)
            self.sensor_history  = [deque([0.0] * HISTORY, maxlen=HISTORY) for _ in range(8)]
            self.step_counter    = 0
            self.anomaly_count   = 0
            self.normal_count    = 0
            self._norm_var.set("0")
            self._anom_var.set("0")

            self._running = True
            self._btn.config(text="■  Stop", bg=RED)
            self._log_msg("=== Session started ===", "info")
            self._thread = threading.Thread(target=run_detector,
                                            args=(self._stop_event,),
                                            daemon=True)
            self._thread.start()

    def _log_msg(self, text: str, tag: str = "info"):
        self._log.config(state="normal")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.config(state="disabled")

    def _on_close(self):
        self._stop_event.set()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = AnomalyDashboard()
    app.mainloop()
