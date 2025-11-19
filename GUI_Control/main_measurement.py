
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 02:51:11 2025

@author: gani 
"""
"""
Main program for I–V sweep GUI using PyQt5 + PyVISA.

- Uses your existing UI file: Instrument.py (class Ui_MainWindow)
- Reads start/end voltages from the two QDoubleSpinBox widgets
- Start/Stop controls run a background QThread for VISA I/O
- Live Matplotlib plot of I–V data

Important:
- Forces VISA backend to '/usr/lib/librsvisa.so' (R&S VISA) to match your working script.
  Change VISA_BACKEND below if you want to try the default backend.

Run:
    python3 main_measurement.py
"""

import sys
import time
import numpy as np
import pyvisa

from PyQt5 import QtCore, QtGui, QtWidgets
from Instrument import Ui_MainWindow  # your UI file

# ---- Matplotlib (embedded) ----
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# ---------------- Config ----------------
# Force the same VISA backend that works in your standalone script
VISA_BACKEND = '/usr/lib/librsvisa.so'  # R&S VISA on Linux. Set to None to try default first.


# ---------------- Worker Thread for Sweep ----------------
class SweepWorker(QtCore.QThread):
    point_measured = QtCore.pyqtSignal(float, float)       # v, i
    sweep_done = QtCore.pyqtSignal(np.ndarray, np.ndarray) # volts, currents
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(self, visa_backend: str, visa_resource: str, start_v: float, end_v: float,
                 step: float = 0.5, channel: int = 1, settle_s: float = 1.0,
                 parent=None):
        super().__init__(parent)
        self.visa_backend = visa_backend  # may be None (use default) or path to backend lib
        self.visa_resource = visa_resource
        self.start_v = float(start_v)
        self.end_v = float(end_v)
        self.step = abs(float(step)) if step != 0 else 0.5
        self.channel = int(channel)
        self.settle_s = max(0.0, float(settle_s))
        self._stop = False
        self._inst = None

    def stop(self):
        self._stop = True

    def _build_sweep(self):
        """Direction-aware sweep that includes the end value."""
        if self.start_v <= self.end_v:
            volts = np.arange(self.start_v, self.end_v + 1e-12, self.step)
            if abs(self.end_v - volts[-1]) > 1e-9:
                volts = np.append(volts, self.end_v)
        else:
            volts = np.arange(self.start_v, self.end_v - 1e-12, -self.step)
            if abs(self.end_v - volts[-1]) > 1e-9:
                volts = np.append(volts, self.end_v)
        return volts

    def _safe_output_off(self):
        try:
            if self._inst is not None:
                self._inst.write('OUTP OFF')
        except Exception:
            pass

    def run(self):
        volts = self._build_sweep()
        currents = []

        # Connect VISA using the specified backend
        try:
            self.status.emit(f"Connecting VISA (backend={self.visa_backend or 'default'})...")
            if self.visa_backend:
                rm = pyvisa.ResourceManager(self.visa_backend)
            else:
                rm = pyvisa.ResourceManager()
            self._inst = rm.open_resource(self.visa_resource)
            self._inst.write_termination = '\n'
            self._inst.read_termination = '\n'
            self._inst.timeout = 10_000  # ms

            idn = self._inst.query("*IDN?")
            self.status.emit(f"Connected: {idn.strip()}")
        except Exception as e:
            self.error.emit(f"VISA connection failed: {e}")
            return

        try:
            # Select channel & prepare output (adjust SCPI to your PSU if needed)
            self._inst.write(f':INST:NSEL {self.channel}')
            try:
                self._inst.write("INST OUT1")    # Some PSUs accept this alias
            except Exception:
                pass
            try:
                self._inst.write("OUTP:SEL ON")  # Multi-channel selection (if supported)
            except Exception:
                pass

            # Optional: set current limit for safety
            # self._inst.write('CURR 0.5')

            # Sweep
            for v in volts:
                if self._stop:
                    self.status.emit("Sweep stopped by user.")
                    break
                try:
                    self._inst.write(f'VOLT {v}')
                    self._inst.write('OUTP ON')
                    if self.settle_s > 0:
                        time.sleep(self.settle_s)
                    current = float(self._inst.query('MEAS:CURR?'))
                except Exception as inner_e:
                    self.error.emit(f"SCPI I/O error at {v:.3f} V: {inner_e}")
                    self._safe_output_off()
                    break

                currents.append(current)
                self.point_measured.emit(float(v), float(current))
                self.status.emit(f"V={v:.3f} V, I={current:.6f} A")

            # After sweep or stop
            self._safe_output_off()

            if len(currents) > 0:
                self.sweep_done.emit(volts[:len(currents)], np.array(currents))
        finally:
            try:
                if self._inst is not None:
                    self._inst.close()
            except Exception:
                pass


# ---------------- Main Window ----------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Embed Matplotlib canvas into the placeholder verticalLayout in your UI
        self.figure = Figure(figsize=(4.0, 3.0), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("I–V Characteristics")
        self.ax.set_xlabel("Voltage (V)")
        self.ax.set_ylabel("Current (A)")
        self.ax.grid(True, alpha=0.3)
        self.iv_line, = self.ax.plot([], [], 'o-', lw=1.5)
        self.ui.verticalLayout.addWidget(self.canvas)

        # Defaults for the spin boxes
        self.ui.doubleSpinBox.setValue(-3.0)  # Start
        self.ui.doubleSpinBox_2.setValue(6.0) # End

        # Start/Stop connections
        self.ui.pushButton.clicked.connect(self.on_start)
        self.ui.pushButton_2.clicked.connect(self.on_stop)
        self.ui.pushButton_2.setEnabled(False)

        # VISA discovery (using same backend strategy as worker)
        self._visa_backend = VISA_BACKEND  # keep consistent across discovery & connection
        self._visa_resource = None
        self._discover_instruments()

        # State
        self._worker = None
        self._volt_data = []
        self._curr_data = []

    def _discover_instruments(self):
        resources = []
        errors = []

        # First try the configured backend (preferred)
        if self._visa_backend:
            try:
                rm = pyvisa.ResourceManager(self._visa_backend)
                resources = rm.list_resources()
            except Exception as e:
                errors.append(f"Backend {self._visa_backend}: {e}")

        # If none found and no fixed backend, try default
        if not resources and not self._visa_backend:
            try:
                rm_def = pyvisa.ResourceManager()
                resources = rm_def.list_resources()
            except Exception as e:
                errors.append(f"Default backend: {e}")

        if resources:
            self._visa_resource = resources[0]
            self.statusBar().showMessage(f"Found instruments: {resources} | Using: {self._visa_resource}", 10_000)
        else:
            self._visa_resource = None
            msg = "No instruments found."
            if errors:
                msg += " Errors: " + " | ".join(errors)
            self.statusBar().showMessage(msg, 15_000)

    def on_start(self):
        if self._worker is not None and self._worker.isRunning():
            self.statusBar().showMessage("Sweep already running...", 5000)
            return

        if self._visa_resource is None:
            self.statusBar().showMessage("No VISA instrument available.", 8000)
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Instrument Not Found")
            msg.setText(
                "No VISA instruments detected.\n"
                f"Tried backend: {self._visa_backend or 'default'}\n"
                "Make sure your PSU is connected and visible to VISA."
            )
            msg.exec_()
            return

        start_v = float(self.ui.doubleSpinBox.value())
        end_v = float(self.ui.doubleSpinBox_2.value())
        step = 0.5   # fixed per your requirement
        settle = 1.0 # seconds

        # Reset plot/data
        self._volt_data = []
        self._curr_data = []
        self.iv_line.set_data([], [])
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

        # Start worker with the SAME backend + resource used in discovery
        self._worker = SweepWorker(
            visa_backend=self._visa_backend,
            visa_resource=self._visa_resource,
            start_v=start_v,
            end_v=end_v,
            step=step,
            channel=1,
            settle_s=settle
        )
        self._worker.point_measured.connect(self.on_point)
        self._worker.sweep_done.connect(self.on_done)
        self._worker.status.connect(self.on_status)
        self._worker.error.connect(self.on_error)
        self._worker.finished.connect(self.on_finished)

        self.ui.pushButton.setEnabled(False)
        self.ui.pushButton_2.setEnabled(True)
        self.statusBar().showMessage("Starting sweep...")
        self._worker.start()

    def on_stop(self):
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.statusBar().showMessage("Stopping sweep...", 5000)

    @QtCore.pyqtSlot(float, float)
    def on_point(self, v, i):
        self._volt_data.append(v)
        self._curr_data.append(i)
        self.iv_line.set_data(self._volt_data, self._curr_data)
        self.ax.relim()
               # dynamic autoscale for live plotting
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    @QtCore.pyqtSlot(np.ndarray, np.ndarray)
    def on_done(self, volts, currents):
        self.iv_line.set_data(volts, currents)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()
        self.statusBar().showMessage("Sweep complete.", 8000)

    @QtCore.pyqtSlot(str)
    def on_status(self, msg):
        self.statusBar().showMessage(msg, 5000)

    @QtCore.pyqtSlot(str)
    def on_error(self, msg):
        self.statusBar().showMessage(msg, 10_000)
        QtWidgets.QMessageBox.critical(self, "SCPI / VISA Error", msg)

    def on_finished(self):
        self.ui.pushButton.setEnabled(True)
        self.ui.pushButton_2.setEnabled(False)

    def closeEvent(self, event: QtGui.QCloseEvent):
        # Ensure worker stops and PSU output goes off
        try:
            if self._worker is not None and self._worker.isRunning():
                self._worker.stop()
                self._worker.wait(3000)
        except Exception:
            pass
        return super().closeEvent(event)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
