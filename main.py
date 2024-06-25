import csv
import datetime
import logging
import multiprocessing
import os
import sys
import threading
import time

import numpy as np
from qcodes.instrument_drivers.Keithley import Keithley2450
from qcodes.instrument_drivers.Lakeshore import LakeshoreModel325
from qtpy import QtCore, QtWidgets, uic

from plotting import PlotWindow

try:
    os.chdir(sys._MEIPASS)
except:  # noqa: E722
    pass


# HEATER_PARAMETERS: dict[tuple[int, int], tuple[str, int, int]] = {
#     (2, 10): ("1", 30, 40, 40),
#     (10, 15): ("2", 25, 30, 30),
#     (15, 100): ("2", 35, 40, 40),
#     (100, 275): ("2", 40, 40, 40),
#     (275, np.inf): ("2", 40, 60, 40),
# }  #: Heater and PID parameters for each temperature range
HEATER_PARAMETERS: dict[tuple[int, int], tuple[str, int, int]] = {
    (0, 9): ("1", 100, 40, 40),
    (9, 17): ("1", 70, 35, 30),
    (17, 30): ("2", 35, 40, 40),
    (30, 75): ("2", 35, 40, 40),
    (75, 275): ("2", 40, 40, 40),
    (275, np.inf): ("2", 40, 70, 40),
}  #: Heater and PID parameters for each temperature range

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)


class WritingProc(multiprocessing.Process):
    def __init__(self, filename: os.PathLike):
        super().__init__()
        self.filename = str(filename)
        self._stopped = multiprocessing.Event()
        self.queue = multiprocessing.Queue()
        self.start_datetime = None

        # Write header if file does not exist
        if not os.path.isfile(self.filename):
            with open(self.filename, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Date & Time",
                        "Elapsed Time (s)",
                        "Temperature (K)",
                        "Resistance (Ohm)",
                        "Current (A)",
                    ]
                )

    def run(self):
        self.start_datetime = datetime.datetime.now()
        self._stopped.clear()
        while not self._stopped.is_set():
            time.sleep(0.02)

            if self.queue.empty():
                continue

            # retrieve message from queue
            dt, msg = self.queue.get()
            try:
                with open(self.filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    elapsed = (dt - self.start_datetime).total_seconds()
                    writer.writerow([dt.isoformat(), f"{elapsed:.3f}", *msg])
            except PermissionError:
                # put back the retrieved message in the queue
                n_left = int(self.queue.qsize())
                self.queue.put((dt, msg))
                for _ in range(n_left):
                    self.queue.put(self.queue.get())
                continue

    def stop(self):
        n_left = int(self.queue.qsize())
        if n_left != 0:
            print(
                f"Failed to write {n_left} data "
                + ("entries:" if n_left > 1 else "entry:")
            )
            for _ in range(n_left):
                dt, msg = self.queue.get()
                print(f"{dt} | {msg}")
        self._stopped.set()
        self.join()

    def append(self, timestamp: datetime.datetime, content):
        if isinstance(content, str):
            content = [content]
        self.queue.put((timestamp, content))


def measure(
    filename: os.PathLike,
    tempstart: float,
    tempend: float,
    coolrate: float,
    heatrate: float,
    curr: float,
    delta: float,
    mode: int,
    updatesignal: QtCore.SignalInstance | None = None,
    heatingsignal: QtCore.SignalInstance | None = None,
    abortflag: threading.Event | None = None,
):
    """Loop for the R-T measurement.

    Optional arguments are for GUI integration. If not provided, it is possible to run
    the function without a GUI.

    Parameters
    ----------
    filename
        Name of .csv file.
    tempstart
        Low temperature in Kelvins.
    tempend
        High temperature in Kelvins.
    coolrate
        Cooling rate in Kelvins per minute
    heatrate
        Heating rate in Kelvins per minute.
    curr
        Current in Amperes.
    delta
        Interval between each measurement loop in seconds. Note that the real logging
        interval is larger than this value, and depends on the settings of the
        sourcemeter such as NPLC and count. This is NOT the same as the input value to
        the GUI interface.
    mode
        One of 0, 1, 2, each corresponding to the offset-compensated ohms method,
        current reversal method, and the delta method.
    updatesignal : optional
        Emits the elapsed time and data with each update, by default None
    heatingsignal : optional
        Emitted on starting ramp to `tempend`, by default None
    abortflag : optional
        The loop is aborted when this event is set, by default None

    """
    # Connect to GPIB instruments
    lake: LakeshoreModel325 = LakeshoreModel325("lake", "GPIB0::12::INSTR")
    keithley: Keithley2450 = Keithley2450("keithley", "GPIB1::18::INSTR")

    def adjust_heater(temperature):
        for temprange, params in HEATER_PARAMETERS.items():
            if temprange[0] < temperature < temprange[1]:
                lake.write(f"RANGE 1,{params[0]}")
                lake.write(f"PID 1,{params[1]},{params[2]},{params[3]}")
                return

    # Keithley 2450 setup
    keithley.reset()
    keithley.write('SENS:FUNC "VOLT"')
    keithley.write("SENS:VOLT:UNIT OHM")
    keithley.write("SENS:VOLT:RANG:AUTO ON")
    if mode == 0:  # offset-compensated ohms method
        keithley.write("SENS:VOLT:OCOM ON")
        keithley.write("SENS:VOLT:NPLC 4")
    elif mode == 1:  # current-reversal method
        keithley.write("SENS:VOLT:OCOM ON")
        keithley.write("SENS:VOLT:NPLC 1.5")
    elif mode == 2:  # delta method
        keithley.write("SENS:VOLT:OCOM OFF")
        keithley.write("SENS:VOLT:NPLC 1.5")
    keithley.write("SENS:VOLT:RSEN ON")  # 4-wire mode

    keithley.write("SOUR:FUNC CURR")
    keithley.write("SOUR:CURR:RANG:AUTO ON")
    keithley.write("SOUR:CURR:VLIM 10")
    keithley.write(f"SOUR:CURR {curr:.15f}")

    # LakeShore325 temperature controller
    lake.write("OUTMODE 1,1,2,1")
    temperature = lake.sensor_B.temperature()

    if np.abs(temperature - tempstart) > 10:
        lake.write("RAMP 1,1,0")
        lake.heater_1.setpoint(temperature + 1.0)
        time.sleep(2)

    # Start data writer
    writer = WritingProc(filename)
    writer.start()

    for k in range(2):
        # k = 0 : measure while going to the Start Temperature
        # k = 1 : measure while going to the End Temperature.
        if k == 0:
            target = tempstart
            temprate = coolrate
        elif k == 1:
            target = tempend
            temprate = heatrate
            if heatingsignal is not None:
                heatingsignal.emit()
            # Additional time between cooling and heating
            time.sleep(10)

        log.info(f"[Set temperature {target} K ]")
        lake.write(f"RAMP 1,1,{temprate}")
        lake.heater_1.setpoint(target)
        adjust_heater(300.0)

        while True:
            # In order to compensate for voltage measurement time delay, time and
            # temperature are measured twice and averaged.

            temperature: float = lake.sensor_B.temperature()

            now = datetime.datetime.now()
            keithley.write("OUTP ON")
            if mode == 0:  # offset-compensated ohms method
                resistance: str = keithley.ask("MEAS:VOLT?")

            elif mode == 1:  # current-reversal method
                keithley.wait()
                rp = float(keithley.ask("MEAS:VOLT?"))

                keithley.write(f"SOUR:CURR -{curr:.15f}")
                keithley.wait()
                rm = float(keithley.ask("MEAS:VOLT?"))

                keithley.write(f"SOUR:CURR {curr:.15f}")

                resistance = str((abs(rp) + abs(rm)) / 2)

            elif mode == 2:  # delta method
                sgn = np.sign(float(keithley.ask("SOUR:CURR?")))

                keithley.write(f"SOUR:CURR {-sgn * curr:.15f}")
                keithley.wait()
                r1 = float(keithley.ask("MEAS:VOLT?"))

                keithley.write(f"SOUR:CURR {sgn * curr:.15f}")
                keithley.wait()
                r2 = float(keithley.ask("MEAS:VOLT?"))

                keithley.write(f"SOUR:CURR {-sgn * curr:.15f}")
                keithley.wait()
                r3 = float(keithley.ask("MEAS:VOLT?"))

                ra, rb = (r1 - r2) / 2, (r3 - r2) / 2
                resistance = str(abs(ra - rb) / 2)

            keithley.write("OUTP OFF")
            now = now + (datetime.datetime.now() - now) / 2

            temperature += lake.sensor_B.temperature()
            temperature /= 2.0

            current: str = keithley.ask("SOUR:CURR?")

            writer.append(now, [str(temperature), resistance, current])
            log.info(
                f"{now} | {temperature:>7.3f} K | {float(resistance):.9f} Ω "
                f"at {float(current):.9f} A"
            )
            if updatesignal is not None:
                updatesignal.emit(now, (temperature, float(resistance), float(current)))
            adjust_heater(temperature)

            if np.abs(target - temperature) < 0.5:
                lake.write("PID 1,30,40,40")
                lake.write("RAMP 1,1,1")  # why?
                break

            if abortflag is not None:
                if abortflag.is_set():
                    break

            time.sleep(delta)

        if abortflag is not None:
            if abortflag.is_set():
                break

    writer.stop()
    keithley.write("SOUR:CURR 0")
    keithley.close()
    lake.close()


class MeasureThread(QtCore.QThread):
    sigStarted = QtCore.Signal()
    sigFinished = QtCore.Signal()
    sigUpdated = QtCore.Signal(object, object)
    sigHeating = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.aborted = threading.Event()
        self.measure_params = None

    def run(self):
        self.sigStarted.emit()
        self.aborted.clear()
        measure(
            **self.measure_params,
            updatesignal=self.sigUpdated,
            heatingsignal=self.sigHeating,
            abortflag=self.aborted,
        )
        self.sigFinished.emit()


class MainWindow(*uic.loadUiType("main.ui")):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("R-T Measurement")
        self.file_btn.clicked.connect(self.choose_file)
        self.start_btn.clicked.connect(self.toggle_measurement)

        self.plot = PlotWindow()
        self.actionplot.triggered.connect(self.toggle_plot)

        self.measurement_thread = MeasureThread()
        self.measurement_thread.sigStarted.connect(self.started)
        self.measurement_thread.sigStarted.connect(self.plot.started)
        self.measurement_thread.sigHeating.connect(self.plot.started_heating)
        self.measurement_thread.sigUpdated.connect(self.plot.update_data)
        self.measurement_thread.sigUpdated.connect(self.updated)
        self.measurement_thread.sigFinished.connect(self.finished)

    @property
    def measurement_parameters(self) -> dict:
        return {
            "filename": self.file_line.text(),
            "tempstart": self.spin_start.value(),
            "tempend": self.spin_end.value(),
            "coolrate": self.spin_rate.value(),
            "heatrate": self.spin_rateh.value(),
            "curr": self.spin_curr.value() * 1e-3,
            "delta": self.spin_delta.value(),  # est. from NPLC settings
            "mode": self.mode_combo.currentIndex(),
        }

    @QtCore.Slot()
    def toggle_measurement(self):
        if self.measurement_thread.isRunning():
            self.abort_measurement()
        else:
            self.start_measurement()

    @QtCore.Slot()
    def start_measurement(self):
        if self.file_line.text() == "":
            QtWidgets.QMessageBox.critical(
                self, "Empty File", "Select a file before starting the measurement."
            )
            return

        params = self.measurement_parameters

        if params["tempstart"] >= params["tempend"]:
            QtWidgets.QMessageBox.critical(
                self,
                "Invalid Temperature",
                "Low Temperature must be lower than High Temperature.",
            )
            return

        ret = QtWidgets.QMessageBox.question(
            self,
            "Confirm Parameters",
            "\n".join(
                [
                    f"Save to {params['filename']}",
                    f"Low {params['tempstart']} K",
                    f"High {params['tempend']} K",
                    f"Cool {params['coolrate']} K/min",
                    f"Heat {params['heatrate']} K/min",
                    f"Current {params['curr']} A",
                    f"Every {params['delta']} s",
                ]
            ),
        )
        if ret == QtWidgets.QMessageBox.Yes:
            self.measurement_thread.measure_params = self.measurement_parameters
            self.measurement_thread.start()

    @QtCore.Slot()
    def abort_measurement(self):
        self.measurement_thread.aborted.set()
        self.measurement_thread.wait()

    @QtCore.Slot()
    def started(self):
        for w in (
            self.file_line,
            self.spin_delta,
            self.spin_curr,
            self.spin_start,
            self.spin_end,
            self.spin_rate,
            self.spin_rateh,
            self.mode_combo,
        ):
            w.setDisabled(True)
        self.start_btn.setText("Abort Measurement")
        self.statusBar().setVisible(True)

    @QtCore.Slot(object, object)
    def updated(self, _, data: tuple[float, float, float]):
        temp, res, _ = data
        self.statusBar().showMessage(f"T = {temp:.5g} K, R = {res:.5g} Ω")

    @QtCore.Slot()
    def finished(self):
        for w in (
            self.file_line,
            self.spin_delta,
            self.spin_curr,
            self.spin_start,
            self.spin_end,
            self.spin_rate,
            self.spin_rateh,
            self.mode_combo,
        ):
            w.setDisabled(False)
        self.start_btn.setText("Start Measurement")
        self.statusBar().setVisible(False)

    @QtCore.Slot()
    def toggle_plot(self):
        if self.plot.isVisible():
            self.plot.hide()
        else:
            self.plot.show()

    @QtCore.Slot()
    def choose_file(self):
        dialog = QtWidgets.QFileDialog(self)
        dialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        dialog.setNameFilter("Comma-separated values (*.csv)")
        # dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

        if dialog.exec():
            self.file_line.setText(dialog.selectedFiles()[0])

    def closeEvent(self, *args, **kwargs):
        self.abort_measurement()
        self.plot.close()
        super().closeEvent(*args, **kwargs)


if __name__ == "__main__":
    multiprocessing.freeze_support()

    qapp: QtWidgets.QApplication = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")

    win = MainWindow()
    win.show()
    win.activateWindow()

    qapp.exec()
