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
except:
    pass


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)

HEATER_PARAMETERS = {
    (2, 10): ("Low (2.5W)", 30, 40, 40),
    (10, 15): ("High (25W)", 25, 30, 30),
    (15, 100): ("High (25W)", 35, 40, 40),
    (100, np.inf): ("High (25W)", 40, 40, 40),
}


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
                    writer.writerow([dt.isoformat(), f"{elapsed:.3f}"] + msg)
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
    filename,
    delta,
    curr,
    tempstart,
    tempend,
    temprate,
    updatesignal=None,
    heatingsignal=None,
    abortflag=None,
):
    # Connect to GPIB instruments
    lake = LakeshoreModel325("lake", "GPIB0::12::INSTR")
    keithley = Keithley2450("keithley", "GPIB1::18::INSTR")

    def adjust_heater(temperature):
        for temprange, params in HEATER_PARAMETERS.items():
            if temprange[0] < temperature < temprange[1]:
                lake.heater_1.output_range(params[0])
                lake.heater_1.P(params[1])
                lake.heater_1.I(params[2])
                lake.heater_1.D(params[3])
                return

    # Keithley 2450 setup
    # keithley.reset()
    keithley.write("SENS:FUNC VOLT")
    keithley.write("SENS:VOLT:RANG:AUTO ON")
    keithley.write("SENS:VOLT:UNIT OHM")
    keithley.write("SENS:VOLT:RSEN ON")
    keithley.write("SENS:VOLT:OCOM ON")
    keithley.write("SENS:VOLT:NPLC 10")

    keithley.write("SOUR:FUNC CURR")
    keithley.write("SOUR:CURR:RANG:AUTO ON")
    keithley.write(f"SOUR:CURR {curr:.15f}")
    keithley.write("SOUR:CURR:VLIM 10")
    keithley.write("OUTP ON")

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
        elif k == 1:
            target = tempend
            if heatingsignal is not None:
                heatingsignal.emit()

        log.info(f"[Set temperature {target} K ]")
        lake.write(f"RAMP 1,1,{temprate}")
        lake.heater_1.setpoint(target)
        adjust_heater(300.0)

        while True:
            resistance: str = keithley.ask("MEAS:VOLT?")
            temperature: float = lake.sensor_B.temperature()
            current: str = keithley.ask("SOUR:CURR?")
            now = datetime.datetime.now()

            writer.append(now, [str(temperature), resistance, current])
            log.info(f"{now}\t{temperature:14.3f}\t{resistance}\t{current}")
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
    keithley.write("OUTP OFF")
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
        self.measurement_thread.sigUpdated.connect(self.plot.update_data)
        self.measurement_thread.sigHeating.connect(self.plot.started_heating)
        self.measurement_thread.sigStarted.connect(self.started)
        self.measurement_thread.sigStarted.connect(self.plot.started)
        self.measurement_thread.sigFinished.connect(self.finished)

    @property
    def measurement_parameters(self) -> dict:
        return dict(
            filename=self.file_line.text(),
            delta=self.spin_delta.value(),
            curr=self.spin_curr.value() * 1e-3,
            tempstart=self.spin_start.value(),
            tempend=self.spin_end.value(),
            temprate=self.spin_rate.value(),
        )

    @QtCore.Slot()
    def toggle_measurement(self):
        if self.measurement_thread.isRunning():
            self.abort_measurement()
        else:
            self.start_measurement()

    @QtCore.Slot()
    def start_measurement(self):
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
        ):
            w.setDisabled(True)
        self.start_btn.setText("Abort Measurement")

    @QtCore.Slot()
    def finished(self):
        for w in (
            self.file_line,
            self.spin_delta,
            self.spin_curr,
            self.spin_start,
            self.spin_end,
            self.spin_rate,
        ):
            w.setDisabled(False)
        self.start_btn.setText("Start Measurement")

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
        dialog.setOption(QtWidgets.QFileDialog.Option.DontUseNativeDialog)

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

    win = MainWindow()
    win.show()
    win.activateWindow()

    qapp.exec()
