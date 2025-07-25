from __future__ import annotations

import collections
import contextlib
import csv
import datetime
import logging
import multiprocessing
import os
import queue
import sys
import threading
import time
from typing import Literal

import numpy as np
import pyvisa
from qtpy import QtCore, QtWidgets, uic

from instrument import RequestHandler
from plotting import PlotWindow

with contextlib.suppress(Exception):
    os.chdir(sys._MEIPASS)

RESOURCE_LAKESHORE: str = "GPIB0::12::INSTR"  #: Resource name for the LakeShore 325
RESOURCE_KEITHLEY: str = "GPIB1::18::INSTR"  #: Resource name for the Keithley 2450


# HEATER_PARAMETERS: dict[tuple[int, int], tuple[str, int, int]] = {
#     (2, 10): ("1", 30, 40, 40),
#     (10, 15): ("2", 25, 30, 30),
#     (15, 100): ("2", 35, 40, 40),
#     (100, 275): ("2", 40, 40, 40),
#     (275, np.inf): ("2", 40, 60, 40),
# }
HEATER_PARAMETERS: dict[tuple[int, int], tuple[str, int, int]] = {
    (0, 18): ("1", 10, 670, 0),
    (18, 22): ("2", 10, 670, 0),
    (22, 26): ("2", 40, 30, 0),
    (26, 75): ("2", 100, 30, 0),
    (75, 150): ("2", 175, 30, 0),
    (150, 250): ("2", 250, 31, 0),
    (250, 350): ("2", 300, 33, 0),
}  #: Heater range and PID parameters for each temperature range. Maximum 10 zones.


TEMP_SENSOR_LOG: str = "B"  #: Temperature sensor to use for logging
TEMP_SENSOR_LOOP: str = "B"  #: Temperature sensor to use for PID control

T_DIFF: float = 0.3  #: Temperature difference in Kelvins to consider measurement done

# TB is near the sample, and TA is near the cold head & heater. If TB reads lower than
# TA at low temperatures with the heater off, the two ports on the LakeShore 325 may not
# be connected properly.

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
log.addHandler(handler)

exc_handler = logging.FileHandler("exceptions.log")
exc_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
exc_handler.setLevel(logging.ERROR)
log.addHandler(exc_handler)


def _log_uncaught_exceptions(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    log.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = _log_uncaught_exceptions


def measure(
    filename: os.PathLike,
    tempstart: float,
    tempend: float,
    coolrate: float,
    heatrate: float,
    curr: float,
    delay: float,
    nplc: float,
    mode: Literal[0, 1, 2],
    pmode: Literal[0, 1],
    manual: bool = False,
    resetlake: bool = True,
    resetkeithley: bool = True,
    updatesignal: QtCore.SignalInstance | None = None,
    heatingsignal: QtCore.SignalInstance | None = None,
    abortflag: threading.Event | None = None,
    queue: queue.Queue | None = None,
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
    delay
        Delay in minutes after reaching `tempstart` before starting the ramp to
        `tempend`.
    nplc
        Number of power line cycles for the Keithley 2450. Values under 1 are not
        recommended. Larger values increase the measurement time but reduce noise.
    mode
        One of 0, 1, 2, each corresponding to the offset-compensated ohms method,
        current reversal method, and the delta method.
    pmode
        One of 0 and 1 each corresponding to the 4wire probe and 2wire probe mode
    manual : optional
        If True, the heater is controlled manually, and the program only does the
        temperature-resistance logging. `tempstart`, `tempend`, `coolrate`, and `delay`
        are ignored, by default False
    resetlake : optional
        Resets the LakeShore 325 to default settings before acquisition, by default True
    resetkeithley : optional
        Resets the Keithley 2450 to default settings before acquisition, by default True
    updatesignal : optional
        Emits the time as a datetime object and the data as a 3-tuple of floats, by
        default None
    heatingsignal : optional
        Emitted on starting ramp to `tempend`, by default None
    abortflag : optional
        The loop is aborted when this event is set, by default None

    """
    # Connect to GPIB instruments
    lake = RequestHandler(RESOURCE_LAKESHORE)
    lake.open()
    log.info("[Connected to %s]", lake.query("*IDN?").strip())

    keithley = RequestHandler(RESOURCE_KEITHLEY, interval_ms=0)
    keithley.open()
    log.info("[Connected to %s]", keithley.query("*IDN?").strip())

    def flush_commands():
        if queue is not None:
            communicate(lake, queue)

    log.info("[Configured to log T%s]", TEMP_SENSOR_LOG)
    log.info("[Configured to use T%s in PID loop]", TEMP_SENSOR_LOOP)

    def get_krdg() -> float:
        return float(lake.query(f"KRDG? {TEMP_SENSOR_LOG}").strip())

    # Keithley 2450 setup
    if resetkeithley:
        keithley.write("*RST")
    keithley.write('SENS:FUNC "VOLT"')
    if pmode == 0:  # 4-wire mode
        keithley.write("SENS:VOLT:RSEN ON")  # 4-wire mode
    elif pmode == 1:  # 2-wire mode
        keithley.write("SENS:VOLT:RSEN OFF")
    keithley.write("SENS:VOLT:UNIT OHM")
    keithley.write("SENS:VOLT:RANG:AUTO ON")
    if mode == 0:  # offset-compensated ohms method
        keithley.write("SENS:VOLT:OCOM ON")
    elif mode == 1:  # current-reversal method
        # q_res = collections.deque(maxlen=2)
        # q_temp = collections.deque(maxlen=2)
        keithley.write("SENS:VOLT:OCOM OFF")
    elif mode == 2:  # delta method
        q_res = collections.deque(maxlen=3)
        q_temp = collections.deque(maxlen=3)
        keithley.write("SENS:VOLT:OCOM OFF")
    keithley.write(f"SENS:VOLT:NPLC {nplc:.2f}")

    keithley.write("SOUR:FUNC CURR")
    keithley.write("SOUR:CURR:RANG:AUTO ON")
    keithley.write("SOUR:CURR:VLIM 10")

    match mode:
        case 0:
            # Offset-compensated ohms method
            keithley.write(f"SOUR:CURR {curr:.15f}")
        case 1:  # Current-reversal method
            keithley.write(f":SOUR:SWE:CURR:LIN {curr:.15f}, {-curr:.15f}, 2")
        case 2:  # Delta method
            keithley.write(f"SOUR:CURR {curr:.15f}")

    # LakeShore325 temperature controller
    if resetlake:
        lake.write("*RST")
    lake.write(f"CSET 1,{TEMP_SENSOR_LOOP},1,0,2")  # Set loop 1 to control temperature

    # Populate zone 1 with PID parameters
    for i, (temprange, params) in enumerate(HEATER_PARAMETERS.items()):
        lake.write(
            f"ZONE 1,{i + 1},{temprange[1]},"
            f"{params[1]},{params[2]},{params[3]},"
            f"0, {params[0]}"
        )

    # Set loop 1 to use zone
    lake.write("CMODE 1,2")

    temperature: float = get_krdg()

    if not manual:
        log.info("[Estimated Measurement Timeline]")
        for s in _estimated_time_info(
            temperature, tempstart, tempend, coolrate, heatrate, delay, offset=3.0
        ):
            log.info("[%s]", s)

    if not manual and np.abs(temperature - tempstart) > 10:
        # If current temperature is far from the start temperature, setpoint to current
        # temperature first before measuring
        lake.write(f"RAMP 1,1,0; SETP 1,{temperature:.2f}")
        time.sleep(2)

    # Start data writer
    writer = WritingProc(filename, datetime.datetime.now())
    writer.start()

    # Variable to store time when waiting before heating
    t_cool_end: float | None = None

    # Start measurement
    keithley.write("OUTP ON")

    log.info("[Starting measurement]")

    match mode:
        case 2:
            curr_sign = 1.0  # Store sign of previously measured current

    for k in range(2):
        # k = 0 : measure while going to the Start Temperature
        # k = 1 : measure while going to the End Temperature.
        if k == 0:
            target, temprate = tempstart, coolrate
        elif k == 1:
            target, temprate = tempend, heatrate
            if heatingsignal is not None:
                heatingsignal.emit()
            # Add nan row before heating
            writer.append(datetime.datetime.now(), ["nan"] * 3)

            if not manual:
                # Avoid sudden output when starting heating at base temp
                temperature = get_krdg()
                if temperature < 3.0:
                    # Turn heater off
                    lake.write("RANGE 1,0")
                    # Set ramp rate to 0
                    lake.write("RAMP 1,1,0")
                    # Setpoint to current + 0.1 K
                    lake.write(f"SETP 1,{temperature + 0.5:.2f}; RANGE 1,1")

        if not manual:
            log.info("[Set temperature %s K ]", target)
            lake.write(f"RAMP 1,1,{temprate}; SETP 1,{target:.2f}")

        while True:
            flush_commands()

            # In order to compensate for voltage measurement time, the time and
            # temperature are measured twice and averaged.
            now: datetime.datetime = datetime.datetime.now()
            temperature = get_krdg()

            match mode:
                case 0:  # Offset-compensated ohms method
                    resistance, current = (
                        keithley.query(":MEAS:VOLT?; :SOUR:CURR?").strip().split(";")
                    )

                case 1:  # Current-reversal method
                    current = str(curr)
                    keithley.write("INIT")
                    keithley.write("*WAI")
                    msg = keithley.query('TRAC:DATA? 1, 2, "defbuffer1"')

                case 2:  # Delta method
                    keithley.write(f"SOUR:CURR {curr_sign * curr:.15f}")
                    res_i, current = (
                        keithley.query(":MEAS:VOLT?; :SOUR:CURR?").strip().split(";")
                    )

            now = now + (datetime.datetime.now() - now) / 2
            temperature = (temperature + get_krdg()) / 2

            match mode:
                case 1:  # Current-reversal method
                    # Take average of the two measurements
                    resistance = str(sum(map(float, msg.split(","))) / 2)

                case 2:  # Delta method
                    q_res.append(float(res_i))
                    q_temp.append(float(temperature))

                    if len(q_res) < 3:
                        resistance = "nan"
                    else:
                        resistance = str(np.abs(q_res[0] + q_res[2] + 2 * q_res[1]) / 4)

                        # Take moving average of temperature
                        temperature = sum(q_temp) / len(q_temp)

                    # Reverse current sign for next measurement
                    curr_sign *= -1.0

            if resistance != "nan":
                writer.append(now, [str(temperature), resistance, current])
                if updatesignal is not None:
                    updatesignal.emit(
                        now, (temperature, float(resistance), float(current))
                    )

                if not manual and np.abs(target - temperature) < T_DIFF:
                    if t_cool_end is None:
                        t_cool_end = time.perf_counter()

                    # Time elapsed since reaching Temp 1
                    time_elapsed = time.perf_counter() - t_cool_end

                    if time_elapsed >= delay * 60:
                        break  # Exit loop

            if abortflag is not None and abortflag.is_set():
                break

        if abortflag is not None and abortflag.is_set():
            log.info("[Measurement aborted]")
            break

    # Stop data writer
    writer.stop(2.0)

    # Stop measurement and close instruments
    keithley.write(":OUTP OFF; :SOUR:CURR 0")
    keithley.close()
    lake.close()


def communicate(handler: RequestHandler, queue: collections.deque):
    if not queue.empty():
        message, replysignal, is_query = queue.get()

        if not is_query:  # Write only
            try:
                handler.write(message)
            except (pyvisa.VisaIOError, pyvisa.InvalidSession):
                log.exception("Error writing command")
            else:
                log.info("[<- %s]", message.strip())
                replysignal.emit("Command sent.", datetime.datetime.now())
        else:  # Query
            try:
                rep = handler.query(message)
            except (pyvisa.VisaIOError, pyvisa.InvalidSession):
                log.exception("Error querying command")
            else:
                log.info("[<- %s]", message.strip())
                log.info("[-> %s]", rep.strip())
                replysignal.emit(rep, datetime.datetime.now())


def _format_minutes(minutes: float) -> str:
    hours, remainder = divmod(minutes * 60, 3600)
    minutes, seconds = divmod(remainder, 60)
    out = []

    hours, minutes, seconds = map(int, (hours, minutes, seconds))
    if hours >= 1:
        out.append(f"{hours} hour")
        if hours != 1:
            out[-1] += "s"
    if minutes >= 1:
        out.append(f"{minutes} minute")
        if minutes != 1:
            out[-1] += "s"
    if seconds >= 1:
        out.append(f"{seconds} second")
        if seconds != 1:
            out[-1] += "s"

    if len(out) > 1:
        if len(out) == 3:
            out[0] = out[0] + ","
        out.insert(-1, "and")

    return " ".join(out)


def _format_time(dt: datetime.datetime) -> str:
    return dt.strftime("%X")


def _estimated_time_info(
    temperature: float,
    tempstart: float,
    tempend: float,
    coolrate: float,
    heatrate: float,
    delay: float,
    offset: float = 0.0,
) -> list[str]:
    out = []

    cool_time = np.abs(temperature - tempstart) / coolrate
    heat_time = np.abs(tempstart - tempend) / heatrate

    start_time = datetime.datetime.now() + datetime.timedelta(seconds=offset)
    cool_end = start_time + datetime.timedelta(seconds=cool_time * 60)
    heat_start = cool_end + datetime.timedelta(seconds=delay * 60)
    heat_end = heat_start + datetime.timedelta(seconds=heat_time * 60)

    cool_time, delay, heat_time, total_time = map(
        _format_minutes, (cool_time, delay, heat_time, cool_time + delay + heat_time)
    )
    cool_end, heat_start, heat_end = map(_format_time, (cool_end, heat_start, heat_end))

    out.append(f"[1] {cool_end} ({cool_time})")
    out.append(f"[2] {heat_start} ({delay})")
    out.append(f"[3] {heat_end} ({heat_time})")
    out.append(f"Total {total_time}")

    if tempstart < 160.0 < temperature:
        time_elapsed_160 = np.abs(160 - temperature) / coolrate
        time_160 = datetime.datetime.now() + datetime.timedelta(
            seconds=time_elapsed_160 * 60
        )
        out.append(
            f"Close valve at {_format_time(time_160)} "
            f"({_format_minutes(time_elapsed_160)} from now)"
        )

    return out


def _make_log_string(
    dt: datetime.datetime, temperature: str, resistance: str, current: str
) -> str:
    """Create a formatted log string for the measurement."""
    log_str = f"  {dt}  "
    log_str += f"|  {float(temperature):>7.3f} K  "
    if float(resistance) > 1e3:
        log_str += f"|  {float(resistance) / 1e3:>10.5f} kΩ  "
    else:
        log_str += f"|  {float(resistance):>11.5f} Ω  "
    log_str += f"|  {float(current) * 1e3:+.6f} mA  "

    return log_str


class WritingProc(multiprocessing.Process):
    def __init__(self, filename: os.PathLike, start_datetime: datetime.datetime):
        super().__init__()
        self.filename = filename
        self._stopped = multiprocessing.Event()
        self.queue = multiprocessing.Queue()
        self.start_datetime: datetime.datetime = start_datetime

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
            else:
                log.info(_make_log_string(dt, *msg))

    def stop(self, timeout: int | None = None):
        n_left = int(self.queue.qsize())
        if n_left != 0:
            if timeout is not None:
                time.sleep(timeout)
                self.stop()
                return
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


class MeasureThread(QtCore.QThread):
    sigStarted = QtCore.Signal()
    sigFinished = QtCore.Signal()
    sigUpdated = QtCore.Signal(object, object)
    sigHeating = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.aborted = threading.Event()
        self.measure_params = None
        self.mutex: QtCore.QMutex | None = None

    def lock_mutex(self):
        """Locks the mutex to ensure thread safety."""
        if self.mutex is not None:
            self.mutex.lock()

    def unlock_mutex(self):
        """Unlocks the mutex to release the lock."""
        if self.mutex is not None:
            self.mutex.unlock()

    def request_query(self, message: str, signal: QtCore.SignalInstance):
        """Add a query request to the queue.

        Parameters
        ----------
        message : str
            The query message to send.
        signal : QtCore.SignalInstance
            The signal to emit the result of the query when the query is complete. The
            signal must take a string as the first argument and a object as the second.
            The second argument is the datetime object indicating the time the query was
            placed.
        """
        self.lock_mutex()
        self.queue.put((message, signal, True))
        self.unlock_mutex()

    def request_write(self, message: str, signal: QtCore.SignalInstance):
        """Add a write request to the queue.

        Parameters
        ----------
        message : str
            The message to write.
        signal : QtCore.SignalInstance
            The signal to emit a message when successfully written. The signal must take
            a string as the first argument and a object as the second. The second
            argument is the datetime object indicating the time the query was placed.
        """
        self.lock_mutex()
        self.queue.put((message, signal, False))
        self.unlock_mutex()

    def run(self):
        self.mutex = QtCore.QMutex()
        self.queue = queue.Queue()

        self.sigStarted.emit()
        self.aborted.clear()
        measure(
            **self.measure_params,
            updatesignal=self.sigUpdated,
            heatingsignal=self.sigHeating,
            abortflag=self.aborted,
            queue=self.queue,
        )
        self.sigFinished.emit()


class CommandWidget(*uic.loadUiType("command.ui")):
    sigReply = QtCore.Signal(str, object)

    def __init__(self, measure_thread: MeasureThread):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("Lakeshore 325")

        self.write_btn.clicked.connect(self.write)
        self.query_btn.clicked.connect(self.query)

        self.measure_thread = measure_thread

        self.sigReply.connect(self.set_reply)

    @property
    def input(self) -> str:
        return self.text_in.toPlainText().strip()

    @QtCore.Slot(str, object)
    def set_reply(self, message: str, _: datetime.datetime):
        self.text_out.setPlainText(message)

    @QtCore.Slot()
    def write(self):
        if self.measure_thread.isRunning():
            self.measure_thread.request_write(self.input, self.sigReply)
        else:
            handler = RequestHandler(RESOURCE_LAKESHORE)
            handler.open()
            q = queue.Queue()
            q.put((self.input, self.sigReply, False))
            communicate(handler, q)
            handler.close()

    @QtCore.Slot()
    def query(self):
        if self.measure_thread.isRunning():
            self.measure_thread.request_query(self.input, self.sigReply)
        else:
            handler = RequestHandler(RESOURCE_LAKESHORE)
            handler.open()
            q = queue.Queue()
            q.put((self.input, self.sigReply, True))
            communicate(handler, q)
            handler.close()


class MainWindow(*uic.loadUiType("main.ui")):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("R-T Measurement")
        self.file_btn.clicked.connect(self.choose_file)
        self.start_btn.clicked.connect(self.toggle_measurement)

        self.plot = PlotWindow()
        self.actionplot.triggered.connect(self.toggle_plot)

        self.actionmanual.toggled.connect(self.manual_toggled)

        self.measurement_thread = MeasureThread()
        self.measurement_thread.sigStarted.connect(self.started)
        self.measurement_thread.sigStarted.connect(self.plot.started)
        self.measurement_thread.sigHeating.connect(self.plot.started_heating)
        self.measurement_thread.sigUpdated.connect(self.plot.update_data)
        self.measurement_thread.sigUpdated.connect(self.updated)
        self.measurement_thread.sigFinished.connect(self.finished)

        self.command_widget = CommandWidget(self.measurement_thread)
        self.actioncommand.triggered.connect(self.command_widget.show)

    @property
    def measurement_parameters(self) -> dict:
        return {
            "filename": self.file_line.text(),
            "tempstart": self.spin_start.value(),
            "tempend": self.spin_end.value(),
            "coolrate": self.spin_rate.value(),
            "heatrate": self.spin_rateh.value(),
            "curr": self.spin_curr.value() * 1e-3,
            "manual": self.actionmanual.isChecked(),
            "delay": self.spin_delay.value(),
            "nplc": self.spin_delta.value(),
            "mode": self.mode_combo.currentIndex(),
            "resetlake": self.actionresetlake.isChecked(),
            "resetkeithley": self.actionresetkeithley.isChecked(),
            "pmode": self.pmode_combo.currentIndex(),
        }

    @QtCore.Slot()
    def manual_toggled(self):
        for w in (
            self.spin_start,
            self.spin_end,
            self.spin_rate,
            self.spin_rateh,
            self.spin_delay,
        ):
            w.setDisabled(self.actionmanual.isChecked())

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
                "T1 must be lower than T2",
            )
            return

        if params["manual"]:
            msg = "\n".join(
                [
                    f"Save to {params['filename']}",
                    f"Source Current {params['curr']} A",
                    f"NPLC {params['nplc']}",
                    "Manual Control",
                ]
            )
        else:
            handler = RequestHandler(RESOURCE_LAKESHORE)
            handler.open()
            temperature = float(handler.query(f"KRDG? {TEMP_SENSOR_LOOP}").strip())
            handler.close()
            msg = "<br>".join(
                [
                    f"Save to {params['filename']}",
                    f"Source Current {params['curr']} A",
                    f"NPLC {params['nplc']}",
                    f"Current Temperature (T{TEMP_SENSOR_LOOP}) {temperature:.2f} K",
                    "<br><b>Measurement Steps</b>",
                    f"[1] Ramp to {params['tempstart']} K, {params['coolrate']} K/min",
                    f"[2] Wait {params['delay']} min",
                    f"[3] Ramp to {params['tempend']} K, {params['heatrate']} K/min",
                    "<br><b>Estimated Timeline</b>",
                    *_estimated_time_info(
                        temperature,
                        params["tempstart"],
                        params["tempend"],
                        params["coolrate"],
                        params["heatrate"],
                        params["delay"],
                        offset=10.0,
                    ),
                ]
            )

        ret = QtWidgets.QMessageBox.question(
            self, "Confirm Measurement Parameters", msg
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
            self.file_btn,
            self.spin_delta,
            self.spin_curr,
            self.spin_start,
            self.spin_end,
            self.spin_delay,
            self.spin_rate,
            self.spin_rateh,
            self.mode_combo,
            self.actionmanual,
            self.pmode_combo,
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
            self.file_btn,
            self.spin_delta,
            self.spin_curr,
            self.spin_start,
            self.spin_end,
            self.spin_delay,
            self.spin_rate,
            self.spin_rateh,
            self.mode_combo,
            self.actionmanual,
            self.pmode_combo,
        ):
            w.setDisabled(False)
        self.start_btn.setText("Start Measurement")
        self.statusBar().setVisible(False)
        self.manual_toggled()

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
        self.command_widget.close()
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
