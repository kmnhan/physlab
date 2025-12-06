import datetime
import sys

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets, uic


class DiscreteInfiniteLine(pg.InfiniteLine):
    sigDragStarted = QtCore.Signal(object)

    def temp_value(self) -> float:
        if hasattr(self, "_temp_value"):
            return self._temp_value
        return self.value()

    def mouseDragEvent(self, ev):
        if (
            QtCore.Qt.KeyboardModifier.ControlModifier
            not in QtWidgets.QApplication.keyboardModifiers()
        ):
            if self.movable and ev.button() == QtCore.Qt.MouseButton.LeftButton:
                if ev.isStart():
                    self.moving = True
                    self.cursorOffset = self.pos() - self.mapToParent(
                        ev.buttonDownPos()
                    )
                    self.startPosition = self.pos()
                    self.sigDragStarted.emit(self)
                ev.accept()

                if not self.moving:
                    return

                new_position = self.cursorOffset + self.mapToParent(ev.pos())
                if self.angle % 180 == 0:
                    self._temp_value = new_position.y()
                elif self.angle % 180 == 90:
                    self._temp_value = new_position.x()

                self.sigDragged.emit(self)
                if ev.isFinish():
                    self.moving = False
                    self.sigPositionChangeFinished.emit(self)
        else:
            self.setMouseHover(False)
            self.plotItem.mouseDragEvent(ev)


class PlotWindow(*uic.loadUiType("plotting.ui")):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("R-T Measurement Plot")
        self.combo.currentIndexChanged.connect(self.update_axes)
        self.bin_spin.valueChanged.connect(self.update_plot)
        self.avg_spin.valueChanged.connect(self.update_plot)
        self.log_res_check.toggled.connect(self.update_axes)
        self.inv_temp_check.toggled.connect(self.update_axes)
        self.plot_widget.plotItem.vb.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        self.curve0: pg.PlotDataItem = self.plot_widget.plot(pen="c")
        self.curve1: pg.PlotDataItem = self.plot_widget.plot(pen="m")
        self.start_dt = None
        self.line = DiscreteInfiniteLine(
            angle=90,
            movable=True,
            label="",
            labelOpts={
                "position": 0.75,
                "movable": True,
                "color": (200, 200, 100),
                "fill": (200, 200, 200, 50),
            },
        )
        self.target = pg.TargetItem(size=5, movable=False)
        self.line.sigDragged.connect(self.cursor_moved)
        self.line.sigPositionChanged.connect(self.cursor_moved)
        self.cursor_check.toggled.connect(self.toggle_cursor)
        self.cursor_check.setChecked(False)
        self.plot_widget.addItem(self.line)
        self.plot_widget.addItem(self.target)

        self.cooling_check.toggled.connect(self.curve0.setVisible)

        self.reset_data()
        self.update_axes()

    @QtCore.Slot()
    def reset_data(self):
        self._data = [[], [], []]
        # self.about_to_heat = False
        self.t_heat = None
        self.cooling_check.setChecked(True)
        self.cooling_check.setDisabled(True)
        self.curve0.setData(x=None, y=None)
        self.curve1.setData(x=None, y=None)

    @QtCore.Slot()
    def started(self):
        self.start_dt = datetime.datetime.now()
        self.reset_data()

    @QtCore.Slot()
    def started_heating(self):
        self.t_heat = (datetime.datetime.now() - self.start_dt).total_seconds()
        self.cooling_check.setDisabled(False)

    @QtCore.Slot()
    def toggle_cursor(self):
        self.line.setVisible(self.cursor_check.isChecked())
        self.target.setVisible(self.cursor_check.isChecked())

        if self.cursor_check.isChecked():
            xmin, xmax = self.plot_widget.plotItem.viewRange()[0]
            self.line.setValue((xmin + xmax) / 2)
            self.cursor_moved()

    @QtCore.Slot(object, object)
    def update_data(self, dt, data):
        sec = (dt - self.start_dt).total_seconds()
        temp, res, _ = data

        self._data[0].append(sec)
        self._data[1].append(temp)
        self._data[2].append(res)

        self.bin_spin.setMaximum(len(self._data[0]))
        self.avg_spin.setMaximum(len(self._data[0]))

        self.update_plot()

    @QtCore.Slot()
    def update_axes(self):
        self.plot_widget.setLabel(axis="bottom", text=self.xlabel)
        self.plot_widget.setLabel(axis="left", text=self.ylabel)
        self.plot_widget.enableAutoRange()
        self.update_plot()
        self.cursor_moved()

    @QtCore.Slot()
    def update_plot(self):
        if len(self._data[0]) == 0:
            return
        x, y = self.xydata
        if x.size == 0 or y.size == 0:
            return
        self.line.setBounds((np.nanmin(x.values), np.nanmax(x.values)))
        if self.t_heat is None:
            self.curve0.setData(x=x.values, y=y.values)
        else:
            self.curve0.setData(
                x=x.sel(time=slice(None, self.t_heat)).values,
                y=y.sel(time=slice(None, self.t_heat)).values,
            )
            self.curve1.setData(
                x=x.sel(time=slice(self.t_heat, None)).values,
                y=y.sel(time=slice(self.t_heat, None)).values,
            )

        if self.combo.currentIndex() != 2:
            self.plot_widget.plotItem.setLogMode(False, self.log_res_check.isChecked())
        else:
            self.plot_widget.plotItem.setLogMode(False, False)

    @QtCore.Slot()
    def cursor_moved(self):
        if len(self._data[0]) == 0:
            return
        x, y = self.xydata
        # If R-T plot and heating, make cursor print only current values
        if self.combo.currentIndex() == 0 and self.t_heat is not None:
            x = x.sel(time=slice(self.t_heat, None))
            y = y.sel(time=slice(self.t_heat, None))
        x_idx = np.abs(x.values - self.line.temp_value()).argmin()
        xval, yval = x.values[x_idx], y.values[x_idx]

        xv_coord, yv_coord = float(xval), float(yval)
        if self.curve0.opts["logMode"][0]:
            xv_coord = np.log10(np.clip(xv_coord, a_min=1e-15, a_max=None))
        if self.curve0.opts["logMode"][1]:
            yv_coord = np.log10(np.clip(yv_coord, a_min=1e-15, a_max=None))
        self.target.setPos(xv_coord, yv_coord)
        self.line.blockSignals(True)
        self.line.setPos(xv_coord)
        self.line.blockSignals(False)

        self.line.label.setText(
            f"{self.format_value_for_label(self.xlabel, xval)}\n"
            f"{self.format_value_for_label(self.ylabel, yval)}"
        )

    @staticmethod
    def format_value_for_label(label: str, value: float):
        match label:
            case "Temperature (K)":
                return f"T = {value:.5g} [K]"
            case "1/Temperature (K⁻¹)":
                return f"1/T = {value:.5g} [K⁻¹]\nT = {1 / value:.5g} [K]"
            case "Resistance (Ohm)":
                return f"R = {value:.5g} [Ω]"
            case "Time (s)":
                return f"t = {value:.5g} [s]"
            case _:
                return f"{value:.5g}"

    @property
    def dataset(self) -> xr.Dataset:
        return (
            xr.Dataset(
                data_vars={
                    "temp": ("time", self._data[1]),
                    "res": ("time", self._data[2]),
                },
                coords={"time": self._data[0]},
            )
            .rolling(time=self.avg_spin.value(), center=True)
            .mean()
            .coarsen(time=self.bin_spin.value(), boundary="pad")
            .mean()
        )

    @property
    def xydata(self):
        ds = self.dataset.dropna("time")
        match self.combo.currentIndex():
            case 0:
                # R vs T
                if self.inv_temp_check.isChecked():
                    return 1 / ds.temp, ds.res
                return ds.temp, ds.res
            case 1:
                # R vs t
                return ds.time, ds.res
            case _:
                # T vs t
                if self.inv_temp_check.isChecked():
                    return ds.time, 1 / ds.temp
                return ds.time, ds.temp

    @property
    def temp_label(self) -> str:
        if self.inv_temp_check.isChecked():
            return "1/Temperature (K⁻¹)"
        return "Temperature (K)"

    @property
    def xlabel(self) -> str:
        if self.combo.currentIndex() == 0:
            return self.temp_label
        return "Time (s)"

    @property
    def ylabel(self) -> str:
        if self.combo.currentIndex() == 2:
            return self.temp_label
        return "Resistance (Ohm)"


if __name__ == "__main__":
    qapp: QtWidgets.QApplication = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)
    qapp.setStyle("Fusion")

    win = PlotWindow()
    win.show()
    win.activateWindow()

    import random

    def sampledata(i=None):
        if i is None:
            i = 1
        temp = 300 + random.random() - 0.5
        res = 0.5 * random.random() + 0.1
        curr = 1e-3

        temp *= i
        res *= i

        return temp, res, curr

    win.started()

    # win.update_data(datetime.datetime.now(), sampledata())
    # import time

    for i in range(10):
        win.update_data(datetime.datetime.now(), sampledata(i))
        # time.sleep(0.1)

    win.started_heating()
    for i in range(10, -1, -1):
        win.update_data(datetime.datetime.now(), sampledata(i))
        # time.sleep(0.1)

    qapp.exec()
