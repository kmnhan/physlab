import datetime
import sys

import numpy as np
import pyqtgraph as pg
import xarray as xr
from qtpy import QtCore, QtGui, QtWidgets, uic


class PlotWindow(*uic.loadUiType("plotting.ui")):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("R-T Measurement Plot")
        self.setupUi(self)
        self.combo.currentIndexChanged.connect(self.update_axes)
        self.bin_spin.valueChanged.connect(self.update_plot)

        self.plot_widget.plotItem.vb.setCursor(QtGui.QCursor(QtCore.Qt.CrossCursor))

        self.curve0 = self.plot_widget.plot(pen="c")
        self.curve1 = self.plot_widget.plot(pen="m")
        self.start_dt = None

        self.line = pg.InfiniteLine(
            angle=90,
            movable=True,
            label="",
            labelOpts=dict(
                position=0.75,
                movable=True,
                color=(200, 200, 100),
                fill=(200, 200, 200, 50),
            ),
        )
        self.target = pg.TargetItem(size=5)
        self.line.sigPositionChanged.connect(self.cursor_moved)
        self.cursor_check.toggled.connect(self.line.setVisible)
        self.cursor_check.toggled.connect(self.target.setVisible)
        self.cursor_check.setChecked(False)
        self.plot_widget.addItem(self.line)
        self.plot_widget.addItem(self.target)

        self.reset_data()
        self.update_axes()

    @QtCore.Slot()
    def reset_data(self):
        self._data = [[], [], []]
        self.about_to_heat = False
        self.t_heat = None

    @QtCore.Slot()
    def started(self):
        self.start_dt = datetime.datetime.now()
        self.reset_data()

    @QtCore.Slot()
    def started_heating(self):
        self.about_to_heat = True

    @QtCore.Slot(object, object)
    def update_data(self, dt, data):
        sec = (dt - self.start_dt).total_seconds()
        temp, res, _ = data

        if self.about_to_heat:
            self.about_to_heat = False
            self.t_heat = sec

        self._data[0].append(sec)
        self._data[1].append(temp)
        self._data[2].append(res)

        self.bin_spin.setMaximum(len(self._data[0]))

        self.update_plot()

    @QtCore.Slot()
    def update_axes(self):
        self.plot_widget.setLabel(axis="bottom", text=self.xlabel)
        self.plot_widget.setLabel(axis="left", text=self.ylabel)
        self.plot_widget.enableAutoRange()
        self.update_plot()

    @QtCore.Slot()
    def update_plot(self):
        if len(self._data[0]) == 0:
            return
        x, y = self.xydata
        self.line.setBounds((min(x.values), max(x.values)))
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

    @QtCore.Slot()
    def cursor_moved(self):
        x, y = self.xydata
        if self.combo.currentIndex() == 0 and self.t_heat is not None:
            x = x.sel(time=slice(self.t_heat, None))
            y = y.sel(time=slice(self.t_heat, None))
        x_idx = np.abs(x.values - self.line.value()).argmin()
        xval, yval = x.values[x_idx], y.values[x_idx]

        self.target.setPos(xval, yval)
        self.line.label.setText(
            f"{self.xlabel} {np.format_float_positional(xval)}\n"
            + f"{self.ylabel} {np.format_float_positional(yval)}"
        )

        self.line.blockSignals(True)
        self.line.setPos(xval)
        self.line.blockSignals(False)

    @property
    def dataset(self) -> xr.Dataset:
        return (
            xr.Dataset(
                data_vars=dict(
                    temp=("time", self._data[1]),
                    res=("time", self._data[2]),
                ),
                coords=dict(time=self._data[0]),
            )
            .coarsen(time=self.bin_spin.value(), boundary="trim")
            .mean()
        )

    @property
    def xydata(self):
        ds = self.dataset
        if self.combo.currentIndex() == 0:
            # R vs T
            return ds.temp, ds.res
        elif self.combo.currentIndex() == 1:
            # R vs t
            return ds.time, ds.res
        elif self.combo.currentIndex() == 2:
            # T vs t
            return ds.time, ds.temp

    @property
    def xdata(self) -> list[float]:
        if self.combo.currentIndex() == 0:
            # R-T
            return self.dataset.temp
        else:
            # R-t or T-t
            return self.dataset.time

    @property
    def ydata(self) -> list[float]:
        if self.combo.currentIndex() == 2:
            # T-t
            return self.dataset.temp
        else:
            # R-T or R-t
            return self.dataset.res

    @property
    def xlabel(self) -> str:
        if self.combo.currentIndex() == 0:
            return "Temperature (K)"
        else:
            return "Time (s)"

    @property
    def ylabel(self) -> str:
        if self.combo.currentIndex() == 2:
            return "Temperature (K)"
        else:
            return "Resistance (Ohm)"


if __name__ == "__main__":
    qapp: QtWidgets.QApplication = QtWidgets.QApplication.instance()
    if not qapp:
        qapp = QtWidgets.QApplication(sys.argv)

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
    import time

    for i in range(10):
        win.update_data(datetime.datetime.now(), sampledata(i))
        # time.sleep(0.1)

    win.started_heating()
    for i in range(10, -1, -1):
        win.update_data(datetime.datetime.now(), sampledata(i))
        # time.sleep(0.1)

    qapp.exec()
