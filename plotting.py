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

        self.curve = self.plot_widget.plot(pen="c")
        self.start_dt = None

        self.line = pg.InfiniteLine(
            angle=90,
            movable=True,
            label="",
            labelOpts=dict(position=0.75, movable=True, fill=(200, 200, 200, 50)),
        )
        self.line.sigPositionChanged.connect(self.cursor_moved)
        self.cursor_check.toggled.connect(self.line.setVisible)
        self.cursor_check.setChecked(False)
        self.plot_widget.addItem(self.line)

        self.reset_data()
        self.update_axes()

    def reset_data(self):
        self._data = [[], [], []]

    @QtCore.Slot()
    def started(self):
        self.start_dt = datetime.datetime.now()
        self.reset_data()

    @QtCore.Slot(object, object)
    def update_data(self, dt, data):
        sec = (dt - self.start_dt).total_seconds()
        temp, res, _ = data

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
        x, y = self.xdata, self.ydata
        self.line.setBounds((min(x), max(x)))
        self.curve.setData(x=x, y=y)

    @QtCore.Slot()
    def cursor_moved(self):
        xdat = np.asarray(self.xdata)
        x_idx = np.abs(xdat - self.line.value()).argmin()
        xval, yval = xdat[x_idx], self.ydata[x_idx]
        self.line.label.setText(
            f"{self.xlabel} {np.format_float_positional(xval)}\n"
            + f"{self.ylabel} {np.format_float_positional(yval)}"
        )

    @property
    def data(self):
        if self.bin_spin.value() == 1:
            return self._data
        else:
            ds = self.dataset
            return [ds.time.values, ds.temp.values, ds.res.values]

    @property
    def dataset(self) -> xr.Dataset:
        return (
            xr.Dataset(
                data_vars=dict(
                    temp=("time", self._data[1]), res=("time", self._data[2])
                ),
                coords=dict(time=self._data[0]),
            )
            .coarsen(time=self.bin_spin.value(), boundary="trim")
            .mean()
        )

    @property
    def xdata(self) -> list[float]:
        if self.combo.currentIndex() == 0:
            # R-T
            return self.data[1]
        else:
            # R-t or T-t
            return self.data[0]

    @property
    def ydata(self) -> list[float]:
        if self.combo.currentIndex() == 2:
            # T-t
            return self.data[1]
        else:
            # R-T or R-t
            return self.data[2]

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

    def sampledata():
        temp = 300 + random.random() - 0.5
        res = 0.1 * random.random() + 0.1
        curr = 1e-3
        return temp, res, curr

    win.started()

    # win.update_data(datetime.datetime.now(), sampledata())
    for i in range(100):
        win.update_data(datetime.datetime.now(), sampledata())
        # time.sleep(0.05)

    qapp.exec()
