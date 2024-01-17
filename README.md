# PhysLab DAQ

## Introduction

Data acquisition program for 4-probe resistance measurements, written in python.

Uses the Keithley 2450 SourceMeter and Lakeshore 325 temperature controller.

Assumes that the GPIB-USB-HS driver is installed.

## Running

* Create a conda environment with the packages in `environment.yml`.
* Modify `start.bat` to match the path to your conda installation.

## Notable changes

Most of the measurement process follows the original C++ program. However, there are some minor changes.

* Data are now saved as a `.csv` file.
* Selecting an existing `.csv` file from previous measurements will not overwrite its contents. New data is appended after the last row.
* Due to the delay (dependent on NPLC, counts, etc.) of the SourceMeter output, the measuring interval parameter did not reflect true measurement intervals. Therefore, the default delta value is set to 0. The minimum delay time at delta 0 depends on the measurement method.
* Time averaging functionality is removed.
* Some PID parameters have been adjusted.
* Two additional measurement modes are added.
