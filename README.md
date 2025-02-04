# PhysLab DAQ

NOTE: Please contact <khan@kaist.ac.kr> before making any changes to the program or experimental setup!

## Introduction

Data acquisition program for 4-probe resistance measurements, written in python.

Uses the Keithley 2450 SourceMeter and Lakeshore 325 temperature controller.

Assumes that the GPIB-USB-HS driver is installed.

## Running

* [Install uv](https://docs.astral.sh/uv/getting-started/installation/)
* Double-click `start.bat` to run the program.

## Notable changes

Most of the measurement process follows the original C++ program. However, there are some minor changes.

* Data is now saved as a `.csv` file.
* Selecting an existing `.csv` file from a previous measurement will *not* overwrite its contents. New data will be appended after the last row.
* Due to the delay (dependent on NPLC, counts, etc.) of the SourceMeter output, the measuring interval parameter did not reflect true measurement intervals. Therefore, the default delta value is set to 0. The minimum delay time at delta 0 depends on the measurement method.
* Time averaging functionality is removed.
* Some PID parameters have been adjusted.
* Two additional measurement modes are added.
