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
* Due to the delay (~ 1.4 s) of the SourceMeter output, the measuring interval parameter did not reflect true measurement intervals. Now, it has changed to reflect the true  For ins
* Time averaging functionality is removed.
* Some PID parameters have been adjusted.
