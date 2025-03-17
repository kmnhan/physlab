# PhysLab DAQ

NOTE: Please contact <khan@kaist.ac.kr> before making any changes to the program or experimental setup!

## Introduction

Data acquisition program for 4-probe resistance measurements, written in python.

Uses the Keithley 2450 SourceMeter and Lakeshore 325 temperature controller.

Assumes that the GPIB-USB-HS driver is installed.

Logs 4-wire resistance and temperature data to a `.csv` file while controlling the temperature. The sequence is as follows:

1. Set ramp rate to [Rate 1] K/min
2. Setpoint [Temperature 1] K
3. Once temperature is within ±0.3 K of [Temperature 1] K, wait for [Delay] min
4. Set ramp rate to [Rate 2] K/min
5. Setpoint [Temperature 2] K
6. Once temperature is within ±0.3 K of [Temperature 2] K, end the measurement

There are three measurement modes for the resistance: the offset-compensated ohms method, the current reversal method, and the delta method. For more information, refer to the Keithley Low Level Measurements Handbook, section 3-2 and 3-3.

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
