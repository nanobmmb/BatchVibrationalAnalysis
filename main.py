import os
import re
import sys
from collections import defaultdict
import numpy as np
from scipy.signal import butter, sosfilt
import pandas as pd
import matplotlib as mpl
from copy import deepcopy
from functools import cache
import time
from itertools import chain
import argparse


mpl.use('TkAgg')
from scipy import signal

data_dir = 'data'

data_sets = defaultdict(list)

# initialize the butterworth filter
cutoff_freq_hz = 0.06103515625  # cutoff frequency
bw_sos = butter(
    6,
    cutoff_freq_hz,
    btype="highpass",
    analog=False,
    output='sos',
    fs=100  # sample frequency in Hz
)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def get_psd(np_arr, fs, samples_per_segment):
    f, psd = signal.welch(np_arr,
                          fs=fs,
                          nperseg=samples_per_segment,
                          window='hanning',
                          detrend=False
                          )
    return f, psd


def rms_from_psd(psd, frequency_step):
    rms = psd * frequency_step
    rms = np.cumsum(rms)
    crms = (rms ** 0.5) * 10
    rms = [crms[i] - crms[i-1] for i in range(1,len(crms))]
    return rms, crms


@cache
def process_dataset(data_set_name, filter=False):
    """
    This function will look inside the "/data" folder for files with a name matching `{data_set_name}_{number}.xslx`.
    The function will then do some signal conditioning (moving averages) and calculate:
    * displacement (moving avg)
    * velocity (moving avg)
    * acceleration (moving avg)
    * power
    The function uses the `@cache` decorator, which implements an LRU cache so that data sets are only processed once.
    When the function is called with the same argument twice, it quickly returns the result the second time.
    :param data_set_name:
    :return:
    """
    dir_contents = os.listdir(data_dir)
    # read excel contents
    ds = []
    constants = pd.read_excel("spring_constants.xlsx")
    for file_name in dir_contents:
        if not file_name.startswith(data_set_name):
            continue
        data = np.loadtxt(os.path.join(data_dir, file_name))
        name_re = re.compile(r'^(.*)_(\d+).txt$')
        name = name_re.search(file_name)
        set_name = name.group(1)
        seq_num = int(name.group(2))
        row_index_of_sequence_in_constants = constants[data_set_name][constants[data_set_name] == seq_num].index[0]
        col_name = constants.columns[constants.columns.get_loc(set_name) + 1]
        spring_const = constants.get(col_name)[row_index_of_sequence_in_constants]
        ds.append(
            {"time": data[:, 0], "force_pn": data[:, 1], "seq": seq_num, "spring_const": spring_const})

    # signal conditioning and power calculation
    for subset in ds:
        subset['displacement_pm'] = 1000 * subset["force_pn"]/subset['spring_const']  # calc displacement in pm
        subset['displacement_pm_moving_avg'] = moving_average(subset['displacement_pm'], 10)
        displacement_against_time = np.column_stack((subset['time'][:-9], subset['displacement_pm_moving_avg']))
        velocity_array, _ = np.gradient(displacement_against_time)
        subset['velocity_pm_per_s'] = velocity_array[:, 1] * 100
        subset['velocity_pm_per_s_moving_avg'] = moving_average(subset['velocity_pm_per_s'], 10)
        velocity_against_time = np.column_stack((subset['time'][:-18], subset['velocity_pm_per_s_moving_avg']))
        acceleration_array, _ = np.gradient(velocity_against_time)
        subset['acceleration_pm_per_s^2'] = acceleration_array[:, 1] * 100
        subset['acceleration_pm_per_s^2_moving_avg'] = moving_average(subset['acceleration_pm_per_s^2'], 10)
        subset['acceleration_pg_moving_avg'] = subset['acceleration_pm_per_s^2_moving_avg'] / 9.81
        velocity_len = len(subset['velocity_pm_per_s'])
        subset['power_yW'] = np.multiply(subset['force_pn'][:velocity_len], subset['velocity_pm_per_s'])

    # FFT and PSD calculations
    ts = 0.01  # sample time in seconds
    fs = 1 / ts
    signal_keys = ['force_pn', 'velocity_pm_per_s_moving_avg', 'acceleration_pg_moving_avg', 'power_yW']
    for subset in ds:
        for key in signal_keys:
            measurement = subset[key]
            time = subset['time'][:len(measurement)]
            samples_per_segment = 16384 # samples per window
            if filter:
                filtered_measurement = sosfilt(bw_sos, measurement)
                measurement = filtered_measurement
            f, pxx = get_psd(measurement, fs, samples_per_segment)
            subset[f"psd_{key}"] = deepcopy(pxx)
            subset[f"psd_{key}_f"] = f
            subset[f"psd_{key}_rms"], subset[f"psd_{key}_crms"] = rms_from_psd(deepcopy(pxx), 1/samples_per_segment)
    return ds


def _init_excel_sheet(workbook_name: str, processed_data):
    """
    A function to initiate a new excel sheet.
    This will use the workbook name to determine if the sheet if a time or frequency domain sheet
    And will use the processed data dictionary to determine the length of the independent axis
    :param workbook_name:
    :param processed_data:
    :return:
    """
    if workbook_name.startswith("psd_"):
        label = "f(Hz)"
        axis_name = workbook_name.replace("_crms", "").replace("_rms", "")
        axis_name = f"{axis_name}_f"
    else:
        label = "t(s)"
        axis_name = "time"
    return {label: processed_data[0][axis_name][:len(processed_data[0][workbook_name])]}


def _populate_excel_sheet(workbook_name, processed_data, excel_writer, sheet, sheet_name):
    # add the force columns
    seq_labels = []
    for subset in processed_data:
        seq_label = str(subset['seq'])
        seq_labels.append(seq_label)
        sheet[seq_label] = subset[workbook_name]
    # create the averages
    avg = []
    std_dev = []
    for i in range(len(sheet[seq_labels[0]])):
        row = [sheet[label][i] for label in seq_labels]
        avg.append(np.mean(row))
        std_dev.append(np.std(row))
    sheet['avg'] = np.array(avg)
    sheet['std_dev'] = np.array(std_dev)
    df = pd.DataFrame(sheet)
    df.to_excel(excel_writer, sheet_name=sheet_name)


def main(filter):
    sets = ['N_FN', 'N_PEG', 'N_COL6', 'BareSi']
    time_workbooks = [
        'power_yW', 'force_pn', 'velocity_pm_per_s_moving_avg', 'acceleration_pg_moving_avg',
    ]
    psd_workbooks = [f"psd_{book}" for book in time_workbooks]
    crms_workbooks = [f"{book}_crms" for book in psd_workbooks]
    rms_workbooks = [f"{book}_rms" for book in psd_workbooks]
    # to skip a workbook, remove it from the chain
    workbooks = chain(time_workbooks, psd_workbooks, crms_workbooks, rms_workbooks)
    for workbook_name in workbooks:
        workbook_path = os.path.join("output", f"{workbook_name}.xlsx")
        writer = pd.ExcelWriter(workbook_path)
        for set_name in sets:
            print(f"Creating {workbook_name}.xlsx:{set_name}")
            t = time.time_ns()
            processed = process_dataset(set_name, filter=filter)
            print(f"Data set processing call took {(time.time_ns()-t) / 1e6} ms")
            sheet = _init_excel_sheet(workbook_name, processed)
            _populate_excel_sheet(workbook_name, processed, writer, sheet, set_name)
        print("Writing to disk...")
        t = time.time_ns()
        writer.save()
        print(f"Writing to disk took {(time.time_ns()-t)/1e6} ms")


if __name__ == '__main__':
    #  when the script is called, run the main function
    parser = argparse.ArgumentParser(description='AFM frequency analysis processing pipeline')
    parser.add_argument('-f', '--filter', default=False, help="apply BW highpass filter", action="store_true")
    args = parser.parse_args()
    main(args.filter)
