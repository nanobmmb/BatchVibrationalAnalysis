import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from copy import deepcopy

mpl.use('TkAgg')
import timeit
import numpy.fft as fft
from scipy import signal

data_dir = 'data'

data_sets = defaultdict(list)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def get_psd(np_arr, fs, samples_per_segment):
    f, psd = signal.welch(np_arr,
                          fs=fs,
                          nperseg=samples_per_segment,
                          window='hanning',
                          axis=0
                          )
    return f, psd


def rms_from_psd(psd, samples_per_segment):
    rms = psd / samples_per_segment
    rms = np.cumsum(rms)
    rms = (rms ** 0.5) * 10
    return rms


def process_dataset(data_set_name="N_FN"):
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
        subset['velocity_pm/s'] = velocity_array[:, 1] * 100
        subset['velocity_pm/s_moving_avg'] = moving_average(subset['velocity_pm/s'], 10)
        velocity_against_time = np.column_stack((subset['time'][:-18], subset['velocity_pm/s_moving_avg']))
        acceleration_array, _ = np.gradient(velocity_against_time)
        subset['acceleration_pm/s^2'] = acceleration_array[:, 1] * 100
        subset['acceleration_pm/s^2_moving_avg'] = moving_average(subset['acceleration_pm/s^2'], 10)
        subset['acceleration_pg_moving_avg'] = subset['acceleration_pm/s^2_moving_avg'] / 9.81
        velocity_len = len(subset['velocity_pm/s'])
        subset['power_yW'] = np.multiply(subset['force_pn'][:velocity_len], subset['velocity_pm/s'])

    # FFT and PSD calculations
    ts = 0.01  # sample time in seconds
    fs = 1 / ts
    signal_keys = ['force_pn', 'velocity_pm/s_moving_avg', 'acceleration_pm/s^2_moving_avg', 'power_yW']
    for subset in ds:
        for key in signal_keys:
            measurement = subset[key]
            time = subset['time'][:len(measurement)]
            samples_per_segment = 16384
            f, pxx = get_psd(np.column_stack((time, measurement)), fs, samples_per_segment)
            subset[f"psd_{key}"] = deepcopy(pxx[:, 1])
            subset[f"psd_{key}_f"] = f
            subset[f"psd_{key}_crms"] = rms_from_psd(deepcopy(pxx[:, 1]), samples_per_segment)
    return ds


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sets = ['N_PEG', 'N_FN', 'N_COL6', 'BareSi']
    # sets = ['N_COL6', 'BareSi']
    writer = pd.ExcelWriter("processed_output.xlsx")
    for set_name in sets:
        processed = process_dataset(set_name)
        sheet = {"t(s)": processed[0]['time']}
        # add the force columns
        seq_labels = []
        for subset in processed:
            seq_label = str(subset['seq'])
            seq_labels.append(seq_label)
            sheet[seq_label] = subset['force_pn']
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
        df.to_excel(writer, sheet_name=set_name)
        writer.save()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
