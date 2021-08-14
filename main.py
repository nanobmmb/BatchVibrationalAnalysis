import os
import re
from collections import defaultdict
import numpy as np
from numpy.lib.function_base import gradient
import pandas as pd
import matplotlib as mpl
from copy import deepcopy
from functools import cache
import time
from matplotlib import pyplot as plt

mpl.use('TkAgg')
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
def process_dataset(data_set_name):
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
            samples_per_segment = 16384
            f, pxx = get_psd(measurement, fs, samples_per_segment)
            subset[f"psd_{key}"] = deepcopy(pxx)
            subset[f"psd_{key}_f"] = f
            subset[f"psd_{key}_rms"], subset[f"psd_{key}_crms"] = rms_from_psd(deepcopy(pxx), 1/samples_per_segment)
    return ds


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sets = ['N_FN', 'N_PEG', 'N_COL6', 'BareSi']
    TIME_WORKBOOKS = [
        'power_yW', 'force_pn', 'velocity_pm_per_s_moving_avg', 'acceleration_pg_moving_avg',
    ]
    TIME_SHEETS = [zip([book_name]*len(sets), sets) for book_name in TIME_WORKBOOKS]
    PSD_WORKBOOKS = [f"psd_{book}" for book in TIME_WORKBOOKS]
    PSD_SHEETS = [zip([book_name]*len(sets), sets) for book_name in PSD_WORKBOOKS]
    CRMS_WORKBOOKS = [f"{book}_crms" for book in PSD_WORKBOOKS]
    CRMS_SHEETS = [zip([book_name]*len(sets), sets) for book_name in CRMS_WORKBOOKS]
    RMS_WORKBOOKS = [f"{book}_rms" for book in PSD_WORKBOOKS]
    RMS_SHEETS = [zip([book_name]*len(sets), sets) for book_name in CRMS_WORKBOOKS]
    # make the time workbook
    processed_sets = []
    make_time_workbooks = True
    for workbook_name in TIME_WORKBOOKS:
        if not make_time_workbooks:
            break
        writer = pd.ExcelWriter(f"{workbook_name}.xlsx")
        for set_name in sets:
            print(f"Creating {workbook_name}.xlsx:{set_name}")
            t = time.time_ns()
            processed = process_dataset(set_name)
            print(f"Data set processing call took {(time.time_ns()-t) / 1e6} ms")
            sheet = {"t(s)": processed[0]['time'][:len(processed[0][workbook_name])]}
            # add the force columns
            seq_labels = []
            for subset in processed:
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
            df.to_excel(writer, sheet_name=set_name)
            print(f"Done.")
        print("Writing to disk...")
        t = time.time_ns()
        writer.save()
        print(f"Writing to disk took {(time.time_ns()-t)/1e6} ms")

    make_psd_workbooks = True
    for workbook_name in PSD_WORKBOOKS:
        if not make_psd_workbooks:
            break
        writer = pd.ExcelWriter(f"{workbook_name}.xlsx")
        for set_name in sets:
            print(f"Creating {workbook_name}.xlsx:{set_name}")
            t = time.time_ns()
            processed = process_dataset(set_name)
            print(f"Data set processing call took {(time.time_ns()-t) / 1e6} ms")
            sheet = {"f(Hz)": processed[0][f"{workbook_name}_f"][:len(processed[0][workbook_name])]}
            # add the force columns
            seq_labels = []
            for subset in processed:
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
            df.to_excel(writer, sheet_name=set_name)
            print(f"Done.")
        print("Writing to disk...")
        t = time.time_ns()
        writer.save()
        print(f"Writing to disk took {(time.time_ns()-t)/1e6} ms")

    make_crms_workbooks = True
    for workbook_name in CRMS_WORKBOOKS:
        if not make_crms_workbooks:
            break
        writer = pd.ExcelWriter(f"{workbook_name}.xlsx")
        for set_name in sets:
            print(f"Creating {workbook_name}.xlsx:{set_name}")
            t = time.time_ns()
            processed = process_dataset(set_name)
            print(f"Data set processing call took {(time.time_ns()-t) / 1e6} ms")
            sheet = {"f(Hz)": processed[0][f"{workbook_name.replace('_crms', '_f')}"][:len(processed[0][workbook_name])]}
            # add the force columns
            seq_labels = []
            for subset in processed:
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
            df.to_excel(writer, sheet_name=set_name)
            print(f"Done.")
        print("Writing to disk...")
        t = time.time_ns()
        writer.save()
        print(f"Writing to disk took {(time.time_ns()-t)/1e6} ms")
    
    make_rms_workbooks = True
    for workbook_name in RMS_WORKBOOKS:
        if not make_rms_workbooks:
            break
        writer = pd.ExcelWriter(f"{workbook_name}.xlsx")
        for set_name in sets:
            print(f"Creating {workbook_name}.xlsx:{set_name}")
            t = time.time_ns()
            processed = process_dataset(set_name)
            print(f"Data set processing call took {(time.time_ns()-t) / 1e6} ms")
            sheet = {"f(Hz)": processed[0][f"{workbook_name.replace('_rms', '_f')}"][:len(processed[0][workbook_name])]}
            # add the force columns
            seq_labels = []
            for subset in processed:
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
            df.to_excel(writer, sheet_name=set_name)
            print(f"Done.")
        print("Writing to disk...")
        t = time.time_ns()
        writer.save()
        print(f"Writing to disk took {(time.time_ns()-t)/1e6} ms")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
