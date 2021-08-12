import os
import re
from collections import defaultdict
import numpy as np
import pandas as pd

data_dir = 'data'

data_sets = defaultdict(list)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def main(data_set_name="N_FN"):
    dir_contents = os.listdir(data_dir)
    # read excel contents
    constants = pd.read_excel("spring_constants.xlsx")
    for file_name in dir_contents:
        if not file_name.startswith(data_set_name):
            continue
        data = np.loadtxt(os.path.join(data_dir, file_name))
        name_re = re.compile(r'^(.*)_(\d+).txt$')
        name = name_re.search(file_name)
        set_name = name.group(1)
        seq_num = int(name.group(2))
        row_index_of_sequence_in_constants = constants.N_FN[constants.N_FN == seq_num].index[0]
        col_name = constants.columns[constants.columns.get_loc(set_name) + 1]
        spring_const = constants.get(col_name)[row_index_of_sequence_in_constants]
        data_sets[set_name].append({"time": data[:, 0], "force_pn": data[:, 1], "seq": seq_num, "spring_const": spring_const})

    for set_name, subsets in data_sets.items():
        for subset in subsets:
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
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
