import os
import numpy as np

data_dir = 'data'

data_sets = {}


def main():
    dir_contents = os.listdir(data_dir)
    for file in dir_contents:
        data = np.loadtxt(os.path.join(data_dir, file))
        data_sets[file.replace(".txt", "")] = data
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
