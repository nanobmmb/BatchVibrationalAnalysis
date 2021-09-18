## Data processing pipeline

### Project structure

#### Directories:
* `/data`: directory housing input data in TXT formats with the name `{Sample}_{Number}.txt`
* `/output`: create this directory (folder) where the main script will save the generated spreadsheets of procesed data 

#### Files:
* `/main.py` the main python file. can be run with `python3 main.py` while in the project's root directory
* `/requirements.txt` the file describing the necessary packages for the python script to run
* `/spring_constantx.xlsx` excel sheet with spring constants for different data sets and subsets

### Running the program
* Make sure python3 is installed (https://www.python.org/downloads/). Try running `python3` on the command line. It will either launch python or ask you to install it (Windows)
* Open a command prompt inside this directory. You can do this by typing `cmd` and hitting Enter in the location address bar in windows explorer
![cmd-img](img/open_cmd.png)
* create a virtual environment by executing the command below. This will create a new directory named `venv` in the folder
```
python -m venv venv
```
* activate the virtual environment and install the prerequisites by running the command illustrated below
![activate-install](img/activate_and_install.png)
* Run `python main.py [-f]` to run the script, including `-f` will apply a high-pass butterworth filter with `fc=0.06103515625`
![img.png](img/running.png)

### Editing parameters
* To make changes to code, make sure PyCharm (https://www.jetbrains.com/pycharm/) or another suitable integrated developement environment (IDE) is installed
* See comments in code for locations to change sample frequency, cutoff frequency, and and Highpass Butterworth filter window size (window size controls the number of windows)
