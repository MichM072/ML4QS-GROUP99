import sys
import os
import glob
import pandas as pd

sys.path.append('../Python3Code/')

from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import re

VEHICLES = ['train', 'bus', 'metro', 'tram', 'car', 'scooter', 'bike', 'walking']

def simplify_column_names(df):
    for col in df.columns:
        df.rename(columns={col: col.split('(')[0].strip()}, inplace=True)

def check_experiment_name(experiment_name):
    pattern = '|'.join(VEHICLES)
    vehicle_name = re.search(pattern, experiment_name.lower())

    return vehicle_name.group(0) if vehicle_name else None

def load_phybox_data(student):

    if os.path.exists(Path('./datasets/ML4QS-Vehicle/')):
        DATASET_PATH = Path('./datasets/ML4QS-Vehicle/')
    else:
        DATASET_PATH = Path(f'/local/data/{student}/datasets/ML4QS-Vehicle/')

    for experiment in DATASET_PATH.iterdir():
        start_time = None
        if not experiment.is_dir():
            continue
        exp_type = check_experiment_name(experiment.name)
        time_csv = experiment / 'meta' / 'time.csv'
        with open(time_csv, 'r') as f:
            for line in f.readlines():
                if 'START' in line:
                    start_time = float(line.split(',')[2])


        print(f'Experiment {experiment.name} has type {exp_type} and start time {start_time}')


        if start_time is not None:
            print(f'Processing {experiment.name}')
            for csv in (x for x in experiment.rglob('*.csv') if 'meta' not in str(x.parent)):
                df = pd.read_csv(csv, delimiter=',')
                simplify_column_names(df)

                # if df.columns.__contains__('unix_timestamp'):
                #     continue

                df['unix_timestamp'] = df['Time'] + start_time
                df['label'] = exp_type
                df.to_csv(csv, index=False)

def create_dataset(student, experiment_name, granularities=None):

    if granularities is None:
        granularities = [60000, 250]

    if os.path.exists(Path('./datasets/ML4QS-Vehicle/')):
        DATASET_PATH = Path('./datasets/ML4QS-Vehicle/')
    else:
        DATASET_PATH = Path(f'/local/data/{student}/datasets/ML4QS-Vehicle/{experiment_name}')

    RESULT_PATH = Path('./intermediate_datafiles/')
    RESULT_FNAME = 'ML4QS_results.csv'

    # Set a granularity (the discrete step size of our time series data). We'll use a course-grained granularity of one
    # instance per minute, and a fine-grained one with four instances per second.

    # We can call Path.mkdir(exist_ok=True) to make any required directories if they don't already exist.
    [path.mkdir(exist_ok=True, parents=True) for path in [DATASET_PATH, RESULT_PATH]]

    print('Please wait, this will take a while to run!')

    datasets = []
    for milliseconds_per_instance in granularities:
        print(
            f'Creating numerical datasets from files in {DATASET_PATH} using granularity {milliseconds_per_instance}.')

        # Create an initial dataset object with the base directory for our data and a granularity
        dataset = CreateDataset(DATASET_PATH, milliseconds_per_instance)

        # Add the selected measurements to it.

        # We add the accelerometer data (continuous numerical measurements) of the phone and the smartwatch
        # and aggregate the values per timestep by averaging the values
        dataset.add_numerical_dataset('Accelerometer.csv', 'unix_timestamp', ['X', 'Y', 'Z'], 'avg', 'acc_phone_')
        dataset.add_numerical_dataset('Linear Accelerometer.csv', 'unix_timestamp', ['X', 'Y', 'Z'], 'avg', 'lin_acc_phone_')

        # We add the gyroscope data (continuous numerical measurements) of the phone and the smartwatch
        # and aggregate the values per timestep by averaging the values
        dataset.add_numerical_dataset('Gyroscope.csv', 'unix_timestamp', ['X', 'Y', 'Z'], 'avg', 'gyr_phone_')

        # We add the amount of light sensed by the phone (continuous numerical measurements) and aggregate by averaging
        dataset.add_numerical_dataset('Location.csv', 'unix_timestamp', ['Latitude','Longitude','Height','Velocity','Direction','Horizontal Accuracy','Vertical Accuracy'], 'avg', 'location_phone_')

        # We add the magnetometer data (continuous numerical measurements) of the phone and the smartwatch
        # and aggregate the values per timestep by averaging the values
        dataset.add_numerical_dataset('Magnetometer.csv', 'unix_timestamp', ['X', 'Y', 'Z'], 'avg', 'mag_phone_')

        # We add the pressure sensed by the phone (continuous numerical measurements) and aggregate by averaging again
        dataset.add_numerical_dataset('Proximity.csv', 'unix_timestamp', ['Distance'], 'avg', 'proximity_phone_')

        # Get the resulting pandas data table
        dataset = dataset.data_table

        # Plot the data
        DataViz = VisualizeDataset(__file__)

        # Boxplot
        DataViz.plot_dataset_boxplot(dataset,
                                     ['acc_phone_X', 'acc_phone_Y', 'acc_phone_Z'])

        # Plot all data
        # DataViz.plot_dataset(dataset,
        #                      ['acc_', 'gyr_', 'hr_watch_rate', 'light_phone_lux', 'mag_', 'press_phone_', 'label'],
        #                      ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'],
        #                      ['line', 'line', 'line', 'line', 'line', 'line', 'points', 'points'])

        # And print a summary of the dataset.
        util.print_statistics(dataset)
        datasets.append(copy.deepcopy(dataset))

        # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
        # dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')

    # Make a table like the one shown in the book, comparing the two datasets produced.
    util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

    # Finally, store the last dataset we generated (250 ms).
    dataset.to_csv(RESULT_PATH / RESULT_FNAME)

    # Lastly, print a statement to know the code went through

    print('The code has run through successfully!')
    return dataset



def main():
    load_phybox_data('mmr497')
    dataset = create_dataset('mmr497', 'ML4QS-Vehicle-Bus 2025-06-04 14-00-16')
    print(dataset)


if __name__ == '__main__':
    main()