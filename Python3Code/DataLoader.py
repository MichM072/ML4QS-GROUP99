import sys
import os

sys.path.append('../Python3Code/')

from Chapter2.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import datetime
import copy
import re
import numpy as np
import glob
import pandas as pd

class PhyboxDatasetLoader:

    def __init__(self, student, exp_dir = 'ML4QS-Vehicle'):
        self.student = student
        self.exp_dir = exp_dir
        self.vehicles = ['train', 'bus', 'metro', 'tram', 'car', 'scooter', 'bike', 'walking']
        self.dataset_path = self.set_dataset_path(student, exp_dir)

    @staticmethod
    def set_dataset_path(student, exp_dir):
        if os.path.exists(Path(f'./datasets/{exp_dir}/')):
            return Path(f'./datasets/{exp_dir}/')
        else:
            return Path(f'/local/data/{student}/datasets/{exp_dir}/')

    @staticmethod
    def simplify_column_names(df):
        for col in df.columns:
            df.rename(columns={col: col.split('(')[0].strip()}, inplace=True)

    @staticmethod
    def create_labels_bandaid(start_time, end_time, experiment, exp_type):

        df_labels = pd.DataFrame([{
            'experiment': experiment.name,
            'label': exp_type,
            'label_start': int(start_time * 1_000_000_000),
            'label_end': int(end_time),
        }])

        df_labels.to_csv(f'{experiment}/labels.csv', index=False)

    @staticmethod
    def create_labels(start_times, end_times, experiment, exp_types):

        assert len(start_times) == len(end_times), "Missing end times for labels. Make sure that the number of start times and end times are equal."
        assert len(start_times) >= len(exp_types), "There are more types than pauses, the experiment is incomplete."

        rows = []

        if len(start_times) == len(exp_types):
            for start_time, end_time, exp_type in zip(start_times, end_times, exp_types):
                rows.append({
                    'experiment': experiment.name,
                    'label': exp_type,
                    'label_start': int(start_time * 1_000_000_000),
                    'label_end': int(end_time * 1_000_000_000)
                })

        else:
            for start_time, end_time in zip(start_times, end_times):
                rows.append({
                    'experiment': experiment.name,
                    'label': exp_types,
                    'label_start': int(start_time * 1_000_000_000),
                    'label_end': int(end_time * 1_000_000_000)
                })

        df_labels = pd.DataFrame(rows)

        df_labels.to_csv(f'{experiment}/labels.csv', index=False)


    def check_experiment_name(self, experiment_name):
        pattern = '|'.join(self.vehicles)
        vehicle_names = re.findall(pattern, experiment_name.lower())

        return vehicle_names if vehicle_names else None

    def load_phybox_data(self):

        for experiment in self.dataset_path.iterdir():
            start_times = []
            end_times_exp = []
            end_times = []
            if not experiment.is_dir():
                continue
            exp_type = self.check_experiment_name(experiment.name)
            time_csv = experiment / 'meta' / 'time.csv'
            with open(time_csv, 'r') as f:
                for line in f.readlines():
                    if 'START' in line:
                        start_times.append(float(line.split(',')[2]))
                    if 'PAUSE' in line:
                        end_times.append(float(line.split(',')[2]))
                        end_times_exp.append(float(line.split(',')[1]))


            # print(f'Experiment {experiment.name} has type {exp_type} and start time {start_time}')

            if end_times:
                self.create_labels(start_times, end_times, experiment, exp_type)

            if start_times:
                print(f'Processing {experiment.name}')
                # TODO: end times are not correct for missing pause.
                latest_unix_timestamp = 0
                for csv in (x for x in experiment.rglob('*.csv') if 'meta' not in str(x.parent)):
                    df = pd.read_csv(csv, delimiter=',')

                    if not df.columns.__contains__('Time'):
                        # Bandaid fix to prevent invalid frames
                        continue

                    self.simplify_column_names(df)

                    if df.columns.__contains__('unix_timestamp'):
                        continue

                    if end_times:
                        df['unix_timestamp'] = 0
                        for start_time, exp_time in zip(start_times, end_times_exp):
                            mask = df['Time'] <= exp_time
                            df.loc[mask, 'unix_timestamp'] = (
                                        (df.loc[mask, 'Time'] + start_time) * 1_000_000_000).astype(np.int64)
                    else:
                        df['unix_timestamp'] = ((df['Time'] + start_times[0]) * 1_000_000_000).astype(np.int64)
                        latest_unix_timestamp = max(latest_unix_timestamp, df['unix_timestamp'].max())

                    df.to_csv(csv, index=False)

                if not end_times:
                    self.create_labels_bandaid(start_times[0], latest_unix_timestamp, experiment, exp_type)


    def create_dataset(self, experiment_path, granularities=None, overwrite=True):

        if granularities is None:
            granularities = [600, 250]

        DATASET_PATH = Path(experiment_path)

        experiment_name = experiment_path.split('/')[-1].strip()
        RESULT_PATH = Path('./intermediate_datafiles/')
        RESULT_FNAME = f'{experiment_name}_results.parquet'

        if not overwrite and os.path.exists(RESULT_PATH / RESULT_FNAME):
            print(f"Intermediate results file {RESULT_FNAME} already exists. Overwrite is set to False. Skipping this experiment and using available parquet.")
            dataset = pd.read_parquet(RESULT_PATH / RESULT_FNAME)
            return dataset

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

            dataset.add_event_dataset('labels.csv', 'label_start', 'label_end', 'label', 'binary')

            # Get the resulting pandas data table
            dataset = dataset.data_table

            # Plot the data
            # DataViz = VisualizeDataset(__file__)
            #
            # # Boxplot
            # DataViz.plot_dataset_boxplot(dataset,
            #                              ['acc_phone_X', 'acc_phone_Y', 'acc_phone_Z'])
            #
            # # Plot all data
            # DataViz.plot_dataset(dataset,
            #                      ['acc_', 'gyr_', 'lin_', 'location_', 'mag_', 'proximity_phone_'],
            #                      ['like', 'like', 'like', 'like', 'like', 'like', 'like'],
            #                      ['line', 'line', 'line', 'line', 'line', 'line', 'points'])

            # And print a summary of the dataset.
            util.print_statistics(dataset)
            datasets.append(copy.deepcopy(dataset))

            # If needed, we could save the various versions of the dataset we create in the loop with logical filenames:
            # dataset.to_csv(RESULT_PATH / f'chapter2_result_{milliseconds_per_instance}')

        # Make a table like the one shown in the book, comparing the two datasets produced.
        util.print_latex_table_statistics_two_datasets(datasets[0], datasets[1])

        print(dataset.head())

        # Finally, store the last dataset we generated (250 ms).
        dataset.reset_index(inplace=True)
        dataset.rename(columns={'index': 'timestamp'}, inplace=True)
        dataset.to_parquet(RESULT_PATH / RESULT_FNAME, version='2.6', allow_truncated_timestamps=True)

        # Lastly, print a statement to know the code went through

        print('The code has run through successfully!')
        return dataset

    def create_all_datasets(self, granularities=None, overwrite=True):
        dataset = []
        instance = 0
        print("Loading phybox data...")
        self.load_phybox_data()
        print("Done loading phybox data.")
        print("Creating datasets...")
        for experiment in glob.glob(f'{self.dataset_path}/*'):
            if os.path.isdir(experiment):
                print(f'Processing {experiment}')
                intermed_df = self.create_dataset(experiment, granularities, overwrite=overwrite)
                intermed_df.insert(0, 'id', instance)
                dataset.append(intermed_df)
                instance += 1
                print(f'Done processing {experiment}')
        print("Done creating datasets.")
        return pd.concat(dataset, ignore_index=True)



def main():
    dataset_loader = PhyboxDatasetLoader('mmr497', exp_dir='ML4QS-Vehicle-2')
    datasets = dataset_loader.create_all_datasets(overwrite=True)
    datasets.to_parquet('./intermediate_datafiles/ML4QS_combined_results_2.parquet', version='2.6', allow_truncated_timestamps=True)

if __name__ == '__main__':
    main()