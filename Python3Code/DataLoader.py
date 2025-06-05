import sys
import os
import glob
import pandas as pd

sys.path.append('../Python3Code/')

# from Chapter2.CreateDataset import CreateDataset
# from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import re

VEHICLES = ['train', 'bus', 'metro', 'tram', 'car', 'scooter', 'bike', 'walking']
STUDENT = 'mmr497'

def check_experiment_name(experiment_name):
    pattern = '|'.join(VEHICLES)
    vehicle_name = re.search(pattern, experiment_name.lower())

    return vehicle_name.group(0) if vehicle_name else None

if os.path.exists(Path('./datasets/ML4QS-Vehicle/')):
    DATASET_PATH = Path('./datasets/ML4QS-Vehicle/')
else:
    DATASET_PATH = Path(f'/local/data/{STUDENT}/datasets/ML4QS-Vehicle/')

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
        for csv in experiment.glob('[!meta]/**/*.csv'):
            df = pd.read_csv(csv, delimiter=',')
            print(df)
            df['unix_timestamp'] = df['Time (s)'] + start_time
            df.to_csv(csv, index=False)
