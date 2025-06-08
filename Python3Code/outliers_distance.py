import os
import pandas as pd
import matplotlib.pyplot as plt
from Chapter3.OutlierDetection import DistanceBasedOutlierDetection

def sort_datasets(vehicles, csv_files, folder='intermediate_datafiles'):
    data_by_vehicle = {vehicle: [] for vehicle in vehicles}
    for file in csv_files:
        for label in vehicles:
            if label in file.lower():
                path = os.path.join(folder, file)
                df = pd.read_csv(path, index_col=0, parse_dates=True)
                data_by_vehicle[label].append(df)
                break
    return data_by_vehicle

def concat_data_per_vehicle(data_by_vehicle):
    concatenated = {}
    for vehicle, df_list in data_by_vehicle.items():
        if df_list:
            concatenated[vehicle] = pd.concat(df_list)
        else:
            concatenated[vehicle] = pd.DataFrame()
    return concatenated

def concat_data_filter(data_by_vehicle):
    concatenated = {}
    for vehicle, df_list in data_by_vehicle.items():
        if df_list:
            df = pd.concat(df_list)
            
            if not df.empty:

                df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                start_time = df.index[0]
                end_time = df.index[-1]

                start_time_plus_10s = start_time + pd.Timedelta(seconds=10)
                end_time_minus_10s = end_time - pd.Timedelta(seconds=10)

                df = df[(df.index > start_time_plus_10s) & (df.index < end_time_minus_10s)]
            
            concatenated[vehicle] = df
        else:
            concatenated[vehicle] = pd.DataFrame()
    return concatenated


##################################################
# OUTLIERS
##################################################


sensor_prefixes = [
    'acc_phone_',
    'lin_acc_phone_',
    'gyr_phone_',
    'mag_phone_',
]

folder = 'intermediate_datafiles'
output_folder = 'outliers2'
vehicles = ['train', 'bus', 'metro', 'tram', 'car','walking']

csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
data_by_vehicle = sort_datasets(vehicles, csv_files, folder)
raw_data = concat_data_per_vehicle(data_by_vehicle)
concat_data = concat_data_filter(data_by_vehicle)

outlier_detector = DistanceBasedOutlierDetection()

cols_to_check_acc = ['acc_phone_X', 'acc_phone_Y', 'acc_phone_Z']
cols_to_check_lin_acc = ['lin_acc_phone_X', 'lin_acc_phone_Y', 'lin_acc_phone_Z']
cols_to_check_gyr = ['gyr_phone_X', 'gyr_phone_Y', 'gyr_phone_Z']
cols_to_check_mag = ['mag_phone_X', 'mag_phone_Y', 'mag_phone_Z']
cols_to_check_accuracy = ['location_phone_Horizontal Accuracy', 'location_phone_Vertical Accuracy']

d_function = 'euclidean'
vehicle_params = {
    'train': {'dmin': 0.15, 'fmin': 0.1},
    'bus': {'dmin': 0.5, 'fmin': 0.1},
    'metro': {'dmin': 0.4, 'fmin': 0.1},
    'tram': {'dmin': 0.45, 'fmin': 0.1},
    'car': {'dmin': 0.1, 'fmin': 0.05},
    'walking': {'dmin': 0.65, 'fmin': 0.1},
}

for vehicle, df in raw_data.items():
    raw_output_file = os.path.join(output_folder, f"{vehicle}_raw.csv")
    df.to_csv(raw_output_file)
    print(f"Saved raw {vehicle} data to {raw_output_file}")


for vehicle, df in concat_data.items():
    print(f"Processing {vehicle} data...")

    vehicle_dmin = vehicle_params[vehicle]['dmin']
    vehicle_fmin = vehicle_params[vehicle]['fmin']

    df = outlier_detector.simple_distance_based(df, cols=cols_to_check_acc, d_function=d_function, dmin=vehicle_dmin, fmin=vehicle_fmin)
    df = outlier_detector.simple_distance_based(df, cols=cols_to_check_lin_acc, d_function=d_function, dmin=vehicle_dmin, fmin=vehicle_fmin)
    df = outlier_detector.simple_distance_based(df, cols=cols_to_check_gyr, d_function=d_function, dmin=vehicle_dmin, fmin=vehicle_fmin)
    df = outlier_detector.simple_distance_based(df, cols=cols_to_check_mag, d_function=d_function, dmin=vehicle_dmin, fmin=vehicle_fmin)

    df_filtered = df[df['simple_dist_outlier'] == False] 

##################################################
##################################################

    output_file = os.path.join(output_folder, f"{vehicle}_filtered.csv")
    df_filtered.to_csv(output_file)
    print(f"Saved filtered {vehicle} data to {output_file}")
