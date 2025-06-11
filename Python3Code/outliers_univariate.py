import os
import pandas as pd
import matplotlib.pyplot as plt
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection


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
# OUTLIERS USING CHAUVEVET CRITERION PER SENSOR
##################################################

folder = 'intermediate_datafiles'
output_folder = 'outlier_data'
vehicles = ['train', 'bus', 'metro', 'tram', 'car', 'walking']

csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]
data_by_vehicle = sort_datasets(vehicles, csv_files, folder)
concat_data = concat_data_filter(data_by_vehicle)

# Sensor-specific Chauvenet constants
chauvenet_params = {
    'acc_phone_': 2,
    'lin_acc_phone_': 2,
    'gyr_phone_': 2.5,
    'mag_phone_': 2.5,
    'location_phone_': 1.8,
    'proximity_phone_': 1.2
}

sensor_prefixes = list(chauvenet_params.keys())
outlier_detector = DistributionBasedOutlierDetection()

for vehicle, df in concat_data.items():
    print(f"Processing {vehicle} data...")

    if df.empty:
        print(f"Skipping {vehicle} (no data).")
        continue

    numeric_cols = df.select_dtypes(include='number').columns

    for col in numeric_cols:
        if df[col].isna().mean() > 0.5:
            continue  # Skip mostly missing columns

        # Detect sensor type by prefix
        C = 2  # default fallback
        for prefix in sensor_prefixes:
            if prefix in col:
                C = chauvenet_params[prefix]
                break

        # Apply Chauvenet criterion
        df = outlier_detector.chauvenet(df, col, C=C)
        df.loc[df[col + '_outlier'] == True, col] = float('nan')

    output_file = os.path.join(output_folder, f"{vehicle}_filtered.csv")
    df.to_csv(output_file)
    print(df.head())
    print(f"Saved filtered {vehicle} data to {output_file}")
