import os
import pandas as pd

def sort_datasets(vehicles, csv_files, folder):
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

def save_datasets(raw_data, output_folder):
    for vehicle, df in raw_data.items():
        raw_output_file = os.path.join(output_folder, f"{vehicle}_raw.csv")
        df.to_csv(raw_output_file)
        print(f"Saved raw {vehicle} data to {raw_output_file}")

def main():
    input_folder = 'intermediate_datafiles'
    output_folder = 'raw_data'
    vehicles = ['train', 'bus', 'metro', 'tram', 'car','walking']

    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    data_by_vehicle = sort_datasets(vehicles, csv_files, input_folder)
    raw_data = concat_data_per_vehicle(data_by_vehicle)
    save_datasets(raw_data, output_folder)

if __name__ == '__main__':
    main()
