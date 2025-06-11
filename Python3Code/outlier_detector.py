import os
import pandas as pd
import matplotlib.pyplot as plt
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection

folder = 'intermediate_datafiles'
output_folder = 'outlier_data'
vehicles = ['train', 'bus', 'metro', 'tram', 'car', 'walking']

class OutlierDetector:

    def __init__(self, intermediate_dataset):
        self.outlier_detector = None
        self.intermediate_dataset = intermediate_dataset
        self.fitted_data = None
        self.fitted_cols = []

    @staticmethod
    def select_detector(outlier_type='chauvenet'):
        outlier_type = outlier_type.lower()
        distribution_based = ['chauvenet', 'mixture_model']
        distance_based = ['simple_distance', 'lof',]

        if outlier_type in distribution_based:
            return DistributionBasedOutlierDetection()
        elif outlier_type in distance_based:
            return DistanceBasedOutlierDetection()
        else:
            raise ValueError(f"Unknown outlier detector type: {outlier_type}")

    def fit_chauvenet(self, chauvenet_params = None, C=2):
        df = self.intermediate_dataset.copy()

        if chauvenet_params is None:
            chauvenet_params = {
                'acc_phone_': 2,
                'lin_acc_phone_': 2,
                'gyr_phone_': 2.5,
                'mag_phone_': 2.5,
                'location_phone_': 1.8,
                'proximity_phone_': 1.2
            }

        sensor_prefixes = list(chauvenet_params.keys())

        numeric_cols = df.select_dtypes(include='number').columns

        for col in numeric_cols:
            if df[col].isna().mean() > 0.5:
                continue  # Skip mostly missing columns

                # Detect sensor type by prefix
            for prefix in sensor_prefixes:
                if prefix in col:
                    C = chauvenet_params[prefix]
                    break

            # Apply Chauvenet criterion
            df = self.outlier_detector.chauvenet(df, col, C=C)
            self.fitted_cols.extend(col)

        self.fitted_data = df

        # output_file = os.path.join(output_folder, f"{vehicle}_filtered.csv")
        # df.to_csv(output_file)
        # print(df.head())
        # print(f"Saved filtered {vehicle} data to {output_file}")

    def fit(self, outlier_detector = 'chauvenet', outlier_params = None, **kwargs):

        self.outlier_detector = self.select_detector(outlier_detector)

        if outlier_detector == 'chauvenet':
            self.fit_chauvenet(outlier_params, **kwargs)
        else:
            raise NotImplemented(f"Unknown outlier detector type: {outlier_detector}")


    def transform(self, outlier_behavior = 'nan'):
        df = self.fitted_data.copy()
        if outlier_behavior == 'nan':
            for col in self.fitted_cols:
                df.loc[df[col + '_outlier'] == True, col] = float('nan')
        elif outlier_behavior == 'drop':
            # raise NotImplemented(f"Dropping outliers not yet implemented.")
            pass
        else:
            raise ValueError(f"Unknown outlier behavior: {outlier_behavior}")

        return df

    def fit_transform(self, outlier_detector = 'chauvenet', outlier_params = None, outlier_behaviour='nan', **kwargs):
        self.fit(outlier_detector, outlier_params, **kwargs)
        return self.transform(outlier_behaviour)