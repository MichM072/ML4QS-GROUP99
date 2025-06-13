import os
import pandas as pd
import matplotlib.pyplot as plt
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection

folder = 'intermediate_datafiles'
output_folder = 'outlier_data'
vehicles = ['train', 'bus', 'metro', 'tram', 'car', 'walking']
outlier_col_mapping = {
    'chauvenet': '_outlier',
    'mixture': '_mixture',
    'mixture_model': '_mixture',
}

class OutlierDetector:

    def __init__(self, intermediate_dataset):
        self.outlier_detector = None
        self.intermediate_dataset = intermediate_dataset
        self.fitted_data = None
        self.fitted_cols = set()
        self.fitted_detector = None

    @staticmethod
    def select_detector(outlier_type='chauvenet'):
        outlier_type = outlier_type.lower()
        distribution_based = ['chauvenet', 'mixture_model', 'mixture',]
        distance_based = ['simple_distance', 'lof',]

        if outlier_type in distribution_based:
            return DistributionBasedOutlierDetection()
        elif outlier_type in distance_based:
            return DistanceBasedOutlierDetection()
        else:
            raise ValueError(f"Unknown outlier detector type: {outlier_type}")

    def fit_mixture(self, dataframe):
        df = dataframe

        numeric_cols = df.select_dtypes(include='number').columns

        for col in numeric_cols:
            if df[col].isna().mean() > 0.5:
                continue

            mask = df[col].notna()

            filtered_df = self.outlier_detector.mixture_model(df[mask], col)
            df[col + '_mixture'] = pd.NA
            df.loc[filtered_df.index] = filtered_df

            self.fitted_cols.add(col)

        self.fitted_data = df

    def fit_chauvenet(self, dataframe, chauvenet_params = None, C=2):
        df = dataframe

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
            self.fitted_cols.add(col)

        self.fitted_data = df

        # output_file = os.path.join(output_folder, f"{vehicle}_filtered.csv")
        # df.to_csv(output_file)
        # print(df.head())
        # print(f"Saved filtered {vehicle} data to {output_file}")

    def fit(self, cols=None, outlier_detector = 'chauvenet', outlier_params = None, **kwargs):

        if cols:
            df = self.intermediate_dataset[cols].copy()
        else:
            invalid_cols = ['label', 'id', 'time', 'outlier', 'mixture']
            cols = [col for col in self.intermediate_dataset.columns if not any(inv_col in col for inv_col in invalid_cols)]
            df = self.intermediate_dataset[cols].copy()

        self.outlier_detector = self.select_detector(outlier_detector)

        if outlier_detector == 'chauvenet':
            self.fit_chauvenet(df, outlier_params, **kwargs)
        elif outlier_detector in ['mixture', 'mixture_model']:
            self.fit_mixture(df)
        else:
            raise NotImplemented(f"Unknown outlier detector type: {outlier_detector}")

        self.fitted_detector = outlier_detector


    def transform(self, outlier_behavior = 'nan'):
        df = self.fitted_data.copy()
        outlier_ext = outlier_col_mapping[self.fitted_detector]
        if outlier_behavior == 'nan':
            for col in self.fitted_cols:
                df.loc[df[col + outlier_ext] == True, col] = float('nan')
        elif outlier_behavior == 'drop':
            # raise NotImplemented(f"Dropping outliers not yet implemented.")
            pass
        else:
            raise ValueError(f"Unknown outlier behavior: {outlier_behavior}")

        # Drop outlier columns
        df.drop(columns=[col + outlier_ext for col in self.fitted_cols], inplace=True)

        resulting_df = self.intermediate_dataset.copy()
        resulting_df.update(df)

        return resulting_df

    def fit_transform(self, outlier_detector = 'chauvenet', cols=None,
                      outlier_params = None, outlier_behaviour='nan', **kwargs):
        """
        Fits the data with specified outlier detection method and parameters, then applies the transformation
        to replace outliers based on the defined behavior.

        Parameters
        ----------
        outlier_detector : str, optional
            The method to use for outlier detection (default is 'chauvenet').

        cols : list, optional
            List of column names to apply outlier detection on. If None, all columns are used.

        outlier_params : dict, optional
            Parameters for the specified outlier detection method. If None, default parameters will be used.

        outlier_behaviour : str, optional
            Defines how outliers are treated during the transformation. Options include 'nan' (default) or
            other supported behaviors.

        **kwargs : dict
            Additional keyword arguments passed to the fit method.

        Returns
        -------
        Any
            Transformed data with outliers handled as per the specified behavior.

        """
        self.fit(cols, outlier_detector, outlier_params, **kwargs)
        return self.transform(outlier_behaviour)