import pandas as pd
import numpy as np
import Chapter3.ImputationMissingValues as MisVal

class CustomImputer:
    def __init__(self, imputer='interpolation'):
        self.imputer = self.select_imputer(imputer)
        self.imputed_data = None
        self.verbose = None

    @staticmethod
    def select_imputer(imputer):
        imputer = imputer.lower()
        imputer_base = MisVal.ImputationMissingValues()
        imputer_mapping = {'interpolation': imputer_base.impute_interpolate,
                           'mean': imputer_base.impute_mean,
                           'median': imputer_base.impute_median,
                           }
        return imputer_mapping[imputer]

    def fit(self, dataframe, cols=None, verbose=False):
        self.verbose = verbose

        if cols is None:
            cols_to_impute = [col for col in dataframe.columns if
                            col not in ['id', 'timestamp'] and 'label' not in col and 'outlier' not in col]

        imputed_data = dataframe.copy()

        if self.verbose:
            print("NaN values before imputation:")
            print(imputed_data[cols_to_impute].isna().sum())

        for instance in imputed_data.id.unique():
            instance_mask = imputed_data.id == instance
            for col in cols_to_impute:
                if imputed_data.loc[instance_mask, col].isna().any():
                    if self.verbose:
                        print(f"Imputing {col} for instance {instance}...")
                        print(imputed_data.loc[instance_mask, col].isna().sum() / len(imputed_data.loc[instance_mask, col]))
                    # noinspection PyArgumentList
                    imputed_data[instance_mask] = self.imputer(
                        dataset=imputed_data[instance_mask].copy(),
                        col=col
                    )
        self.imputed_data = imputed_data

    def transform(self):
        if self.imputed_data is None:
            raise ValueError("No imputed data available. Please run fit() first.")
        return self.imputed_data

    def fit_transform(self, dataframe, cols=None, verbose=False):
        self.fit(dataframe, cols, verbose)
        return self.transform()