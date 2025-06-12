import pandas as pd
import numpy as np
from util.VisualizeDataset import VisualizeDataset

outlier_col_mapping = {
    'chauvenet': '_outlier',
    'mixture': '_mixture',
    'mixture_model': '_mixture',
}

class Visualiser:

    def __init__(self):
        self.data_viz = VisualizeDataset(__file__)

    def plot_dataset(self, df, plot_type='full', instances=None, time_slice=None, path=None):
        """
        Plots the dataset using various visualization options based on the specified plot type.
        Plot types include 'day', 'full', 'instance', and 'slice'.
        Day: Plots the dataset for each day separately.
        Full: Plots the entire dataset.
        Instance: Plots the dataset for specific instances.
        Slice: Plots the dataset for a specified time slice, instance slice, or both.

        Parameters
        ----------
        df : pandas.DataFrame
            The DataFrame containing the dataset to be plotted. The index of the
            DataFrame should be in datetime format.
        plot_type : str, optional
            The type of plot to be generated. Accepted values are 'day', 'full',
            'instance', and 'slice'. Default is 'full'.
        instances : list, optional
            A list of unique instance IDs to filter and plot specific instances.
            Used in 'instance' and 'slice' plot types. Default is None.
        time_slice : tuple, optional
            A tuple specifying the start and end times for slicing the dataset.
            Used in 'slice' plot type. Should be in datetime format. Default is None.
        path : str, optional
            A str specifying the path where the figures should be saved to. Default is None.
        """

        if path:
            path = path + '.py'
            self.data_viz = VisualizeDataset(path)

        if plot_type == 'day':
            day_slices = df.groupby(df.index.date)
            for day, day_slice in day_slices:
                self.data_viz.plot_dataset(day_slice,
                            ['acc_', 'gyr_', 'lin_' , 'mag_', 'label'],
                            ['like', 'like', 'like', 'like', 'like'],
                            ['line', 'line', 'line', 'line', 'points'])
        elif plot_type == 'full':
            self.data_viz.plot_dataset(df,
                            ['acc_', 'gyr_', 'lin_' , 'mag_', 'label'],
                            ['like', 'like', 'like', 'like', 'like'],
                            ['line', 'line', 'line', 'line', 'points'])

        elif plot_type == 'instance':
            if instances is None:
                print("No instances specified, using all instances.")
                instances = df.id.unique()
            for instance in instances:
                self.data_viz.plot_dataset(df.loc[df.id == instance],
                                           ['acc_', 'gyr_', 'lin_', 'mag_', 'label'],
                                           ['like', 'like', 'like', 'like', 'like'],
                                           ['line', 'line', 'line', 'line', 'points'])
        elif plot_type == 'slice':
            if instances is None and time_slice is None:
                print("No instances or time slice specified, using all instances.")
                self.plot_dataset(df, plot_type='full')
            elif instances and time_slice is None:
                self.plot_dataset(df.loc[df.id.isin(instances)], plot_type='full')
            elif instances is None and time_slice:
                self.plot_dataset(df.loc[df.index.slice_indexer(time_slice[0], time_slice[1])], plot_type='full')
            else:
                self.plot_dataset(df.loc[df.id.isin(instances) & df.index.slice_indexer(time_slice[0], time_slice[1])],
                                  plot_type='full')
        else:
            print("Plot type not recognized.")

    def plot_outliers(self, df, cols = None, outlier_type='mixture', path=None):

        if path:
            path = path + '.py'
            self.data_viz = VisualizeDataset(path)

        outlier_ext = outlier_col_mapping[outlier_type]

        if cols is None:
            cols = [(col, col + outlier_ext) for col in df.columns if col + outlier_ext in df.columns]
        else:
            cols = [(cols, cols + outlier_ext) for cols in cols]

        if outlier_type == 'mixture' or 'mixture_model':
            for col, outlier_col in cols:
                self.data_viz.plot_dataset(df,
                                  [col, outlier_col],
                                  ['exact', 'exact'],
                                  ['line', 'points'])
        elif outlier_type == 'chauvenet':
            for col, outlier_col in cols:
                self.data_viz.plot_binary_outliers(
                    df, col, outlier_col)
        else:
            raise NotImplemented(f"Unknown outlier detector type: {outlier_type}")

    def plot_imputation(self, df, df_imputed, imputation_type='None', cols = None, path=None):

        if path:
            path = path + '.py'
            self.data_viz = VisualizeDataset(path)

        if imputation_type == 'None':
            imputation_type = 'imputed'

        if cols is None:
            invalid_cols = ['label', 'id', 'time', outlier_col_mapping.values()]
            cols = [col for col in df.columns if not any (inv_col in col for inv_col in invalid_cols)]

        for col in cols:
            self.data_viz.plot_imputed_values(df,
                                        ['original', imputation_type],
                                        col,
                                        df_imputed[col])
