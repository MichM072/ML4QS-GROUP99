import pandas as pd
import numpy as np
import warnings
from util.util import write_parquet

warnings.filterwarnings('ignore')

class FeatureCreator:
    def __init__(self, path):
        self.intermediate_path = path

    @staticmethod
    def calculate_magnitude(df, prefix):
        x_col = f"{prefix}_X"
        y_col = f"{prefix}_Y"
        z_col = f"{prefix}_Z"
        return np.sqrt(df[x_col] ** 2 + df[y_col] ** 2 + df[z_col] ** 2)

    @staticmethod
    def temporal_aggregation_no_leak(series, window_size=120, agg_functions=None):

        if agg_functions is None:
            agg_functions = ['mean', 'median', 'min', 'max', 'std']
        result = pd.DataFrame()
        for func in agg_functions:
            if func == 'std':
                result[f"{series.name}_{func}"] = series.rolling(
                    window=window_size, min_periods=1
                ).std().shift(1).fillna(0)
            else:
                result[f"{series.name}_{func}"] = series.rolling(
                    window=window_size, min_periods=1
                ).agg(func).shift(1).fillna(method='bfill')
        return result

    @staticmethod
    def calculate_direction_changes(direction_series, threshold=10):
        direction_diff = direction_series.diff().fillna(0)
        direction_diff = np.where(direction_diff > 180, direction_diff - 360, direction_diff)
        direction_diff = np.where(direction_diff < -180, direction_diff + 360, direction_diff)
        significant_changes = np.abs(direction_diff) > threshold
        return pd.Series(significant_changes.astype(int), index=direction_series.index)

    @staticmethod
    def calculate_acceleration_switches(acceleration_magnitude, threshold=0.1):
        acc_diff = acceleration_magnitude.diff().fillna(0)
        acc_state = np.where(acc_diff > threshold, 1,
                             np.where(acc_diff < -threshold, -1, 0))
        switches = np.abs(np.diff(acc_state, prepend=acc_state[0])) > 0
        return pd.Series(switches.astype(int), index=acceleration_magnitude.index)

    def process_transportation_data(self, df, window_size=120):
        processed_sessions = []

        for session_id, session_data in df.groupby('id'):
            session_df = session_data.copy()

            session_df['acc_phone_magnitude'] = self.calculate_magnitude(session_df, 'acc_phone')
            session_df['lin_acc_phone_magnitude'] = self.calculate_magnitude(session_df, 'lin_acc_phone')
            session_df['gyr_phone_magnitude'] = self.calculate_magnitude(session_df, 'gyr_phone')

            session_df['direction_changes'] = self.calculate_direction_changes(session_df['location_phone_Direction'])
            session_df['acc_switches'] = self.calculate_acceleration_switches(session_df['acc_phone_magnitude'])
            session_df['lin_acc_switches'] = self.calculate_acceleration_switches(session_df['lin_acc_phone_magnitude'])
            session_df['rotation_switches'] = self.calculate_acceleration_switches(session_df['gyr_phone_magnitude'])

            features_to_aggregate = [
                'acc_phone_X', 'acc_phone_Y', 'acc_phone_Z',
                'lin_acc_phone_X', 'lin_acc_phone_Y', 'lin_acc_phone_Z',
                'gyr_phone_X', 'gyr_phone_Y', 'gyr_phone_Z',
                'location_phone_Velocity', 'location_phone_Direction',
                'acc_phone_magnitude', 'lin_acc_phone_magnitude', 'gyr_phone_magnitude',
                'direction_changes', 'acc_switches', 'lin_acc_switches', 'rotation_switches'
            ]

            for feature in features_to_aggregate:
                if feature in session_df.columns:
                    aggregated = self.temporal_aggregation_no_leak(session_df[feature], window_size)
                    session_df = pd.concat([session_df, aggregated], axis=1)

            session_df['session_direction_change_rate'] = session_df['direction_changes'].mean()
            session_df['session_acc_switch_rate'] = session_df['acc_switches'].mean()
            session_df['session_lin_acc_switch_rate'] = session_df['lin_acc_switches'].mean()
            session_df['session_rotation_switch_rate'] = session_df['rotation_switches'].mean()
            session_df['session_avg_velocity'] = session_df['location_phone_Velocity'].mean()
            session_df['session_velocity_std'] = session_df['location_phone_Velocity'].std()

            processed_sessions.append(session_df)

        result_df = pd.concat(processed_sessions, ignore_index=True)
        return result_df

    def create_features(self, df):
        df = df.copy()
        # Process the data
        session_features = self.process_transportation_data(df, window_size=120)

        # Save the result
        write_parquet(session_features, self.intermediate_path / 'session_features_imputed.parquet')
