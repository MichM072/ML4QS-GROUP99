import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import re
import warnings
from util.util import write_parquet, read_parquet

warnings.filterwarnings('ignore')

class FeatureCreatorNF:
    def __init__(self, path):
        self.intermediate_path = path
        print("WARNING THIS FEATURE CREATOR USE NO FOURIER FEATURES")

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

    def create_features(self, df, name=None, overwrite=False):

        if name is None:
            name = "combined_features"


        parquet_path = os.path.join(self.intermediate_path, f'{name}.parquet')

        if os.path.exists(parquet_path) and not overwrite:
            print(f"Combined features already exist at {parquet_path}")
            df = read_parquet(parquet_path)
            print(f"Loaded combined features from {parquet_path}")
            print("If this was not intended, rerun create_features with overwrite=True")
            return df

        df = df.copy()
        # Process the data
        session_features = self.process_transportation_data(df, window_size=120)

        # Save the result
        write_parquet(session_features, self.intermediate_path / f'{name}.parquet')

        return session_features

class FeatureCreatorUpdated:
    def __init__(self, path):
        self.output_dir = path
        self.sampling_rate = 20  # Hz sampling for FFT
        self.eps_length = 256  # FFT window length
        self.min_samples_per_recording = 10  # skip recordings shorter than this (samples)
        self.window_size = 120  # rolling window size for temporal
        os.makedirs(self.output_dir, exist_ok=True)

    # first foureir
    @staticmethod
    def drop_unwanted_columns(df):
        # Drop columns not needed for spectral analysis
        # TODO: Make this prettier like in the other classes
        drop_cols = [c for c in df.columns if (
                'location' in c.lower() or
                'proximity_phone_Distance' in c or
                c.startswith('label') or
                c.startswith('transport_mode_') or
                c.startswith('activity_') or
                c.startswith('time') or
                c.startswith('shifted')
        )]
        return df.drop(columns=drop_cols, errors='ignore')

    @staticmethod
    def add_recording_index(df, max_gap_ms=300):
        recs = [0]
        n = 0
        for i in range(1, len(df)):
            if (df.index[i] - df.index[i - 1]) > timedelta(milliseconds=max_gap_ms):
                n += 1
            recs.append(n)
        df.insert(0, 'recording_number', recs)
        return df

    @staticmethod
    def compute_fft_features(signal, sampling_rate=1, fft_length=256):
        vals = signal.dropna().values
        if len(vals) == 0:
            return pd.Series(dtype=float)
        vals = np.pad(vals, (0, max(0, fft_length - len(vals))))[:fft_length]
        freqs = np.fft.rfftfreq(fft_length, d=1 / sampling_rate)
        amps = np.abs(np.fft.rfft(vals))
        max_f = freqs[np.argmax(amps)]
        w_f = (freqs * amps).sum() / amps.sum() if amps.sum() > 0 else 0
        psd = amps ** 2 / len(amps)
        pdf = psd / psd.sum() if psd.sum() > 0 else psd
        pse = -np.sum(np.log(pdf) * pdf) if np.all(pdf > 0) else 0
        features = {'max_freq': max_f, 'freq_weighted': w_f, 'pse': pse}
        features.update({f"freq_{f:.3f}_Hz": amp for f, amp in zip(freqs, amps)})
        return pd.Series(features)

    @staticmethod
    def clean_fourier(df, n_bins=50):
        base = pd.DataFrame(index=df.index)
        binned = pd.DataFrame(index=df.index)
        prefixes = sorted({c.split('_freq_')[0] for c in df.columns if '_freq_' in c})
        for pre in prefixes:
            cols = [c for c in df.columns if c.startswith(pre)]
            for key in ['freq_0.000_Hz', 'freq_weighted', 'pse']:
                cname = f"{pre}_{key}"
                if cname in df:
                    base[cname] = df[cname]
            amp_cols = [c for c in cols if re.match(rf"{pre}_freq_\d+\.\d*_Hz", c)]
            if amp_cols:
                base[f"{pre}_std"] = df[amp_cols].std(axis=1)
                freqs = [float(c.split('_freq_')[1].split('_Hz')[0]) for c in amp_cols]
                edges = np.linspace(min(freqs), max(freqs), n_bins + 1)
                for i in range(n_bins):
                    sel = [c for c, f in zip(amp_cols, freqs) if edges[i] <= f < edges[i + 1]]
                    binned[f"{pre}_band_{i}"] = df[sel].mean(axis=1) if sel else 0
        return pd.concat([base, binned], axis=1)

    # now the temporal features and magnitude etc
    @staticmethod
    def calculate_magnitude(df, prefix):
        # Vector magnitude for 3-axis data
        return np.sqrt(df[f"{prefix}_X"] ** 2 + df[f"{prefix}_Y"] ** 2 + df[f"{prefix}_Z"] ** 2)

    @staticmethod
    def temporal_aggregation(series, window_size=120, aggs=['mean', 'median', 'min', 'max', 'std']):
        out = pd.DataFrame(index=series.index)
        for func in aggs:
            if func == 'std':
                out[f"{series.name}_{func}"] = series.rolling(window_size, min_periods=1).std().fillna(0)
            else:
                out[f"{series.name}_{func}"] = series.rolling(window_size, min_periods=1).agg(func)
        return out

    @staticmethod
    def calculate_direction_changes(s, thresh=10):
        # Flag > thresh-degree jumps, accounting for circular wrap
        d = s.diff().fillna(0)
        d = np.where(d > 180, d - 360, np.where(d < -180, d + 360, d))
        return pd.Series((abs(d) > thresh).astype(int), index=s.index)

    @staticmethod
    def calculate_switches(mag, thresh=0.1):
        # Count sign changes in thresholded diff
        d = mag.diff().fillna(0)
        state = np.where(d > thresh, 1, np.where(d < -thresh, -1, 0))
        sw = np.abs(np.diff(state, prepend=state[0])) > 0
        return pd.Series(sw.astype(int), index=mag.index)

    def process_temporal(self, df, window_size=120):
        sessions = []
        for sid, grp in df.groupby('id'):
            g = grp.copy()
            # Compute magnitudes
            for p in ['acc_phone', 'lin_acc_phone', 'gyr_phone']:
                g[f"{p}_magnitude"] = self.calculate_magnitude(g, p)
            # Compute direction and acceleration switches
            g['direction_changes'] = self.calculate_direction_changes(g['location_phone_Direction'])
            for p in ['acc_phone_magnitude', 'lin_acc_phone_magnitude', 'gyr_phone_magnitude']:
                key = p.replace('_magnitude', '')
                g[f"{key}_switches"] = self.calculate_switches(g[p])
            # Rolling aggregations
            features = [
                'acc_phone_X', 'acc_phone_Y', 'acc_phone_Z',
                'lin_acc_phone_X', 'lin_acc_phone_Y', 'lin_acc_phone_Z',
                'gyr_phone_X', 'gyr_phone_Y', 'gyr_phone_Z',
                'location_phone_Velocity', 'location_phone_Direction',
                'location_phone_Horizontal Accuracy', 'location_phone_Vertical Accuracy',
                'acc_phone_magnitude', 'lin_acc_phone_magnitude', 'gyr_phone_magnitude',
                'direction_changes', 'acc_phone_switches', 'lin_acc_phone_switches', 'gyr_phone_switches'
            ]
            for feat in features:
                if feat in g:
                    g = pd.concat([g, self.temporal_aggregation(g[feat], window_size)], axis=1)
            # Session-level stats: use the correct switch column names
            switch_cols = ['direction_changes', 'acc_phone_switches', 'lin_acc_phone_switches', 'gyr_phone_switches']
            rates = {f"session_{c}_rate": g[c].mean() for c in switch_cols}
            rates['session_avg_velocity'] = g['location_phone_Velocity'].mean()
            rates['session_velocity_std'] = g['location_phone_Velocity'].std()
            for k, v in rates.items():
                g[k] = v
            sessions.append(g)
        return pd.concat(sessions)

    def create_features(self, dataset, name=None, overwrite=False):

        if name is None:
            name_parquet = 'combined_features.parquet'
            name_csv = 'combined_features.csv'
        else:
            name_parquet = f"{name}_combined_features.parquet"
            name_csv = f"{name}_combined_features.csv"

        parquet_path = os.path.join(self.output_dir, name_parquet)

        if os.path.exists(parquet_path) and not overwrite:
            print(f"Combined features already exist at {parquet_path}")
            df = read_parquet(parquet_path)
            print(f"Loaded combined features from {parquet_path}")
            print("If this was not intended, rerun create_features with overwrite=True")
            return df

        raw_fft = dataset.copy()

        # Load and branch raw data for fourier and temporal (clean only before FFT)
        raw_fft = self.drop_unwanted_columns(raw_fft)  # only affects spectral branch
        raw_fft = raw_fft.dropna()
        raw_fft = self.add_recording_index(raw_fft)

        raw_temp = dataset.copy()  # full data for temporal branch

        fft_rows = []
        for rid, grp in raw_fft.groupby('id'):
            if len(grp) < self.min_samples_per_recording: continue
            feats = {}
            for col in grp.select_dtypes(include='number').columns.drop('recording_number'):
                s = self.compute_fft_features(grp[col], self.sampling_rate, self.eps_length)
                s.index = [f"{col}_{i}" for i in s.index]
                feats.update(s.to_dict())
            feats['id'] = rid
            fft_rows.append(feats)

        df_fft = pd.DataFrame(fft_rows).set_index('id')
        fft_clean = self.clean_fourier(df_fft)

        temp = self.process_temporal(raw_temp, self.window_size)
        temp.reset_index(inplace=True)
        temp.rename(columns={'index': 'timestamp'}, inplace=True)

        if 'id' in temp.columns:
            temp = temp.set_index('id')

        # # Prevent overlapping columns
        # overlapping_cols = set(fft_clean.columns).intersection(set(temp.columns))
        # suffixed_temp = temp.copy()
        # suffixed_temp.rename(columns={col: f"{col}_temp" for col in overlapping_cols}, inplace=True)
        #
        # # Merge on 'id' and ensure cols are not dropped
        # merged = pd.merge(
        #     fft_clean,
        #     suffixed_temp,
        #     left_index=True,
        #     right_index=True,
        #     how='inner'
        # )

        # Merge on 'id'
        merged = fft_clean.join(
            temp,  # temporal features
            how='inner',  # only matching ids
            rsuffix='_temp'  # suffix for any overlapping column names
        )

        # Reset the index back to timestamp
        merged.reset_index(inplace=True)
        merged.rename(columns={'index': 'id'}, inplace=True)
        merged.set_index('timestamp', inplace=True)

        # Save unified features
        # This CSV now contains all FFT-based and rolling temporal features per session (id)
        combined_path = os.path.join(self.output_dir, name_csv)
        merged.to_csv(combined_path)
        print(f"Combined features saved to {combined_path}")
        write_parquet(merged, parquet_path)
        print(f"Combined features saved to {parquet_path}")

        return merged