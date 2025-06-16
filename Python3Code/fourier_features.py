import pandas as pd
import numpy as np
import os
from datetime import timedelta
import matplotlib.pyplot as plt
import re

### ---------- CONFIGURATION ----------
dataset_name = 'ML4QS_imputed_results.parquet'
sampling_rate = 20  
min_samples_per_recording = 10
fft_length = 256  

### ---------- CLEANING & UTILS ----------

def read_parquet(path):
    df = pd.read_parquet(path)
    df.set_index('timestamp', inplace=True)
    return df

def drop_unwanted_columns(df):
    drop_cols = [col for col in df.columns if (
        'location' in col.lower() or
        'proximity_phone_Distance' in col or
        col.startswith('label_') or
        col.startswith('transport_mode_') or
        col.startswith('activity_')
    )]
    return df.drop(columns=drop_cols, errors='ignore')

def add_recording_index(df):
    recording_numbers = [0]
    n_recording = 0
    for i in range(2, len(df)):
        time_diff = df.index[i] - df.index[i - 1]
        if time_diff <= timedelta(milliseconds=300):
            recording_numbers.append(n_recording)
        else:
            n_recording += 1
            recording_numbers.append(n_recording)
    recording_numbers.append(n_recording)
    df.insert(0, 'recording_number', recording_numbers)
    return df

def compute_fft_features(signal, sampling_rate=1, fft_length=256):
    signal = signal.dropna().values
    n = len(signal)
    if n == 0:
        return pd.Series(dtype=float)

    if n < fft_length:
        signal = np.pad(signal, (0, fft_length - n))
    else:
        signal = signal[:fft_length]

    freqs = np.fft.rfftfreq(fft_length, d=1 / sampling_rate)
    fft_result = np.fft.rfft(signal)

    real_ampl = np.abs(fft_result)

    max_freq = freqs[np.argmax(real_ampl)]
    freq_weighted = float(np.sum(freqs * real_ampl)) / np.sum(real_ampl)

    PSD = np.square(real_ampl) / len(real_ampl)
    PSD_pdf = PSD / np.sum(PSD)
    pse = -np.sum(np.log(PSD_pdf) * PSD_pdf) if np.all(PSD_pdf > 0) else 0

    feature_dict = {
        'max_freq': max_freq,
        'freq_weighted': freq_weighted,
        'pse': pse
    }

    spectrum_features = {f"freq_{round(f, 3)}_Hz": a for f, a in zip(freqs, real_ampl)}
    feature_dict.update(spectrum_features)

    return pd.Series(feature_dict)

def clean_fourier_features(df, fft_length=256, sampling_rate=20, n_bins=50):
    import numpy as np
    import pandas as pd
    import re

    clean_features_df = df[['recording_number']].copy()
    binned_df = pd.DataFrame(index=df.index)

    signal_prefixes = sorted(set(col.split('_freq_')[0] for col in df.columns if '_freq_' in col))

    for signal in signal_prefixes:
        freq_cols = [col for col in df.columns if col.startswith(f"{signal}_freq_")]
        if not freq_cols:
            continue

        for suffix in ['freq_0.0_Hz', 'freq_weighted', 'pse']:
            col_name = f"{signal}_{suffix}"
            if col_name in df.columns:
                clean_features_df[col_name] = df[col_name]

        amp_cols = [col for col in freq_cols if re.match(rf"{signal}_freq_\d+\.?\d*_Hz", col)]
        if amp_cols:

            clean_features_df[f"{signal}_std"] = df[amp_cols].std(axis=1)

            freqs = [float(col.split('_freq_')[1].split('_Hz')[0]) for col in amp_cols]
            bin_edges = np.linspace(min(freqs), max(freqs), n_bins + 1)
            for i in range(n_bins):
                in_bin = [
                    col for col, f in zip(amp_cols, freqs)
                    if bin_edges[i] <= f < bin_edges[i + 1]
                ]
                col_name = f"{signal}_band_{i}"
                binned_df[col_name] = df[in_bin].mean(axis=1) if in_bin else 0

    df_cleaned = pd.concat([clean_features_df, binned_df], axis=1)
    return df_cleaned

### ---------- VISUALIZATION ----------

def plot_and_save_spectrum(df, signal_prefix, base_filename, recording_id=0, out_dir='figures/fourier'):
    os.makedirs(out_dir, exist_ok=True)

    cols = [col for col in df.columns if re.match(rf"^{signal_prefix}_freq_\d+\.?\d*_Hz", col)]
    if not cols:
        print(f"No frequency features for '{signal_prefix}' in recording {recording_id}")
        return

    cols_sorted = sorted(cols, key=lambda x: float(x.split('_freq_')[1].split('_Hz')[0]))
    try:
        row = df[df['recording_number'] == recording_id].iloc[0]
    except IndexError:
        print(f"Recording {recording_id} not found in df_features.")
        return

    spectrum = row[cols_sorted]
    freqs = [float(c.split('_freq_')[1].split('_Hz')[0]) for c in cols_sorted]

    plt.figure(figsize=(10, 4))
    nonzero_mask = np.array(freqs) > 0
    plt.plot(np.array(freqs)[nonzero_mask], spectrum.values[nonzero_mask])
    plt.title(f'Spectrum for {signal_prefix} - Recording {recording_id}')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()

    filename = f"{base_filename}_{signal_prefix}_rec{recording_id}.pdf".replace('/', '_')
    filepath = os.path.join(out_dir, filename)
    plt.savefig(filepath)
    plt.close()

    print(f" Saved: {filepath}")

### ---------- MAIN SCRIPT ----------

df = read_parquet(dataset_name)
print(df.head())
df = drop_unwanted_columns(df)
df = df.dropna()

grouped = df.groupby('id')

fft_feature_rows = []
recordings_used = []

for recording_id, group in grouped:
    print(f" Processing recording {recording_id} with {len(group)} rows")
    if len(group) < min_samples_per_recording:
        print(f" Skipping recording {recording_id} (too short)")
        continue

    numeric_cols = group.select_dtypes(include='number').columns
    numeric_cols = [col for col in numeric_cols if col != 'recording_number']

    features = {}
    for col in numeric_cols:
        fft_series = compute_fft_features(group[col], sampling_rate=sampling_rate, fft_length=fft_length)
        fft_series.index = [f"{col}_{idx}" for idx in fft_series.index]
        features.update(fft_series)

    features['recording_number'] = recording_id
    fft_feature_rows.append(features)
    recordings_used.append(recording_id)

df_features = pd.DataFrame(fft_feature_rows)

freq_feature_cols = [col for col in df_features.columns if '_freq_' in col]
signal_prefixes = sorted(set(col.split('_freq_')[0] for col in freq_feature_cols))
base_filename = os.path.splitext(os.path.basename(dataset_name))[0]

for signal in signal_prefixes:
    for rec_id in recordings_used:
        plot_and_save_spectrum(df_features, signal_prefix=signal, base_filename=base_filename, recording_id=rec_id)

df_features = clean_fourier_features(df_features, sampling_rate=sampling_rate)

os.makedirs('fourier_data', exist_ok=True)
output_path = f'fourier_data/{base_filename}_fourier.csv'
df_features.to_csv(output_path, index=False)
print(f"\n Fourier features saved to: {output_path}")

