import os
import pandas as pd

raw_folder = 'raw_data'
filtered_folder = 'outlier_data'

results = []

for filename in os.listdir(filtered_folder):
    filtered_path = os.path.join(filtered_folder, filename)

for filename in os.listdir(raw_folder):
    raw_path = os.path.join(raw_folder, filename)

    raw_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)
    filtered_df = pd.read_csv(filtered_path, index_col=0, parse_dates=True)

    common_cols = raw_df.select_dtypes(include='number').columns.intersection(filtered_df.columns)

    for col in common_cols:
        total = raw_df[col].notna().sum()
        if total == 0:
            continue 
        n_nan = filtered_df[col].isna().sum()
        pct_nan = (n_nan / total) * 100

        results.append({
            'File': filename,
            'Feature': col,
            '% Outliers': round(pct_nan, 2)
        })

outlier_summary = pd.DataFrame(results)
outlier_summary.sort_values(by=['File', 'Feature'], inplace=True)
summary_output_file = 'outlier_summary.csv'
outlier_summary.to_csv(summary_output_file, index=False)
print(f"Outlier summary saved to: {summary_output_file}")
