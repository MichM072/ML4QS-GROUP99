import os
import pandas as pd

raw_folder = 'raw_data'
filtered_folder = 'outlier_data'

#################################
# percentages
#################################

results = []

for i in range(len(os.listdir(raw_folder))):

    filename_raw = os.listdir(raw_folder)[i]
    raw_path = os.path.join(raw_folder, filename_raw)
    raw_df = pd.read_csv(raw_path, index_col=0, parse_dates=True)

    filename_filtered = os.listdir(filtered_folder)[i]
    filtered_path = os.path.join(filtered_folder, filename_filtered)
    filtered_df = pd.read_csv(filtered_path, index_col=0, parse_dates=True)

    common_cols = raw_df.select_dtypes(include='number').columns.intersection(filtered_df.columns)

    for col in common_cols:
        total = raw_df[col].notna().sum()
        if total == 0:
            continue 
        n_nan = filtered_df[col].isna().sum()
        pct_nan = (n_nan / total) * 100

        results.append({
            'File': filename_raw.split('_')[0],
            'Feature': col,
            '% Outliers': round(pct_nan, 2)
        })

outlier_summary = pd.DataFrame(results)
outlier_summary.sort_values(by=['File', 'Feature'], inplace=True)
summary_output_file = 'outlier_summary.csv'
outlier_summary.to_csv(summary_output_file, index=False)
print(f"Outlier summary saved to: {summary_output_file}")
