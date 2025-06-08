import os
import pandas as pd
import matplotlib.pyplot as plt
from util.VisualizeDataset import VisualizeDataset
from Chapter3.OutlierDetection import DistributionBasedOutlierDetection

# Load a single CSV file directly
file_path = os.path.join('intermediate_datafiles', 'tram_martin3_results.csv')
df = pd.read_csv(file_path, index_col=0, parse_dates=True)

sensor_prefixes = [
    'acc_phone_',
    'lin_acc_phone_',
    'gyr_phone_',
    'location_phone_',
    'mag_phone_',
    'proximity_phone_'
]

DataViz = VisualizeDataset('visualizedataset.py')
OutlierDetector = DistributionBasedOutlierDetection()

valid_prefixes = [prefix for prefix in sensor_prefixes if any(col.startswith(prefix) for col in df.columns)]

if valid_prefixes:
    print("Plotting data from:", file_path)
    DataViz.plot_dataset(
        data_table=df,
        columns=valid_prefixes,
        match=['like'] * len(valid_prefixes),
        display=['line'] * len(valid_prefixes),
        title="Sensor Data: Bus Michael 1"
    )
else:
    print("No valid sensor columns found in the file.")

#Outlier Detection with Chauvenet 
target_col = 'acc_phone_X'
if target_col in df.columns:
    df = OutlierDetector.chauvenet(df, target_col, C=2)

    DataViz.plot_binary_outliers(
        data_table=df,
        col=target_col,
        outlier_col=target_col + '_outlier'
    )
else:
    print(f"Column '{target_col}' not found for Chauvenet outlier detection.")

import seaborn as sns
import matplotlib.patches as mpatches


# heatmap
plt.figure(figsize=(12, 6))
cmap = sns.color_palette("viridis", as_cmap=True)
ax = sns.heatmap(df.isna(), cbar=False, yticklabels=False, cmap=cmap)

missing_color = cmap(1.0)

missing_patch = mpatches.Patch(color=missing_color, label='Missing Value')
plt.legend(handles=[missing_patch], loc='upper right', frameon=True)

plt.title('Missing Values Heatmap: Bus Michael 1')
plt.xlabel('Sensor Columns')
plt.tight_layout()

DataViz.save(plt)
plt.show()