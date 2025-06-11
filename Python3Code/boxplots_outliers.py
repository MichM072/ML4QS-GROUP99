import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Specify the folder containing the data and the output folder for the boxplots
folder = 'outliers2'
output_folder = 'figures/boxplots'

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# List all CSV files in the folder
csv_files = [f for f in os.listdir(folder) if f.endswith('.csv')]

# Define the columns you want to plot
cols_to_plot = [
    #'acc_phone_X', 'acc_phone_Y', 'acc_phone_Z',
    #'lin_acc_phone_X', 'lin_acc_phone_Y', 'lin_acc_phone_Z',
    #'gyr_phone_X', 'gyr_phone_Y', 'gyr_phone_Z',
    'mag_phone_X', 'mag_phone_Y', 'mag_phone_Z',
    #'location_phone_Latitude','location_phone_Longitude','location_phone_Height',
    #'location_phone_Velocity','location_phone_Direction',
    #'location_phone_Horizontal Accuracy','location_phone_Vertical Accuracy',
    #'proximity_phone_Distance'
]

# Loop through each CSV file and plot boxplots for the specified columns
for file in csv_files:
    # Load the dataset (adjust the file path as needed)
    file_path = os.path.join(folder, file)
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Ensure that only the specified columns are used for plotting
    df_filtered = df[cols_to_plot]

    # Set the size of the plot
    plt.figure(figsize=(15, 8))

    # Generate boxplots for the specified columns
    sns.boxplot(data=df_filtered, palette="Set2")

    # Customize the plot with title and labels
    plt.title(f'Boxplots of magnometer featurs for trams', fontsize=16)
    plt.xlabel('Features', fontsize=14)
    plt.ylabel('Values', fontsize=14)

    # Save the plot in the specified output folder
    output_file = os.path.join(output_folder, f"{file}_boxplot.pdf")
    plt.tight_layout()
    plt.savefig(output_file)  # Save the figure to the specified file
    print(f"Saved boxplot for {file} in {output_file}")

    # Optionally, display the plot
    plt.close()  # Close the plot after saving it to avoid overlapping with subsequent plots
