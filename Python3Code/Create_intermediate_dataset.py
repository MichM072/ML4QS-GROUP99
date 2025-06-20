from pathlib import Path
import os

STUDENT = 'mmr497'

OUTLIERS_PATH = Path('./outliers2/')
INTERMEDIATE_PATH = Path('./intermediate_datafiles/')
os.chdir(f'/home/{STUDENT}/')
EXPERIMENT_DIR = 'ML4QS-Vehicle-Final'

from util.util import write_parquet
from DataLoader import PhyboxDatasetLoader

# Purely here so I can run this in the background without worrying about the other code.

dataset_loader = PhyboxDatasetLoader(STUDENT, exp_dir=EXPERIMENT_DIR, overwrite_loader=False)
datasets = dataset_loader.create_all_datasets(overwrite=False)
write_parquet(datasets, INTERMEDIATE_PATH / 'ML4QS_combined_results_final.parquet')