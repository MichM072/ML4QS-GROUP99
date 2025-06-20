from pathlib import Path
import os

STUDENT = 'mmr497'
DATA_PATH = Path('/local/data/mmr497')
OUTLIERS_PATH = Path('./outliers2/')
INTERMEDIATE_PATH = Path(f'{DATA_PATH}/intermediate_datafiles/')
os.chdir(f'/home/{STUDENT}/')
from util.util import ignore_actual_time, read_parquet, write_parquet
from CustomPipeline import PreConfiguredPipeline

"""
Just like the other file, this one is just for me to use on the ssh server.
"""

def write_splitted_data(data):

    if len(data[0]) == 1:
        print("Expecting format: (data, name) ... continuing.")

    for data, name in data:
        try:
            write_parquet(data, INTERMEDIATE_PATH / f'{name}_postpipe.parquet')
        except:
            print("Can't write parquet file for ", name, " ... continuing.")

def main():
    intermediate_df = read_parquet(INTERMEDIATE_PATH / 'ML4QS_combined_results_final.parquet')

    pipe = PreConfiguredPipeline(intermediate_path=INTERMEDIATE_PATH)
    split_sets = pipe.fit_transform(intermediate_df, verbose=True, overwrite=True, srtkfold=True)

    split = 1
    for X_train, X_test, y_train, y_test in split_sets:
        split_data = [(X_train, f"X_train_KF{split}"), (X_test, f"X_test_KF{split}"), (y_train, f"y_train_KF{split}"), (y_test, f"y_test_KF{split}")]
        write_splitted_data(split_data)
        split +=1

    print("Done with writing splitted data.")

if __name__ == '__main__':
    main()