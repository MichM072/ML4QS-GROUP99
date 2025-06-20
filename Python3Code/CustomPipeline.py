from FeatureCreator import FeatureCreatorUpdated, FeatureCreatorNF
from outlier_detector import OutlierDetector
from custom_imputer import CustomImputer
from DataLearningLoader import DataLearningLoader
from util.util import ignore_actual_time, read_parquet, write_parquet
import logging

class CustomPipeline:
    def __init__(self, imputer=CustomImputer(), intermediate_path=None):
        self.imputer = imputer
        print("THIS CLASS WILL NEVER BE USED, RAN OUT OF TIME SORRY!")

    def add_component(self, component_name, component):
        ...


class PreConfiguredPipeline:
    def __init__(self, intermediate_path=None, verbose=False, include_fourier=True):
        self.outlier_detector = None
        self.imputer = None
        self.data_loader = DataLearningLoader(df_path=intermediate_path, output_dir=intermediate_path, verbose=False)
        if include_fourier:
            self.feature_creator = FeatureCreatorUpdated(intermediate_path)
        else:
            self.feature_creator = FeatureCreatorNF(intermediate_path)
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def impute(self, df):
        self.imputer = CustomImputer('interpolation')
        return self.imputer.fit_transform(df)

    def detect_outliers(self, df):
        self.outlier_detector = OutlierDetector(df)
        return self.outlier_detector.fit_transform("mixture", outlier_behaviour='nan')

    @staticmethod
    def remove_bad_sensors(df, bad_sensors=None):

        if bad_sensors is None:
            bad_sensors = ['proximity']

        drop_cols = []
        for sensor in bad_sensors:
            drop_cols.extend(df.columns[df.columns.str.contains(sensor)])
        df.drop(drop_cols, axis=1, inplace=True)

    def feature_creation(self, df, name, overwrite):
        return self.feature_creator.create_features(df, name=name, overwrite=overwrite)

    def clean_data(self, df):
        return self.data_loader.prepare_split_data(df)

    @staticmethod
    def ensure_correct_types(df, original_types):
        for col, type in original_types.items():
            df[col] = df[col].astype(type)

    def fit_transform(self, intermediate_df, verbose=False, overwrite=False, pipe_name=None, srtkfold=True):
        if verbose:
            self.logger.setLevel(logging.INFO)

        df = intermediate_df.copy()
        df = ignore_actual_time(df)
        self.logger.info("Removing bad sensors...")
        self.remove_bad_sensors(df)
        splitting_method = "stratified" if srtkfold else "80/20"
        self.logger.info(f"Splitting data into train and test sets using {splitting_method} split")

        if srtkfold:
            split_sets = self.data_loader.stratgrkfold_split(df, 3)
            X_sets = []
            fold = 1
            for X_train, X_test, _, _ in split_sets:
                X_sets.extend([(X_train, f"X_train_{fold}"), (X_test, f"X_test_{fold}")])
                fold += 1
        else:
            X_train, X_test, y_train, y_test = self.data_loader.simple_split_data(df)
            X_sets = [(X_train, "X_train"), (X_test, "X_test")]

        X_sets_configured = []

        self.logger.info("Starting preprocessing pipeline...")
        for X in X_sets:
            X_name = X[1]
            X = X[0]

            if pipe_name is not None:
                X_name = f"{pipe_name}_{X_name}"


            self.logger.info(f"Preprocessing {X_name}...")
            self.logger.info(f"Checking for outliers...")
            original_types = X.dtypes
            X = self.detect_outliers(X)
            self.logger.info(f"Ensuring correct data types...")
            self.ensure_correct_types(X, original_types)
            self.logger.info(f"Imputing missing values...")
            X = self.impute(X)
            self.logger.info(f"Creating features...")
            X = self.feature_creation(X, X_name, overwrite=overwrite)
            self.logger.info(f"Dropping any direction data!")
            X.drop(columns=X.columns[X.columns.str.contains('direction', case=False)], inplace=True)
            self.logger.info(f"Cleaning data...")
            X = self.clean_data(X)
            X_sets_configured.append(X)
            self.logger.info(f"Preprocessing {X_name} complete!")


        if srtkfold:
            result_sets = []
            i,j = [0,2]
            for _, _, y_train, y_test in split_sets:
                x_train, x_test = X_sets_configured[i:j]
                result_sets.append([x_train, x_test, y_train, y_test])
                i += 2
                j += 2
            return result_sets
        else:
            X_train, X_test = X_sets_configured[0:2]
            return X_train, X_test, y_train, y_test