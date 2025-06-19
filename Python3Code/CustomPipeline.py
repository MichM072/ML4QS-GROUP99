from FeatureCreator import FeatureCreatorUpdated
from outlier_detector import OutlierDetector
from custom_imputer import CustomImputer
from DataLearningLoader import DataLearningLoader
from util.util import ignore_actual_time, read_parquet, write_parquet
import logging

class CustomPipeline:
    def __init__(self, imputer=CustomImputer(), intermediate_path=None):
        self.imputer = imputer

    def add_component(self, component_name, component):
        ...


class PreConfiguredPipeline:
    def __init__(self, intermediate_path=None, verbose=False):
        self.outlier_detector = None
        self.imputer = None
        self.data_loader = DataLearningLoader(df_path=intermediate_path, output_dir=intermediate_path, verbose=False)
        self.feature_creator = FeatureCreatorUpdated(intermediate_path)
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
        self.data_loader.clean_data(df)

    @staticmethod
    def ensure_correct_types(df, original_types):
        for col, type in original_types.items():
            df[col] = df[col].astype(type)

    def fit_transform(self, intermediate_df, verbose=False, overwrite=False):
        if verbose:
            self.logger.setLevel(logging.INFO)

        df = intermediate_df.copy()
        df = ignore_actual_time(df)
        self.logger.info("Removing bad sensors...")
        self.remove_bad_sensors(df)
        self.logger.info("Splitting data into train and test sets...")
        X_train, X_test, y_train, y_test = self.data_loader.simple_split_data(df)
        X_sets = [(X_train, "X_train"), (X_test, "X_test")]
        X_sets_configured = []

        self.logger.info("Starting preprocessing pipeline...")
        for X in X_sets:
            X_name = X[1]
            X = X[0]
            self.logger.info(f"Preprocessing {X_name}...")
            self.logger.info(f"Checking for outliers...")
            original_types = X.dtypes
            X = self.detect_outliers(X)
            print(type(X))
            self.logger.info(f"Ensuring correct data types...")
            self.ensure_correct_types(X, original_types)
            self.logger.info(f"Imputing missing values...")
            X = self.impute(X)
            self.logger.info(f"Creating features...")
            X = self.feature_creation(X, X_name, overwrite=overwrite)
            print(type(X))
            self.logger.info(f"Cleaning data...")
            self.clean_data(X)
            print(type(X))
            X_sets_configured.append(X)
            self.logger.info(f"Preprocessing {X_name} complete!")
            print(type(X))

        X_train, X_test = X_sets_configured

        return X_train, X_test, y_train, y_test