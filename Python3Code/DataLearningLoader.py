import numpy as np
import pandas as pd
import logging

class DataLearningLoader:
    def __init__(self, df_path, output_dir, verbose=False):
        self.dataset_path = df_path
        self.output_dir = output_dir
        self.leaking_features = []
        self.feature_cols = []
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    def clean_data(self, df):

        self.logger.info(f"Original DataFrame shape: {df.shape}")
        self.logger.info("Available columns before feature removal:")
        self.logger.info(df.columns.tolist())

        # Define leaking features to remove
        self.leaking_features = [
            'location_phone_Latitude',
            'location_phone_Longitude',
            'location_phone_Direction',
            'location_phone_Velocity',
            'proximity_phone_Distance',
            'location_phone_Horizontal Accuracy',
            'location_phone_Height',
            'location_phone_Vertical Accuracy'
        ]

        # Remove leaking features
        self.logger.info(f"\nRemoving {len(self.leaking_features)} leaking features...")
        features_found = [f for f in self.leaking_features if f in df.columns]
        features_not_found = [f for f in self.leaking_features if f not in df.columns]

        if features_found:
            self.logger.info(f"Removing features: {features_found}")
            df = df.drop(columns=features_found)
        else:
            self.logger.info("No leaking features found in the dataset")

        if features_not_found:
            self.logger.info(f"Features not found (already removed?): {features_not_found}")

        self.logger.info(f"DataFrame shape after removing leaking features: {df.shape}")

        # -------------------------------
        # Inspect remaining columns
        # -------------------------------
        self.logger.info("\nRemaining columns:")
        self.logger.info(df.columns.tolist())
        self.logger.info(f"\nDataFrame shape: {df.shape}")

    def create_labels(self, df):

        # -------------------------------
        # FIND ALL LABEL COLUMNS AUTOMATICALLY
        # -------------------------------
        # Find all label columns (assuming they start with 'label')
        label_columns = [col for col in df.columns if col.startswith('label')]
        self.logger.info(f"\nFound label columns: {label_columns}")

        # Extract transportation modes from label column names
        transportation_modes = [col.replace('label', '') for col in label_columns]
        self.logger.info(f"Transportation modes: {transportation_modes}")

        # Keep this func here for the apply.
        def create_target_label(row):
            """Create target label from all available label columns"""
            for i, col in enumerate(label_columns):
                if row.get(col, 0) == 1:
                    return transportation_modes[i]
            return 'unknown'

        df['transport_mode'] = df.apply(create_target_label, axis=1)

        # Show all available transport modes and their counts
        self.logger.info(f"\nAll transport modes found in data:")
        mode_counts = df['transport_mode'].value_counts()
        self.logger.info(mode_counts)

        # Filter out 'unknown' samples (if any)
        df = df[df['transport_mode'] != 'unknown']

        self.logger.info(f"\nAfter filtering out unknown samples: {df.shape[0]} samples")
        self.logger.info(f"Final class distribution:\n{df['transport_mode'].value_counts()}")

        # Check session distribution by transport mode
        self.logger.info(f"\nSession distribution analysis:")
        session_counts = df.groupby(['id', 'transport_mode']).size().unstack(fill_value=0)

        sessions_per_mode = {}
        for mode in df['transport_mode'].unique():
            sessions_with_mode = (session_counts[mode] > 0).sum() if mode in session_counts.columns else 0
            sessions_per_mode[mode] = sessions_with_mode
            self.logger.info(f"{mode} sessions: {sessions_with_mode}")

        self.logger.info(f"Total unique sessions: {df['id'].nunique()}")

        # Warn about modes with very few sessions
        self.logger.info(f"\n Session distribution warnings:")
        for mode, count in sessions_per_mode.items():
            if count < 3:
                self.logger.info(f"WARNING: {mode} has only {count} session(s) - may cause issues in train/test split")

    def clean_features(self, df):

        # -------------------------------
        # Feature selection
        # -------------------------------
        # Find timestamp column (if exists)
        time_like_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['time', 'date', 'stamp'])]
        timestamp_col = None
        possible_timestamp_cols = ['timestamp', 'time', 'datetime', 'date_time', 'session_time']

        for col in possible_timestamp_cols:
            if col in df.columns:
                timestamp_col = col
                break

        if timestamp_col is None and time_like_columns:
            timestamp_col = time_like_columns[0]

        # Ensure timestamp column is excluded from features to prevent data leakage
        exclude_cols = ['id', 'transport_mode']
        if timestamp_col:
            exclude_cols.append(timestamp_col)

        self.feature_cols = [
            col for col in df.columns
            if not col.startswith('label')
               and col not in exclude_cols
        ]

        self.logger.info(f"\nUsing {len(self.feature_cols)} features (after removing leaking features)")
        if timestamp_col:
            self.logger.info(f"Timestamp column excluded: '{timestamp_col}'")

        # Verify no leaking features remain
        remaining_leaking = [f for f in self.leaking_features if f in self.feature_cols]
        if remaining_leaking:
            self.logger.info(f"Some leaking features still present: {remaining_leaking}")
        else:
            self.logger.info("No leaking features in final feature set")

        # -------------------------------
        # Handle non-numeric columns (especially timedelta)
        # -------------------------------
        self.logger.info(f"\nChecking data types in features...")
        self.logger.info(f"Feature dtypes (first 10):")
        for col in self.feature_cols[:10]:
            if col in df.columns:
                dtype = df[col].dtype
                self.logger.info(f"  {col}: {dtype}")

                if dtype == 'object':
                    sample_values = df[col].dropna().head(3).tolist()
                    self.logger.info(f"    Sample values: {sample_values}")

        # Convert timedelta columns to numeric (seconds)
        timedelta_cols = []
        problematic_cols = []

        for col in self.feature_cols:
            if col in df.columns:
                if df[col].dtype == 'object':
                    sample_val = str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else ""
                    if 'days' in sample_val and ':' in sample_val:
                        print(f"Converting timedelta column '{col}' to seconds...")
                        try:
                            df[col] = pd.to_timedelta(df[col]).dt.total_seconds()
                            timedelta_cols.append(col)
                        except Exception as e:
                            print(f"Failed to convert {col}: {e}")
                            problematic_cols.append(col)

        if timedelta_cols:
            self.logger.info(f"Converted {len(timedelta_cols)} timedelta columns to seconds: {timedelta_cols}")

        # Remove non-numeric columns
        non_numeric_cols = []
        for col in self.feature_cols:
            if col in df.columns:
                if df[col].dtype == 'object' or not pd.api.types.is_numeric_dtype(df[col]):
                    non_numeric_cols.append(col)

        non_numeric_cols.extend(problematic_cols)
        non_numeric_cols = list(set(non_numeric_cols))

        # Ensure no label columns remain in the final data
        label_cols = [col for col in df.columns if col.startswith('label')]
        if label_cols:
            df.drop(columns=label_cols, axis=1, inplace=True)

        if non_numeric_cols:
            self.logger.info(f"Found non-numeric columns that will be excluded: {non_numeric_cols}")
            feature_cols = [col for col in self.feature_cols if col not in non_numeric_cols]
            self.logger.info(f"Updated feature count: {len(feature_cols)}")

    def simple_split_data(self, df):
        # -------------------------------
        # IMPROVED SESSION-LEVEL SPLIT FOR ALL CLASSES
        # -------------------------------

        # First create label col
        self.create_labels(df)

        self.logger.info("\n" + "=" * 50)
        self.logger.info("IMPLEMENTING SESSION-LEVEL SPLIT FOR ALL TRANSPORT MODES")
        self.logger.info("=" * 50)

        # Get unique session IDs for each transport mode
        session_split_info = {}
        all_train_sessions = []
        all_test_sessions = []

        np.random.seed(42)


        for mode in df['transport_mode'].unique():
            mode_sessions = df[df['transport_mode'] == mode]['id'].unique()
            mode_sessions_shuffled = np.random.permutation(mode_sessions)

            n_sessions = len(mode_sessions_shuffled)

            # Handle cases with very few sessions
            if n_sessions == 1:
                # If only 1 session, put it in training
                train_sessions = mode_sessions_shuffled
                test_sessions = []
                print(f"{mode}: Only 1 session - putting in training set")
            elif n_sessions == 2:
                # If only 2 sessions, put 1 in each
                train_sessions = mode_sessions_shuffled[:1]
                test_sessions = mode_sessions_shuffled[1:]
                print(f"{mode}: Only 2 sessions - 1 train, 1 test")
            else:
                # Normal 80/20 split
                split_idx = max(1, int(0.8 * n_sessions))  # Ensure at least 1 in training
                train_sessions = mode_sessions_shuffled[:split_idx]
                test_sessions = mode_sessions_shuffled[split_idx:]

            session_split_info[mode] = {
                'total': n_sessions,
                'train': len(train_sessions),
                'test': len(test_sessions)
            }

            all_train_sessions.extend(train_sessions)
            all_test_sessions.extend(test_sessions)

        self.logger.info(f"\nSession split breakdown:")
        for mode, info in session_split_info.items():
            self.logger.info(f"  {mode}: {info['total']} total -> {info['train']} train, {info['test']} test")

        self.logger.info(f"\nTotal sessions:")
        self.logger.info(f"  Train: {len(all_train_sessions)}")
        self.logger.info(f"  Test: {len(all_test_sessions)}")

        # Create train and test datasets
        df_train = df[df['id'].isin(all_train_sessions)].copy()
        df_test = df[df['id'].isin(all_test_sessions)].copy()

        # Verify no session overlap
        train_session_set = set(df_train['id'].unique())
        test_session_set = set(df_test['id'].unique())
        session_overlap = train_session_set.intersection(test_session_set)

        self.logger.info(f"\n🔍 VERIFICATION:")
        self.logger.info(f"Train sessions: {len(train_session_set)}")
        self.logger.info(f"Test sessions: {len(test_session_set)}")
        self.logger.info(f"Session overlap: {len(session_overlap)}")

        if len(session_overlap) == 0:
            self.logger.info("No session overlap between train and test!")
        else:
            self.logger.info("Session overlap detected!")


        # Prepare features and targets
        X_train = df_train
        X_test = df_test
        y_train = df_train['transport_mode']
        y_test = df_test['transport_mode']

        self.logger.info(f"\nFinal dataset sizes:")
        self.logger.info(f"Train samples: {X_train.shape[0]}")
        self.logger.info(f"Test samples: {X_test.shape[0]}")
        self.logger.info(f"Train class distribution:\n{y_train.value_counts()}")
        self.logger.info(f"Test class distribution:\n{y_test.value_counts()}")

        # Check for missing values
        self.logger.info(f"\nData quality check:")
        self.logger.info(f"Missing values in train features: {X_train.isnull().sum().sum()}")
        numeric_cols = X_train.select_dtypes(include=['number'], exclude=['timedelta64[ns]']).columns

        if len(numeric_cols) > 0:
            # Convert to float just to prevent error, stupid fix!
            numeric_data = X_train[numeric_cols]
            inf_values = np.isinf(numeric_data).sum().sum()
            self.logger.info(f"Infinite values in train features: {inf_values}")

        # Final cleanup of any remaining object columns
        object_cols = X_train.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            self.logger.info(f"Removing remaining object columns: {list(object_cols)}")
            X_train = X_train.select_dtypes(exclude=['object'])
            X_test = X_test.select_dtypes(exclude=['object'])

        self.logger.info(f"Final feature matrix shape: Train {X_train.shape}, Test {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def split_data(self, df):

        # -------------------------------
        # IMPROVED SESSION-LEVEL SPLIT FOR ALL CLASSES
        # -------------------------------
        self.logger.info("\n" + "=" * 50)
        self.logger.info("IMPLEMENTING SESSION-LEVEL SPLIT FOR ALL TRANSPORT MODES")
        self.logger.info("=" * 50)

        # Get unique session IDs for each transport mode
        session_split_info = {}
        all_train_sessions = []
        all_test_sessions = []

        np.random.seed(42)

        for mode in df['transport_mode'].unique():
            mode_sessions = df[df['transport_mode'] == mode]['id'].unique()
            mode_sessions_shuffled = np.random.permutation(mode_sessions)

            n_sessions = len(mode_sessions_shuffled)

            # Handle cases with very few sessions
            if n_sessions == 1:
                # If only 1 session, put it in training
                train_sessions = mode_sessions_shuffled
                test_sessions = []
                print(f"{mode}: Only 1 session - putting in training set")
            elif n_sessions == 2:
                # If only 2 sessions, put 1 in each
                train_sessions = mode_sessions_shuffled[:1]
                test_sessions = mode_sessions_shuffled[1:]
                print(f"{mode}: Only 2 sessions - 1 train, 1 test")
            else:
                # Normal 80/20 split
                split_idx = max(1, int(0.8 * n_sessions))  # Ensure at least 1 in training
                train_sessions = mode_sessions_shuffled[:split_idx]
                test_sessions = mode_sessions_shuffled[split_idx:]

            session_split_info[mode] = {
                'total': n_sessions,
                'train': len(train_sessions),
                'test': len(test_sessions)
            }

            all_train_sessions.extend(train_sessions)
            all_test_sessions.extend(test_sessions)

        self.logger.info(f"\nSession split breakdown:")
        for mode, info in session_split_info.items():
            self.logger.info(f"  {mode}: {info['total']} total -> {info['train']} train, {info['test']} test")

        self.logger.info(f"\nTotal sessions:")
        self.logger.info(f"  Train: {len(all_train_sessions)}")
        self.logger.info(f"  Test: {len(all_test_sessions)}")

        # Create train and test datasets
        df_train = df[df['id'].isin(all_train_sessions)].copy()
        df_test = df[df['id'].isin(all_test_sessions)].copy()

        # Verify no session overlap
        train_session_set = set(df_train['id'].unique())
        test_session_set = set(df_test['id'].unique())
        session_overlap = train_session_set.intersection(test_session_set)

        self.logger.info(f"\n🔍 VERIFICATION:")
        self.logger.info(f"Train sessions: {len(train_session_set)}")
        self.logger.info(f"Test sessions: {len(test_session_set)}")
        self.logger.info(f"Session overlap: {len(session_overlap)}")

        if len(session_overlap) == 0:
            self.logger.info("No session overlap between train and test!")
        else:
            self.logger.info("Session overlap detected!")

        if not self.feature_cols:
            self.logger.info("No feature cols available, run the feature cleanup first. Exiting.")
            return

        # Prepare features and targets
        X_train = df_train[self.feature_cols]
        X_test = df_test[self.feature_cols]
        y_train = df_train['transport_mode']
        y_test = df_test['transport_mode']

        self.logger.info(f"\nFinal dataset sizes:")
        self.logger.info(f"Train samples: {X_train.shape[0]}")
        self.logger.info(f"Test samples: {X_test.shape[0]}")
        self.logger.info(f"Train class distribution:\n{y_train.value_counts()}")
        self.logger.info(f"Test class distribution:\n{y_test.value_counts()}")

        # Check for missing values
        self.logger.info(f"\nData quality check:")
        self.logger.info(f"Missing values in train features: {X_train.isnull().sum().sum()}")
        numeric_cols = X_train.select_dtypes(include=['number'], exclude=['timedelta64[ns]']).columns

        if len(numeric_cols) > 0:
            # Convert to float just to prevent error, stupid fix!
            numeric_data = X_train[numeric_cols]
            inf_values = np.isinf(numeric_data).sum().sum()
            self.logger.info(f"Infinite values in train features: {inf_values}")

        # Final cleanup of any remaining object columns
        object_cols = X_train.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            self.logger.info(f"Removing remaining object columns: {list(object_cols)}")
            X_train = X_train.select_dtypes(exclude=['object'])
            X_test = X_test.select_dtypes(exclude=['object'])

        self.logger.info(f"Final feature matrix shape: Train {X_train.shape}, Test {X_test.shape}")

        return X_train, X_test, y_train, y_test

    def prepare_split_data(self, dataset):
        df = dataset.copy()
        print(f"Cleaning data...")
        self.clean_data(df)
        print(f"Cleaning features...")
        self.clean_features(df)
        return df


    @DeprecationWarning
    def prepare_data(self, dataset):
        self.logger.warning("Deprecated method. Use 'prepare_split_data' instead.")
        df = dataset.copy()
        print(f"Cleaning data...")
        self.clean_data(df)
        print(f"Creating labels...")
        self.create_labels(df)
        print(f"Cleaning features...")
        self.clean_features(df)
        print(f"Splitting data...")
        X_train, X_test, y_train, y_test = self.split_data(df)

        return X_train, X_test, y_train, y_test