import sys
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ------------------------------
# CONFIGURATION CLASS
# ------------------------------

@dataclass
class DataTransformationConfig:
    # This defines where the preprocessing object will be saved
    # After training, the entire pipeline (scaler + encoder) is saved as preprocessor.pkl
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")


# ------------------------------
# MAIN DATA TRANSFORMATION CLASS
# ------------------------------

class DataTransformation:

    def __init__(self):
        # Create config object
        # This gives access to preprocessor file path
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformer_object(self):
        """
        This function creates and returns the preprocessing pipeline.
        It handles:
        - Missing values
        - Scaling
        - Encoding categorical variables
        """

        try:
            # ------------------------------
            # Define Numerical Columns
            # ------------------------------
            numerical_columns = ["writing_score", "reading_score"]

            # ------------------------------
            # Define Categorical Columns
            # ------------------------------
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # ------------------------------
            # NUMERICAL PIPELINE
            # ------------------------------
            # Step 1: Fill missing values using median
            # Step 2: Scale values using StandardScaler
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # ------------------------------
            # CATEGORICAL PIPELINE
            # ------------------------------
            # Step 1: Fill missing values using most frequent value
            # Step 2: Convert categorical values into numbers using OneHotEncoding
            # Step 3: Scale encoded values (without centering because sparse matrix)
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore")),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # ------------------------------
            # Combine Both Pipelines
            # ------------------------------
            # ColumnTransformer applies:
            # - num_pipeline to numerical columns
            # - cat_pipeline to categorical columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            # Return full preprocessing pipeline
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self, train_path, test_path):
        """
        This function:
        1. Loads train & test CSV files
        2. Applies preprocessing
        3. Saves preprocessing object
        4. Returns transformed arrays
        """

        try:
            # ------------------------------
            # Load Train and Test Data
            # ------------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            # Get preprocessing pipeline
            preprocessing_obj = self.get_data_transformer_object()

            # Define target column
            target_column_name = "math_score"

            # ------------------------------
            # Split Features and Target (Train)
            # ------------------------------
            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            # ------------------------------
            # Split Features and Target (Test)
            # ------------------------------
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object")

            # ------------------------------
            # Fit on Train Data
            # ------------------------------
            # fit_transform:
            # - Learns parameters (mean, std, categories)
            # - Transforms train data
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)

            # ------------------------------
            # Transform Test Data
            # ------------------------------
            # Only transform (do NOT fit)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # ------------------------------
            # Combine Features + Target
            # ------------------------------
            # np.c_ horizontally stacks arrays
            train_arr = np.c_[input_feature_train_arr, target_feature_train_df.values]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_df.values]

            # ------------------------------
            # Save Preprocessing Object
            # ------------------------------
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("Preprocessing object saved successfully")

            # Return transformed arrays and preprocessor path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)