import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
# ------------------------------
# CONFIGURATION CLASS
# ------------------------------

@dataclass
class DataIngestionConfig:
    # Define file paths where data will be saved
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


# ------------------------------
# DATA INGESTION CLASS
# ------------------------------

class DataIngestion:
    def __init__(self):
        # Create configuration object
        # So we can access file paths
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # ---------------------------------
            # Step 1: Read dataset
            # ---------------------------------
            df = pd.read_csv(r'notebook\data\stud.csv')
            logging.info('Read the dataset as dataframe')

            # ---------------------------------
            # Step 2: Create artifacts folder if not exists
            # ---------------------------------
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # ---------------------------------
            # Step 3: Save raw data
            # ---------------------------------
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Train test split initiated")

            # ---------------------------------
            # Step 4: Train-Test Split
            # ---------------------------------
            train_set, test_set = train_test_split(
                df,
                test_size=0.2,
                random_state=42
            )

            # ---------------------------------
            # Step 5: Save train and test data
            # ---------------------------------
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Ingestion of the data is completed")

            # Return train and test paths
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)


# ------------------------------
# MAIN EXECUTION BLOCK
# ------------------------------

if __name__ == "__main__":

    # Step 1: Data Ingestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Step 2: Data Transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data,
        test_data
    )

    # Step 3: Model Training
    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr, test_arr))