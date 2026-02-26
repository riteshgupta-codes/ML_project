import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        # No initialization required here
        # This class is used only for prediction
        pass

    def predict(self, features):
        try:
            # Define path where trained model is saved
            model_path = os.path.join("artifacts", "model.pkl")

            # Define path where preprocessor object is saved
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            print("Before Loading")

            # Load trained model (like LinearRegression, RandomForest, etc.)
            model = load_object(file_path=model_path)

            # Load preprocessing pipeline (scaler + encoder)
            preprocessor = load_object(file_path=preprocessor_path)

            print("After Loading")

            # Apply SAME preprocessing to new input data
            # Very important: must match training transformation
            data_scaled = preprocessor.transform(features)

            # Use trained model to make prediction
            preds = model.predict(data_scaled)

            # Return predicted value
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int
    ):
        # Store user input values inside object
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
        try:
            # Convert single user input into dictionary format
            # Each value is inside a list because DataFrame expects list-like structure
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            # Convert dictionary into pandas DataFrame
            # This DataFrame format must match training dataset columns
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)