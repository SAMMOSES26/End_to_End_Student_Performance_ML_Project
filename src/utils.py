import os
import dill

import sys

import pandas as pd
import numpy as np  
from sklearn.metrics import r2_score

from src.logger import logging
from src.exception import CustomException




def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):

    try:
        report = {}

        for i in range(len(models)):

            model = list(models.values())[i]

            model.fit(X_train, y_train)  # Train the model

            y_train_pred = model.predict(X_train)  # Predict on the training set
            y_test_pred = model.predict(X_test)  # Predict on the test set

            test_model_score = r2_score(y_test, y_test_pred)  # Calculate R2 score

            report[list(models.keys())[i]] = test_model_score  # Store the score in the report

        return report

    except Exception as e:
        logging.error("Error occurred during model evaluation.")
        raise CustomException(e, sys)