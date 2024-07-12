import os
import sys
from  sklearn.metrics import r2_score
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from source.logger import logging
from source.exception import custom 
from sklearn.model_selection import train_test_split
from source.utils import Evluate_models


@dataclass
class Model_trainer_config:
    Model_trainer_file_path = os.path.join("artifact","Model_trainer_path.pkl")

class Model_trainer:

    def __init__(self):
        self.model_trainer_config  = Model_trainer_config()

    def initiate_model_training(self,Train_array,test_array):
        try:
            logging.info("Spliting the data into train input and test input ")

            X_train,Y_train,X_test,Y_test = (
                Train_array[:,:-1],
                Train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            Models = {
                "Linear Regression":LinearRegression()
            }

            Model_Report = Evluate_models(X_Train=X_train, Y_Test=Y_test, Y_Train=Y_train,X_Test= X_test, models=Models)
            logging.info("Returning thr model report")
            return Model_Report
            
        except Exception as e:
            raise custom(e,sys)
              