import os
import sys
from  sklearn.metrics import r2_score
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from source.logger import logging
from source.exception import custom 
from sklearn.model_selection import train_test_split
from source.utils import Evluate_models
from source.utils import save_object
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


@dataclass
class Model_trainer_config:
        Model_trainer_file_path = os.path.join("artifact","Model.pkl")

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
                "DecisionTree":DecisionTreeRegressor()
            }

            Model_Report:dict = Evluate_models(X_Train=X_train, Y_Test=Y_test, Y_Train=Y_train,X_Test= X_test, models=Models)
            logging.info("Returning thr model report")
            best_model_score = max(sorted(Model_Report.values()))

            best_model_name = list(Model_Report.keys())[
                list(Model_Report.values()).index(best_model_score)
            ]
            best_model = Models[best_model_name]
            

            save_object(
                file_path=self.model_trainer_config.Model_trainer_file_path,
                obj=best_model


            )
            print(type(best_model))
            predict = best_model.predict(X_test)
            r2_square = r2_score(Y_test, predict)
            return r2_square
            
        except Exception as e:
            raise custom(e,sys)
              