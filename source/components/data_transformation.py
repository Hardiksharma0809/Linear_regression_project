import sys
import pandas as pd 
import numpy as np
import dataclasses as dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from source.exception import custom
from source.logger import logging
import os
from source.utils import save_object



class DataTransformationConfig:
    def __init__(self):

        self.preprocessor_obj_file_path = os.path.join('artifact',"proprocessor.pkl")


class Data_transformation:
    def __init__(self):

        self.datatransformation_config = DataTransformationConfig()



    def Data_get_transformed(self):
        try:

            Numerical_column = [
                "age",
                "bmi",
                "children"
                ]
            
            Categorical_feature = [
                "sex",
                "smoker",
                "region"
            ]
            num_pipeline = Pipeline(
                steps=[
                    ("scaler",StandardScaler(with_mean=False))
                    
                    ]
                
            )
            cat_pipeline = Pipeline(
                steps=[
                    ("onehotencoder",OneHotEncoder(handle_unknown="ignore")),
                    ("scaler",StandardScaler(with_mean=False))
                ]



            )
            logging.info("numerical columns standard scaling is completed")

            logging.info("Categorical feature onehot encoding is done ")

            transformer = [
                ("num_pipeline",num_pipeline,Numerical_column),
                ("cat_pipeline",cat_pipeline,Categorical_feature)
            ]
            processer = ColumnTransformer(transformer)


            return processer
        



        except Exception as e :
            raise custom(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("importing the data using the pandas")

            logging.info("obtaining the preprocessing object")

            processer_obj = self.Data_get_transformed()

            target_column = "expenses"

            input_feature_train_df = train_df.drop(columns=[target_column],axis = 1)
            output_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns = [target_column],axis = 1)
            output_feature_test_df = test_df[target_column]

            logging.info("Applying the processing obj on the train and test data")

           # Fit and transform training data
            input_feature_train_df_arr = processer_obj.fit_transform(input_feature_train_df)

            # Transform test data using parameters learned from training data
            input_feature_test_df_arr = processer_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_df_arr,np.array(output_feature_train_df)

            ]
            test_arr = np.c_[
                input_feature_test_df_arr,np.array(output_feature_test_df)
            ]
            logging.info("Saving the processer object")

            save_object(
                file_path = self.datatransformation_config.preprocessor_obj_file_path,
                obj = processer_obj
            )
            return (
                train_arr,
                test_arr,
                self.datatransformation_config.preprocessor_obj_file_path
                )

            
        except Exception as e:
            raise custom(e,sys)
        
            