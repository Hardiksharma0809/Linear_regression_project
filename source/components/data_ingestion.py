import pandas as pd 
import os
import sys
from source.exception import custom
from source.logger import logging 
from dataclasses import dataclass
from source.components.data_transformation import Data_transformation

from sklearn.model_selection import train_test_split
@dataclass
class dataingestionconfig:
    train_data_path:str = os.path.join("artifact","train.csv")
    test_data_path:str = os.path.join("artifact","test.csv")
    raw_data_path:str = os.path.join("artifact","raw_data.csv")

class dataingestion:
    def __init__(self):
        self.data_ingestion = dataingestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Read the data from dataframe")
        try:
            df=pd.read_csv("notebook/Data/insurance.csv")
            logging.info("Trying to read the dataset")

            os.makedirs(os.path.dirname(dataingestionconfig.train_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion.raw_data_path,index = False,header=True)
            logging.info("Doing train test split")

            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            
            test_set.to_csv(self.data_ingestion.train_data_path,index = False,header=True)
         
            test_set.to_csv(self.data_ingestion.test_data_path,index = False,header=True)
            logging.info("Complete the data ingestion part")

            return (
                self.data_ingestion.train_data_path,
                self.data_ingestion.test_data_path
            )
        except Exception as e:
            raise custom(e,sys)
            

if __name__=="__main__":   
    obj = dataingestion()
    train_data,test_data = obj.initiate_data_ingestion()
    Data_transfromation_obj = Data_transformation()   
    Data_transfromation_obj.initiate_data_transformation(train_data,test_data)