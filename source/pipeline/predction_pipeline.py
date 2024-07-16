import sys
import pandas as pd 
from source.exception import custom
from source.utils import load_object
import joblib


class Predictpipeline:
    def __init__(self):
        pass

    
    def predict_data(self,features):
        try:
            model_path = "artifact/Model.pkl"
            processer_path = "artifact/proprocessor.pkl"
            model = load_object(file_path = model_path)
            processor = load_object(file_path=processer_path)
            
            data_scaled = processor.transform(features)
            pred = model.predict(data_scaled)
            return pred 
        except Exception as e:
            raise custom (e,sys)

class Customdata:

    def __init__(self,
                 age:int,
                 sex:str,
                 bmi:int,
                 children:int,
                 smoker:str,
                 region:str):
        self.age = age
        self.sex = sex
        self.bmi = bmi
        self.children = children
        self.region = region
        self.smoker = smoker


    def get_data_as_dataframe(self):
        try:
            custom_data = {
            "age": [self.age],
            "sex": [self.sex],
            "bmi": [self.bmi],
            "children": [self.children],
            "region": [self.region],
            "smoker": [self.smoker],
        }
            return pd.DataFrame(custom_data)
        except Exception as e:
            raise custom(e,sys)

