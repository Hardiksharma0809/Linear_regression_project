import os 
import sys
import numpy as np 
import pandas as pd 
import dill
from source.exception import custom
from sklearn.metrics import r2_score
import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
    except Exception as e:
        raise custom(e,sys)
        

def Evluate_models(X_Train,Y_Train,X_Test,Y_Test,models):
    try:
        result = {}
        
        for model_name, model_instance in models.items():
            model_instance.fit(X_Train, Y_Train)
            
            y_train_pred = model_instance.predict(X_Train)
            y_test_pred = model_instance.predict(X_Test)
            
            r2_train = r2_score(Y_Train, y_train_pred)
            r2_test = r2_score(Y_Test, y_test_pred)
            
            result[model_name] = r2_test
            
        return result

        
    except Exception as e:
        raise custom(e,sys)
        

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise custom(e,sys)
        