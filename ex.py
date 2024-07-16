import pandas as pd 
import numpy as np
from source.pipeline.predction_pipeline import Predictpipeline,Customdata
from source.utils import load_object
from sklearn.preprocessing import StandardScaler

disct = {"age":19,"sex":"female","bmi":27.9,"children":"0","region":"southwest","smoker":"yes"}

new_data = pd.DataFrame([disct])
model_path = "artifact/Model.pkl"
processer_path = "artifact/proprocessor.pkl"
model = load_object(file_path = model_path)
processor = load_object(file_path=processer_path)
data_scaled = processor.transform(new_data)
pred = model.predict(data_scaled)
print(pred[0])