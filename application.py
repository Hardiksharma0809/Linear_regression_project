import pandas as pd 
import numpy as np
from flask import Flask ,request,render_template
from source.exception import custom
from source.logger import logging 
from sklearn.preprocessing import StandardScaler
from source.pipeline.predction_pipeline import Customdata,Predictpipeline

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods = ["GET",'POST'])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = Customdata(
        
        age=request.form.get("age"),
        sex=request.form.get("sex"),
        bmi=request.form.get("bmi"),
        children=request.form.get("children"),
        smoker=request.form.get("smoker"),
        region=request.form.get("region")
    )
        pred_data = data.get_data_as_dataframe()
        print(pred_data)

        predict_pipeline = Predictpipeline()
        results = predict_pipeline.predict_data(pred_data)

        return render_template("home.html",results = results[0])

        #if results is not None:
            #return render_template("home.html", results=results)
        #else:
            # Handle case where results is None
            #return render_template("home.html", results=None)
    


if __name__=="__main__":
    app.run(host="0.0.0.0")