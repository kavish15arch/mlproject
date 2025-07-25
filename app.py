from flask import Flask,render_template,request
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData , PredictPipline
application=Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            gender=request.form.get('gender'),
            race_ethincity=request.form.get('race_ethincity'),
            parental_level_of_edeucation=request.form.get('parental_level_of_edeucation'),
            lunch=request.form.get('lunch'),
            test_prepration_course=request.form.get('test_prepration_course'),
            reading_score=float(request.form.get('reading_score')),
            writing_score=float(request.form.get('writing_score')))
        

    pred_df=data.get_data_as_data_frame()
    predict_pipeline=PredictPipline()
    result=predict_pipeline.predict(pred_df)
    return render_template('home.html',result=result[0])

if __name__=='__main':
    app.run('0.0.0.0',debug=True)