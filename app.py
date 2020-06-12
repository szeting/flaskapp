import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from werkzeug.wrappers import Request, Response
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

model = load_model('model.pkl')
scaler = pickle.load(open('scaler.pkl', 'rb'))
cols=['Bedroom','Bathroom','SquareFeet','Carpark','Type','Title','Oth_Info','State']

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
        
    int_features = [x for x in request.form.values()]
    
    final_features = np.array(int_features)
    
    data_unseen=pd.DataFrame([final_features],columns=cols)
    
    def type_label (row):
        if row['Type'] == 'Apartments':
            return 0
        if row['Type'] == 'Houses' :
            return 1

    def title_label (row):
        if row['Title'] == 'Freehold':
            return 0
        if row['Title'] == 'Leasehold' :
            return 1

    def oth1_label (row):
        if row['Oth_Info'] == 'Non Bumi Lot':
            return 0
        if row['Oth_Info'] == 'Bumi Lot':
            return 0
        if row['Oth_Info'] == 'Malay Reserved' :
            return 1

    def oth2_label (row):
        if row['Oth_Info'] == 'Malay Reserved':
            return 0
        if row['Oth_Info'] == 'Bumi Lot':
            return 0
        if row['Oth_Info'] == 'Non Bumi Lot' :
            return 1

    def state_label (row):
        if row['State'] == 'Kuala Lumpur':
            return 0
        if row['State'] == 'Selangor' :
            return 1

    data_unseen['Type_ Houses']=data_unseen.apply (lambda row: type_label(row), axis=1)
    data_unseen['Title_Leasehold']=data_unseen.apply (lambda row: title_label(row), axis=1)
    data_unseen['Oth_Info_Malay Reserved']=data_unseen.apply (lambda row: oth1_label(row), axis=1)
    data_unseen['Oth_Info_Non Bumi Lot']=data_unseen.apply (lambda row: oth2_label(row), axis=1)
    data_unseen['State_Selongor']=data_unseen.apply (lambda row: state_label(row), axis=1)

    data_unseen=data_unseen.drop(['Type','Title','Oth_Info','State'],axis=1)
    
    data_unseen=data_unseen.values
    
    #sample=np.array([4,3,2380,0,1,0,0,1,0])
    
    #data_unseen=scaler.fit_transform(data_unseen.reshape(-1,9))
    data_unseen=scaler.transform(data_unseen.reshape(-1,9))
    
    prediction = model.predict(data_unseen)

    output = int(prediction[0])

    return render_template('index.html', prediction_text='The Predicted House Price is: RM {}'.format(output))


if __name__ == '__main__':
    app.run(debug=True)
