

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
Salary_Prediction_Model = pickle.load(open('Salary_Prediction_Model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = Salary_Prediction_Model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Salary Prediction {}'.format(output))


if __name__ == "__main__":
    app.run(debug=True)
