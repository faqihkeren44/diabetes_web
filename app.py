from flask import Flask, render_template, request
import pickle
import pandas as pd

with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

app = Flask(__name__, template_folder='template')

features = ['blood_glucose_level', 'HbA1c_level', 'age', 'bmi']

@app.route('/')
def main():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    blood_glucose_level = request.form['a']
    HbA1c_level = request.form['b']
    age = request.form['c']
    bmi = request.form['d']
    name = request.form['e']
    arr = [[blood_glucose_level, HbA1c_level, age, bmi]]
    arr = pd.DataFrame(arr, columns=features)
    pred = model.predict(arr)
    return render_template('after.html',
                           result=pred,
                           name=name,
                           blood_glucose_level=blood_glucose_level,
                           HbA1c_level=HbA1c_level,
                           age=age,
                           bmi=bmi)
    
if __name__ == "__main__":
    app.run(debug=True)
