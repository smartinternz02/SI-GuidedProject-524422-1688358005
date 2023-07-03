from flask import Flask, render_template, request
import pandas as pd

import pickle

app = Flask(__name__)

# Load the model from the pickle file
with open('diabetes_lj_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Flask route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Flask route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from the form
    name = request.form['Name']
    input_data = {
        'HighBp': 1 if request.form['HighBp'] == 'yes' else 0,
        'HighChol': 1 if request.form['HighChol'] == 'yes' else 0,
        'CholCheck': 1 if request.form['CholCheck'] == 'yes' else 0,
        'BMI': float(request.form['BMI']),
        'Smoker': 1 if request.form['Smoker'] == 'yes' else 0,
        'Stroke': 1 if request.form['Stroke'] == 'yes' else 0,
        'HeartDiseaseorAttack': 1 if request.form['HeartDiseaseorAttack'] == 'yes' else 0,
        'PhysActivity': 1 if request.form['PhysActivity'] == 'yes' else 0,
        'Fruits': 1 if request.form['Fruits'] == 'yes' else 0,
        'Veggies': 1 if request.form['Veggies'] == 'yes' else 0,
        'HvyAlcoholConsump': 1 if request.form['HvyAlcoholConsump'] == 'yes' else 0,
        'AnyHealthcare': 1 if request.form['AnyHealthcare'] == 'yes' else 0,
        'NoDocbcCost': 1 if request.form['NoDocbcCost'] == 'yes' else 0,
        'GenHlth': int(request.form['GenHlth']),
        'MentHlth': int(request.form['MentHlth']),
        'PhysHlth': int(request.form['PhysHlth']),
        'DiffWalk': 1 if request.form['DiffWalk'] == 'yes' else 0,
        'Sex': int(request.form['Sex']),
        'Age': int(request.form['Age']),
        'Education': int(request.form['Education']),
        'Income': int(request.form['Income'])
    }
    print("Incoming form response:")
    for key, value in input_data.items():
        print(key + ": " + str(value))

    # Create a dataframe from the user input
    input_df = pd.DataFrame(input_data, index=[0])

    # Make a prediction using the loaded model
    prediction = model.predict(input_df)[0]

    # Return the prediction as a response
    return render_template('result.html', prediction=prediction, name = name)


if __name__ == '__main__':
    app.run(debug=True)
