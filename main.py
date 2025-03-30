import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify
from constants import diseases_list, symptoms_dict

app = Flask(__name__)

# Load datasets
dataset = pd.read_csv("datasets/Training.csv")
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")

# Load model
svcLoad = pickle.load(open('models/svc.pkl','rb'))

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svcLoad.predict([input_vector])[0]]

# Helper function for returning data about the predicted diseases
def helper(dis):
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values]

    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values]

    wrkout = workout[workout['disease'] == dis] ['workout']

    return desc,pre,med,die,wrkout



# Routes
@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:

            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des, my_precautions=my_precautions, medications=medications, my_diet=rec_diet, workout=workout)

    return render_template('index.html')


@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

# @app.route('/get_symptoms')
# def get_symptoms():
#     symptoms = pd.read_csv('datasets/symptom-severity.csv')['Symptom'].tolist()
#     return jsonify(symptoms)

# @app.route('/get_symptoms_by_part')
# def get_symptoms_by_part():
#     body_part = request.args.get('body_part')
#     df = pd.read_csv('datasets/symptom-severity.csv')
#     symptoms = df[df['Body_Part'] == body_part]['Symptom'].tolist()
#     return jsonify(symptoms)

@app.route('/get_symptoms')
def get_symptoms():
    symptoms = pd.read_csv('datasets/symptom-severity.csv')['Symptom'].tolist()
    return jsonify(symptoms)

@app.route('/get_symptoms_by_part')
def get_symptoms_by_part():
    body_part = request.args.get('body_part')
    df = pd.read_csv('datasets/symptom-severity.csv')
    symptoms = df[df['Body_Part'] == body_part]['Symptom'].tolist()
    return jsonify(symptoms)


# runs the program with auto-reload on updates
if __name__ == '__main__':
    app.run(debug=True)


# set-ExecutionPolicy RemoteSigned -Scope CurrentUser 