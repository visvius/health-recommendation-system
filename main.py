import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, jsonify
from constants import diseases_list, symptoms_dict
import os
import ast


app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dataset paths
dataset = pd.read_csv(os.path.join(BASE_DIR, "datasets", "Training.csv"))
sym_des = pd.read_csv(os.path.join(BASE_DIR, "datasets", "symtoms_df.csv"))
precautions = pd.read_csv(os.path.join(BASE_DIR, "datasets", "precautions_df.csv"))
workout = pd.read_csv(os.path.join(BASE_DIR, "datasets", "workout_df.csv"))
description = pd.read_csv(os.path.join(BASE_DIR, "datasets", "description.csv"))
medications = pd.read_csv(os.path.join(BASE_DIR, "datasets", "medications.csv"))
diets = pd.read_csv(os.path.join(BASE_DIR, "datasets", "diets.csv"))
specialist = pd.read_csv(os.path.join(BASE_DIR, "datasets", "disease_specialist.csv"))

# Model path
svc_path = os.path.join(BASE_DIR, "models", "svc.pkl")
svcLoad = pickle.load(open(svc_path, 'rb'))


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

    doc = specialist[specialist['Disease'] == dis]['Specialist'].iloc[0]

    return desc, pre, med, die, wrkout, doc



# Routes
@app.route("/")
def index():
    return render_template("index.html")

# Define a route for the home page
@app.route('/predict', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        # print(symptoms)
        if symptoms =="Symptoms":
            message = "Please either write symptoms or you have written misspelled symptoms"
            return render_template('index.html', message=message)
        else:

            # Split the user's input into a list of symptoms (assuming they are comma-separated)
            user_symptoms = [s.strip() for s in symptoms.split(',')]
            # Remove any extra characters, if any
            user_symptoms = [symptom.strip("[]' ") for symptom in user_symptoms]
            predicted_disease = get_predicted_value(user_symptoms)
            dis_des, precautions, medications, rec_diet, workout, doc = helper(predicted_disease)

            my_precautions = []
            for i in precautions[0]:
                my_precautions.append(i)

            return render_template('index.html', predicted_disease=predicted_disease, dis_des=dis_des, my_precautions=my_precautions, medications=medications, my_diet=rec_diet, workout=workout, doc=doc)

    return render_template('index.html')

@app.route('/find-doctor')
def find_doctor():
    api_key = os.getenv('FOURSQUARE_API_KEY')  # Load from environment
    # print("Api:" + api_key)
    doc = request.args.get('doc', 'Unknown Specialist')  
    city = request.args.get('city', '')
    return render_template('find-doctor.html', doc=doc, city=city, apikey=api_key)

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/get_symptoms')
def get_symptoms():
    symptoms_df = pd.read_csv(os.path.join(BASE_DIR, 'datasets', 'Symptom-severity.csv'))
    symptoms = symptoms_df['Symptom'].tolist()
    return jsonify(symptoms)

@app.route('/get_symptoms_by_part')
def get_symptoms_by_part():
    body_part = request.args.get('body_part')
    df = pd.read_csv(os.path.join(BASE_DIR, 'datasets', 'Symptom-severity.csv'))
    symptoms = df[df['Body_Part'] == body_part]['Symptom'].tolist()
    return jsonify(symptoms)






@app.route('/predict-json', methods=['POST'])
def predict_json():
    data = request.get_json()
    if not data or 'symptoms' not in data:
        return jsonify({'error': 'No symptoms provided'}), 400

    symptoms = data['symptoms']
    if not isinstance(symptoms, list):
        return jsonify({'error': 'Symptoms should be a list of strings'}), 400

    # 1. Clean incoming symptoms
    user_symptoms = [s.strip("[]' ").strip() for s in symptoms]

    # 2. Predict & get raw outputs
    predicted_disease = get_predicted_value(user_symptoms)
    desc, raw_prec, raw_meds, raw_diet, raw_workout, doc = helper(predicted_disease)

    # 3. Flatten `raw_prec` (which may be [ndarray([...])] or similar)
    my_precautions = []
    if isinstance(raw_prec, (list, tuple)):
        for entry in raw_prec:
            if isinstance(entry, np.ndarray):
                my_precautions.extend(entry.tolist())
            elif isinstance(entry, (list, tuple)):
                my_precautions.extend(entry)
            else:
                my_precautions.append(entry)
    elif isinstance(raw_prec, np.ndarray):
        my_precautions = raw_prec.tolist()
    else:
        my_precautions = [raw_prec]

    # 4. Parse any "['a','b',…]" strings in medications → real list
    clean_meds = []
    for entry in raw_meds:
        if isinstance(entry, str):
            try:
                parsed = ast.literal_eval(entry)
                if isinstance(parsed, list):
                    clean_meds.extend(parsed)
                else:
                    clean_meds.append(str(parsed))
            except Exception:
                clean_meds.append(entry)
        else:
            clean_meds.append(entry)

    # 5. Same for diet
    clean_diet = []
    for entry in raw_diet:
        if isinstance(entry, str):
            try:
                parsed = ast.literal_eval(entry)
                if isinstance(parsed, list):
                    clean_diet.extend(parsed)
                else:
                    clean_diet.append(str(parsed))
            except Exception:
                clean_diet.append(entry)
        else:
            clean_diet.append(entry)

    # 6. Ensure workout is a plain list
    wkt = raw_workout.tolist() if hasattr(raw_workout, 'tolist') else raw_workout

    # 7. Return proper JSON
    return jsonify({
        'disease': predicted_disease,
        'description':       desc,
        'precautions':       my_precautions,
        'medications':       clean_meds,
        'diets':              clean_diet,
        'workouts':           wkt,
        'recommendedDoctor':        doc
    })


# runs the program with auto-reload on updates
if __name__ == '__main__':
    app.run(debug=True)


# set-ExecutionPolicy RemoteSigned -Scope CurrentUser 