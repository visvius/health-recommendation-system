# Imported from .ipynb (notebook) file 

symptoms_dict = {'itching': 0,
 'skin_rash': 1,
 'nodal_skin_eruptions': 2,
 'continuous_sneezing': 3,
 'shivering': 4,
 'chills': 5,
 'joint_pain': 6,
 'stomach_pain': 7,
 'acidity': 8,
 'ulcers_on_tongue': 9,
 'muscle_wasting': 10,
 'vomiting': 11,
 'burning_micturition': 12,
 'spotting_urination': 13,
 'fatigue': 14,
 'weight_gain': 15,
 'anxiety': 16,
 'cold_hands_and_feet': 17,
 'mood_swings': 18,
 'weight_loss': 19,
 'restlessness': 20,
 'lethargy': 21,
 'patches_in_throat': 22,
 'irregular_sugar_level': 23,
 'cough': 24,
 'high_fever': 25,
 'sunken_eyes': 26,
 'breathlessness': 27,
 'sweating': 28,
 'dehydration': 29,
 'indigestion': 30,
 'headache': 31,
 'yellowish_skin': 32,
 'dark_urine': 33,
 'nausea': 34,
 'loss_of_appetite': 35,
 'pain_behind_the_eyes': 36,
 'back_pain': 37,
 'constipation': 38,
 'abdominal_pain': 39,
 'diarrhoea': 40,
 'mild_fever': 41,
 'yellow_urine': 42,
 'yellowing_of_eyes': 43,
 'acute_liver_failure': 44,
 'fluid_overload': 45,
 'swelling_of_stomach': 46,
 'swollen_lymph_nodes': 47,
 'malaise': 48,
 'blurred_and_distorted_vision': 49,
 'phlegm': 50,
 'throat_irritation': 51,
 'redness_of_eyes': 52,
 'sinus_pressure': 53,
 'runny_nose': 54,
 'congestion': 55,
 'chest_pain': 56,
 'weakness_in_limbs': 57,
 'fast_heart_rate': 58,
 'pain_during_bowel_movements': 59,
 'pain_in_anal_region': 60,
 'bloody_stool': 61,
 'irritation_in_anus': 62,
 'neck_pain': 63,
 'dizziness': 64,
 'cramps': 65,
 'bruising': 66,
 'obesity': 67,
 'swollen_legs': 68,
 'swollen_blood_vessels': 69,
 'puffy_face_and_eyes': 70,
 'enlarged_thyroid': 71,
 'brittle_nails': 72,
 'swollen_extremities': 73,
 'excessive_hunger': 74,
 'extra_marital_contacts': 75,
 'drying_and_tingling_lips': 76,
 'slurred_speech': 77,
 'knee_pain': 78,
 'hip_joint_pain': 79,
 'muscle_weakness': 80,
 'stiff_neck': 81,
 'swelling_joints': 82,
 'movement_stiffness': 83,
 'spinning_movements': 84,
 'loss_of_balance': 85,
 'unsteadiness': 86,
 'weakness_of_one_body_side': 87,
 'loss_of_smell': 88,
 'bladder_discomfort': 89,
 'foul_smell_of_urine': 90,
 'continuous_feel_of_urine': 91,
 'passage_of_gases': 92,
 'internal_itching': 93,
 'toxic_look_(typhos)': 94,
 'depression': 95,
 'irritability': 96,
 'muscle_pain': 97,
 'altered_sensorium': 98,
 'red_spots_over_body': 99,
 'belly_pain': 100,
 'abnormal_menstruation': 101,
 'dischromic_patches': 102,
 'watering_from_eyes': 103,
 'increased_appetite': 104,
 'polyuria': 105,
 'family_history': 106,
 'mucoid_sputum': 107,
 'rusty_sputum': 108,
 'lack_of_concentration': 109,
 'visual_disturbances': 110,
 'receiving_blood_transfusion': 111,
 'receiving_unsterile_injections': 112,
 'coma': 113,
 'stomach_bleeding': 114,
 'distention_of_abdomen': 115,
 'history_of_alcohol_consumption': 116,
 'fluid_overload.1': 117,
 'blood_in_sputum': 118,
 'prominent_veins_on_calf': 119,
 'palpitations': 120,
 'painful_walking': 121,
 'pus_filled_pimples': 122,
 'blackheads': 123,
 'scarring': 124,
 'skin_peeling': 125,
 'silver_like_dusting': 126,
 'small_dents_in_nails': 127,
 'inflammatory_nails': 128,
 'blister': 129,
 'red_sore_around_nose': 130,
 'yellow_crust_ooze': 131}


diseases_list = {0: 'AIDS',
 1: 'Acid Reflux',
 2: 'Acne',
 3: 'Alcoholic Hepatitis',
 4: 'Allergic Rhinitis',
 5: 'Allergy',
 6: 'Anemia',
 7: 'Anxiety Disorder',
 8: 'Arthritis',
 9: 'Bronchial Asthma',
 10: 'Bronchitis',
 11: 'COVID-Like Illness',
 12: 'Cervical Spondylosis',
 13: 'Chickenpox',
 14: 'Chronic Cholestasis',
 15: 'Common Cold',
 16: 'Dehydration',
 17: 'Dengue',
 18: 'Diabetes',
 19: 'Dimorphic Hemorrhoids (Piles)',
 20: 'Drug Reaction',
 21: 'Fever',
 22: 'Flu',
 23: 'Food Allergy',
 24: 'Food Poisoning',
 25: 'Fungal Infection',
 26: 'GERD',
 27: 'Gastroenteritis',
 28: 'Headache',
 29: 'Heart Attack',
 30: 'Heat Exhaustion',
 31: 'Hepatitis A',
 32: 'Hepatitis B',
 33: 'Hepatitis C',
 34: 'Hepatitis D',
 35: 'Hepatitis E',
 36: 'Hypertension',
 37: 'Hyperthyroidism',
 38: 'Hypoglycemia',
 39: 'Hypothyroidism',
 40: 'Impetigo',
 41: 'Insomnia',
 42: 'Jaundice',
 43: 'Lactose Intolerance',
 44: 'Malaria',
 45: 'Migraine',
 46: 'Muscle Fatigue',
 47: 'Osteoarthritis',
 48: 'Paralysis (Brain Hemorrhage)',
 49: 'Paroxysmal Positional Vertigo',
 50: 'Peptic Ulcer Disease',
 51: 'Pneumonia',
 52: 'Psoriasis',
 53: 'Sinusitis',
 54: 'Strep Throat',
 55: 'Tonsillitis',
 56: 'Tuberculosis',
 57: 'Typhoid',
 58: 'Urinary Tract Infection',
 59: 'Varicose Veins'}