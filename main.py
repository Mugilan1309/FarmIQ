from fastapi import FastAPI
import joblib
import numpy as np
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore
import uuid

# Initialize Firebase
cred = credentials.Certificate(r"C:\Users\mugil\Downloads\smartfarming-6cf28-firebase-adminsdk-fbsvc-a32b0a633f.json")  # Replace with your JSON key
if not firebase_admin._apps:
    firebase_admin.initialize_app(cred)

db = firestore.client()

app = FastAPI()

# Load trained models
crop_model = joblib.load("crop_recommendation_model.pkl")
fertilizer_model = joblib.load("fertilizer_recommendation_model.pkl")

@app.get("/")
def read_root():
    return {"message": "AI-Based Smart Farming Assistant API is running!"}

@app.post("/predict_crop/")
def predict_crop(data: dict):
    try:
        # Extract input values
        N = data.get("N")
        P = data.get("P")
        K = data.get("K")
        temperature = data.get("temperature")
        humidity = data.get("humidity")
        ph = data.get("ph")
        rainfall = data.get("rainfall")

        if None in [N, P, K, temperature, humidity, ph, rainfall]:
            return {"error": "Missing input values"}

        # Crop prediction (returns 3 best crops)
        crop_input = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)
        crop_predictions = crop_model.predict_proba(crop_input)[0]  # Get probability scores
        top_3_crops = [crop_model.classes_[i] for i in np.argsort(crop_predictions)[-3:][::-1]]  # Top 3 crops

        # Generate session ID
        session_id = str(uuid.uuid4())

        # Store input & crop suggestions in Firebase
        db.collection("users").document(session_id).set({
            "N": N, "P": P, "K": K, "temperature": temperature, "humidity": humidity, 
            "ph": ph, "rainfall": rainfall, "suggested_crops": top_3_crops, "selected_crop": None, 
            "moisture": None, "soil_type": None, "fertilizer_recommendation": None
        })
        print(f"Stored session {session_id} in Firebase")

        return {"session_id": session_id, "recommended_crops": top_3_crops}

    except Exception as e:
        return {"error": str(e)}

@app.post("/select_crop/")
def select_crop(data: dict):
    try:
        session_id = data.get("session_id")
        selected_crop = data.get("selected_crop")

        if not session_id or not selected_crop:
            return {"error": "Missing session_id or selected_crop"}

        # Check if the document exists before updating
        doc_ref = db.collection("users").document(session_id)
        doc = doc_ref.get()
        if not doc.exists:
            return {"error": "Session ID not found"}

        # Update Firebase with the selected crop
        doc_ref.update({"selected_crop": selected_crop})
        print(f"Updated session {session_id} with selected crop '{selected_crop}'")

        return {"message": f"Selected crop '{selected_crop}' saved successfully."}

    except Exception as e:
        return {"error": str(e)}

@app.post("/predict_fertilizer/")
def predict_fertilizer(data: dict):
    try:
        session_id = data.get("session_id")
        moisture = data.get("moisture")
        soil_type = data.get("soil_type")

        if not session_id or not moisture or not soil_type:
            return {"error": "Missing session_id, moisture, or soil_type"}

        # Get stored user data
        doc_ref = db.collection("users").document(session_id)
        doc = doc_ref.get()
        if not doc.exists:
            return {"error": "Session ID not found"}

        user_data = doc.to_dict()
        selected_crop = user_data.get("selected_crop")

        if not selected_crop:
            return {"error": "Selected crop not found. Please choose a crop first."}

        N, P, K, temperature, humidity = user_data["N"], user_data["P"], user_data["K"], user_data["temperature"], user_data["humidity"]

        # Fertilizer prediction
        fert_input = pd.DataFrame([[temperature, humidity, moisture, soil_type, selected_crop, N, P, K]],
                                  columns=["Temperature", "Humidity", "Moisture", "Soil_Type", "Crop_Type", "Nitrogen", "Potassium", "Phosphorous"])

        fert_input = pd.get_dummies(fert_input, columns=['Soil_Type', 'Crop_Type'])

        # Align columns with model
        for col in (set(fertilizer_model.feature_names_in_) - set(fert_input.columns)):
            fert_input[col] = 0
        fert_input = fert_input[fertilizer_model.feature_names_in_]

        fertilizer_prediction = fertilizer_model.predict(fert_input)[0]

        # Update Firebase
        doc_ref.update({"moisture": moisture, "soil_type": soil_type, "fertilizer_recommendation": fertilizer_prediction})
        print(f"Updated session {session_id} with fertilizer recommendation '{fertilizer_prediction}'")

        return {"recommended_fertilizer": fertilizer_prediction}

    except Exception as e:
        return {"error": str(e)}
