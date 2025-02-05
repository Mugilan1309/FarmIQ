import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load datasets
crop_df = pd.read_csv(r"C:\Users\mugil\Downloads\archive (2)\Crop_recommendation.csv")  # Crop recommendation dataset
fertilizer_df = pd.read_csv(r"C:\Users\mugil\Downloads\Fertilizer Recommendation.csv")  # Fertilizer dataset

# Define features and target for crop recommendation
X_crop = crop_df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y_crop = crop_df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# Train Random Forest model for crop recommendation
rf_crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_crop_model.fit(X_train, y_train)

# Evaluate crop model
y_pred_crop = rf_crop_model.predict(X_test)
'''print(f"Crop Model Accuracy: {accuracy_score(y_test, y_pred_crop):.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_crop))'''

# Save the crop model
joblib.dump(rf_crop_model, "crop_recommendation_model.pkl")

# Train Fertilizer Recommendation Model
fertilizer_features = ['Temperature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous']
X_fert = fertilizer_df[fertilizer_features]
y_fert = fertilizer_df['Fertilizer']

# Encode categorical variables (if any)
X_fert = pd.get_dummies(X_fert, columns=['Soil_Type', 'Crop_Type'])

# Train-test split
X_train_fert, X_test_fert, y_train_fert, y_test_fert = train_test_split(X_fert, y_fert, test_size=0.2, random_state=42)

# Train Random Forest model for fertilizer recommendation
rf_fert_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_fert_model.fit(X_train_fert, y_train_fert)

# Evaluate fertilizer model
y_pred_fert = rf_fert_model.predict(X_test_fert)
#print(f"Fertilizer Model Accuracy: {accuracy_score(y_test_fert, y_pred_fert):.2f}")

# Save the fertilizer model
joblib.dump(rf_fert_model, "fertilizer_recommendation_model.pkl")

# Corrected crop-soil mapping
crop_soil_map = {
    'rice': ['Clayey', 'Loamy'],
    'maize': ['Loamy', 'Sandy'],
    'chickpea': ['Loamy', 'Red'],
    'kidneybeans': ['Loamy', 'Red'],
    'pigeonpeas': ['Loamy', 'Black'],
    'mothbeans': ['Sandy', 'Red'],
    'mungbean': ['Loamy', 'Black'],
    'blackgram': ['Loamy', 'Red'],
    'lentil': ['Loamy', 'Black'],
    'pomegranate': ['Loamy', 'Sandy'],
    'banana': ['Loamy', 'Clayey'],
    'mango': ['Loamy', 'Clayey'],
    'grapes': ['Loamy', 'Sandy'],
    'watermelon': ['Sandy', 'Loamy'],
    'muskmelon': ['Sandy', 'Loamy'],
    'apple': ['Loamy', 'Clayey'],
    'orange': ['Loamy', 'Sandy'],
    'papaya': ['Loamy', 'Clayey'],
    'coconut': ['Loamy', 'Sandy'],
    'cotton': ['Loamy', 'Sandy'],
    'jute': ['Loamy', 'Clayey'],
    'coffee': ['Loamy', 'Red']
}

# Function to handle numeric inputs with range validation
def get_numeric_input(prompt, min_value=None, max_value=None, display_range=False):
    while True:
        if display_range:
            print(f"(Range: {min_value} to {max_value})")
        try:
            value = float(input(prompt))
            # Check if the value is within a reasonable range
            if (min_value is not None and value < min_value) or (max_value is not None and value > max_value):
                print(f"Please enter a value between {min_value} and {max_value}.")
            else:
                return value
        except ValueError:
            print("Invalid input. Please enter a valid number.")

# Function to handle categorical inputs
def get_categorical_input(prompt, valid_options, display_options=False):
    while True:
        if display_options:
            print(f"Options: {', '.join(valid_options)}")
        value = input(prompt).strip()
        if value in valid_options:
            return value
        else:
            print(f"Invalid choice. Please select from {', '.join(valid_options)}.")

# Function to predict crop and fertilizer based on user input
def predict_crop_and_fertilizer():
    print("Enter values for the following features:")

    # Get numeric inputs with validation
    N = get_numeric_input("N: ", 0, 200, display_range=True)
    P = get_numeric_input("P: ", 0, 200, display_range=True)
    K = get_numeric_input("K: ", 0, 200, display_range=True)
    temperature = get_numeric_input("Temperature: ", 0, 50, display_range=True)  # Temperature in Celsius
    humidity = get_numeric_input("Humidity: ", 0, 100, display_range=True)  # Humidity in percentage
    ph = get_numeric_input("pH: ", 0, 14, display_range=True)  # pH scale typically ranges from 0 to 14
    rainfall = get_numeric_input("Rainfall: ", 0, 5000, display_range=True)  # Rainfall in mm (could be adjusted based on dataset)

    # Predict top 3 crops
    input_array = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(1, -1)
    probabilities = rf_crop_model.predict_proba(input_array)
    top_indices = np.argsort(probabilities[0])[-3:][::-1]
    top_crops = [rf_crop_model.classes_[i] for i in top_indices]
    
    print(f"Recommended Crops: {', '.join(top_crops)}")
    
    # Accept or decline crops
    chosen_crop = None
    for crop in top_crops:
        choice = input(f"Do you want to proceed with {crop}? (yes/no): ").strip().lower()
        if choice == 'yes':
            chosen_crop = crop
            break
    
    if not chosen_crop:
        print("No crop selected. Exiting...")
        return
    
    # Suggest soil types based on the chosen crop
    print(f"Suggested Soil Types for {chosen_crop}: {', '.join(crop_soil_map.get(chosen_crop, []))}")
    soil_type = get_categorical_input("Soil Type (choose one from the suggested options): ", crop_soil_map.get(chosen_crop, []), display_options=True)

    # Ask for additional features for fertilizer prediction
    moisture = get_numeric_input("Moisture level: ", 0, 100, display_range=True)  # Moisture as percentage
    crop_type = chosen_crop  # Use the chosen crop as crop type for fertilizer prediction
    
    # Prepare input for fertilizer prediction
    fert_input = pd.DataFrame([[temperature, humidity, moisture, soil_type, crop_type, N, P, K]],
                              columns=['Temperature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorous'])
    fert_input = pd.get_dummies(fert_input, columns=['Soil_Type', 'Crop_Type'])
    
    # Align columns with the trained model
    missing_cols = set(X_fert.columns) - set(fert_input.columns)
    for col in missing_cols:
        fert_input[col] = 0  # Add missing columns with default value 0
    fert_input = fert_input[X_fert.columns]  # Ensure column order matches
    
    # Predict fertilizer
    fertilizer_prediction = rf_fert_model.predict(fert_input)[0]
    print(f"Recommended Fertilizer for {chosen_crop}: {fertilizer_prediction}")

# Run the prediction function
predict_crop_and_fertilizer()