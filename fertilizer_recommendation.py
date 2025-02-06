import pandas as pd
import sqlite3
import joblib

# Load trained model
fertilizer_model = joblib.load("fertilizer_recommendation_model.pkl")

# Crop-soil mapping
crop_soil_map = {
    'rice': ['Clayey', 'Loamy'], 'maize': ['Loamy', 'Sandy'], 'chickpea': ['Loamy', 'Red'],
    'kidneybeans': ['Loamy', 'Red'], 'pigeonpeas': ['Loamy', 'Black'], 'mothbeans': ['Sandy', 'Red'],
    'mungbean': ['Loamy', 'Black'], 'blackgram': ['Loamy', 'Red'], 'lentil': ['Loamy', 'Black'],
    'pomegranate': ['Loamy', 'Sandy'], 'banana': ['Loamy', 'Clayey'], 'mango': ['Loamy', 'Clayey'],
    'grapes': ['Loamy', 'Sandy'], 'watermelon': ['Sandy', 'Loamy'], 'muskmelon': ['Sandy', 'Loamy'],
    'apple': ['Loamy', 'Clayey'], 'orange': ['Loamy', 'Sandy'], 'papaya': ['Loamy', 'Clayey'],
    'coconut': ['Loamy', 'Sandy'], 'cotton': ['Loamy', 'Sandy'], 'jute': ['Loamy', 'Clayey'],
    'coffee': ['Loamy', 'Red']
}

# Function to get numeric input if missing
def get_numeric_input(prompt, min_value, max_value):
    while True:
        try:
            value = float(input(f"{prompt} (Range: {min_value}-{max_value}): "))
            if min_value <= value <= max_value:
                return value
            else:
                print(f"Enter a value within {min_value} and {max_value}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Connect to the database
conn = sqlite3.connect("agriculture.db")
cursor = conn.cursor()

# Fetch latest crop data
cursor.execute("""
    SELECT crop_name, temperature, humidity, nitrogen, phosphorus, potassium, moisture, soil_type
    FROM user_data ORDER BY id DESC LIMIT 1
""")
row = cursor.fetchone()

if not row:
    print("No crop data found! Run 'crop_recommendation.py' first.")
    conn.close()
    exit()

# Extract values
chosen_crop, temperature, humidity, nitrogen, phosphorus, potassium, moisture, soil_type = row
print(f"\nChosen crop: {chosen_crop}")

# Handle missing soil type
valid_soils = crop_soil_map.get(chosen_crop, [])
if not soil_type or soil_type not in valid_soils:
    print(f"Valid soil types for {chosen_crop}: {', '.join(valid_soils)}")
    soil_type = input(f"Choose a soil type ({', '.join(valid_soils)}): ").strip()

# Handle missing moisture
if moisture is None:
    moisture = get_numeric_input("Moisture level (%)", 0, 100)

# Prepare input data
fert_input_data = {
    'Temperature': temperature,
    'Humidity': humidity,
    'Moisture': moisture,
    'Nitrogen': nitrogen,
    'Phosphorus': phosphorus,
    'Potassium': potassium,
    'Soil_Type': soil_type,
    'Crop_Type': chosen_crop  # Add crop type for one-hot encoding
}

# Convert to DataFrame
fert_input_df = pd.DataFrame([fert_input_data])

# One-hot encode Soil_Type and Crop_Type
fert_input_df = pd.get_dummies(fert_input_df, columns=['Soil_Type', 'Crop_Type'])

# Load dataset to align columns
fertilizer_df = pd.read_csv(r"C:\Users\mugil\Downloads\Fertilizer Recommendation.csv")
X_fert = pd.get_dummies(fertilizer_df[['Temperature', 'Humidity', 'Moisture', 'Soil_Type', 'Crop_Type', 'Nitrogen', 'Potassium', 'Phosphorus']], 
                        columns=['Soil_Type', 'Crop_Type'])

# Add missing columns
for col in X_fert.columns:
    if col not in fert_input_df:
        fert_input_df[col] = 0

# Ensure correct column order
fert_input_df = fert_input_df[X_fert.columns]

# Predict fertilizer
fertilizer = fertilizer_model.predict(fert_input_df)[0]
print(f"\n✅ Recommended Fertilizer for {chosen_crop}: {fertilizer}")

# Close connection
conn.close()
