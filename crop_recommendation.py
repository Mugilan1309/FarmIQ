import pandas as pd
import numpy as np
import sqlite3
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
crop_df = pd.read_csv(r"C:\Users\mugil\Downloads\archive (2)\Crop_recommendation.csv")

# Define features and target
X_crop = crop_df[['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall']]
y_crop = crop_df['label']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_crop, y_crop, test_size=0.2, random_state=42)

# Train Random Forest model
rf_crop_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_crop_model.fit(X_train, y_train)

# Save the trained model
joblib.dump(rf_crop_model, "crop_recommendation_model.pkl")

# Load the trained model
model = joblib.load("crop_recommendation_model.pkl")

# Function to get numeric input from the user
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

# Predict crop function
def predict_crop():
    print("\nEnter the following details for crop prediction: ")
    user_data = {
        "nitrogen": get_numeric_input("Nitrogen (N)", 0, 200),
        "phosphorus": get_numeric_input("Phosphorus (P)", 0, 200),
        "potassium": get_numeric_input("Potassium (K)", 0, 200),
        "temperature": get_numeric_input("Temperature (°C)", 0, 50),
        "humidity": get_numeric_input("Humidity (%)", 0, 100),
        "ph": get_numeric_input("pH", 0, 14),
        "rainfall": get_numeric_input("Rainfall (mm)", 0, 5000),
    }

    # Make prediction
    input_array = np.array([[user_data[key] for key in user_data]])
    probabilities = model.predict_proba(input_array)
    top_indices = np.argsort(probabilities[0])[-3:][::-1]
    top_crops = [model.classes_[i] for i in top_indices]

    print(f"\nTop 3 Recommended Crops: {', '.join(top_crops)}")

    # User selects a crop
    while True:
        chosen_crop = input(f"Choose a crop from the recommendations ({', '.join(top_crops)}): ").strip().lower()
        if chosen_crop in top_crops:
            user_data["crop_name"] = chosen_crop
            return user_data
        print("Invalid choice. Please select from the recommended crops.")

# Get chosen crop and store it in the database
user_crop_data = predict_crop()
print(f"\nProceeding with Fertilizer Recommendation for: {user_crop_data['crop_name']}")

# Connect to the SQLite database
conn = sqlite3.connect("agriculture.db")
cursor = conn.cursor()

# Ensure the table exists with correct schema
cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        crop_name TEXT NOT NULL,
        temperature REAL NOT NULL,
        humidity REAL NOT NULL,
        nitrogen REAL NOT NULL,
        phosphorus REAL NOT NULL,
        potassium REAL NOT NULL,
        ph REAL NOT NULL,
        rainfall REAL NOT NULL,
        moisture REAL, 
        soil_type TEXT 
    )
""")

# Insert user data into the table
cursor.execute("""
    INSERT INTO user_data (crop_name, temperature, humidity, nitrogen, phosphorus, potassium, ph, rainfall, moisture, soil_type)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
""", (user_crop_data["crop_name"], user_crop_data["temperature"], user_crop_data["humidity"],
      user_crop_data["nitrogen"], user_crop_data["phosphorus"], user_crop_data["potassium"],
      user_crop_data["ph"], user_crop_data["rainfall"]))

# Commit the changes
conn.commit()

# Fetch the latest inserted record
cursor.execute("SELECT * FROM user_data ORDER BY id DESC LIMIT 1")
latest_record = cursor.fetchone()

if latest_record:
    print("\nLatest saved data:")
    column_names = ["ID", "Crop Name", "Temperature", "Humidity", "Nitrogen", "Phosphorus", "Potassium", "pH", "Rainfall", "Moisture", "Soil Type"]
    for name, value in zip(column_names, latest_record):
        print(f"{name}: {value}")
else:
    print("No data found.")

# Close the connection
conn.close()

print("\nCrop selection and inputs saved to the database.")
print("\nNow, run 'fertilizer_recommendation.py' to continue.\n")
