import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('agriculture.db')
cursor = conn.cursor()

# Create the table with required fields
cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,  -- Auto-incrementing ID
        crop_name TEXT NOT NULL,               -- Selected crop
        temperature REAL NOT NULL,             -- Temperature input
        humidity REAL NOT NULL,                -- Humidity input
        nitrogen REAL NOT NULL,                -- Nitrogen input
        phosphorus REAL NOT NULL,              -- Phosphorus input
        potassium REAL NOT NULL,               -- Potassium input
        ph REAL NOT NULL,                      -- pH level
        rainfall REAL NOT NULL,                -- Rainfall input
        moisture REAL,                         -- Moisture (to be updated later)
        soil_type TEXT                         -- Soil Type (to be updated later)
    )
''')

# Commit changes and close connection
conn.commit()
conn.close()

print("Database and table 'user_data' have been created successfully.")
