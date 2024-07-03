import pandas as pd
import json

"""
Formats a .csv into a json for use in the streamlit app.
"""
# Load csv file
csv_path = '../data/exercise_26_test.csv'
data = pd.read_csv(csv_path)

# Convert the DataFrame to a dictionary
data_dict = data.to_dict(orient='records')

# JSON structure
json_data = {"data": data_dict}

# Save JSON structure
json_output_path = 'data.json'
with open(json_output_path, 'w') as json_file:
    json.dump(json_data, json_file, indent=4)

print(f"Data has been saved to {json_output_path}")
