import pandas as pd
import joblib
import numpy as np
from src.data_prep import clean_data, impute_data, scale_data, create_dummies, convert_bool_to_numeric, save_var_reduced
from src.model import initial_feature_selection, final_model, save_model

sample_data = {
    "x0": "-1.018506",
    "x1": "-4.180869",
    "x2": "5.703058724",
    "x3": "-0.522021597",
    "x4": "-1.678553956",
    "x5": "tuesday",
    "x6": "0.18617",
    "x7": "30.162959",
    "x8": "1.200073",
    "x9": "0.373124",
    "x10": "14.973894",
    "x11": "-0.81238",
    "x12": "$6,882.34 ",
    "x13": "0.078341",
    "x14": "32.823072",
    "x15": "0.02048",
    "x16": "0.171077",
    "x17": "14.236199",
    "x18": "-18.646051",
    "x19": "0.575313",
    "x20": "0.068703",
    "x21": "-0.276702",
    "x22": "0.754378",
    "x23": "3.103192",
    "x24": "-101.889723",
    "x25": "1.49565",
    "x26": "3.412199",
    "x27": "0.601394",
    "x28": "14.210012",
    "x29": "0.558285",
    "x30": "4.21066",
    "x31": "germany",
    "x32": "0.07303966",
    "x33": "2.99793546",
    "x34": "-1.91981754",
    "x35": "1.11327381",
    "x36": "-0.75988365",
    "x37": "3.00740356",
    "x38": "-1.76639977",
    "x39": "-1.93067723",
    "x40": "288.2",
    "x41": "129.79",
    "x42": "366.71",
    "x43": "-1134.56",
    "x44": "0.98441208",
    "x45": "1.10833973",
    "x46": "0.495749506",
    "x47": "0.422930348",
    "x48": "1.628712455",
    "x49": "0.402797858",
    "x50": "-0.272326826",
    "x51": "1.48269105",
    "x52": "-2.095101799",
    "x53": "0.33612654",
    "x54": "0.39604464",
    "x55": "0.43767884",
    "x56": "0.137700027",
    "x57": "0.53142961",
    "x58": "0.228881625",
    "x59": "-0.222421763",
    "x60": "0.561192069",
    "x61": "1.129407195",
    "x62": "0.373941237",
    "x63": "62.59%",
    "x64": "33.79248734",
    "x65": "-0.1522697",
    "x66": "0.34106988",
    "x67": "14.39211979",
    "x68": "-20.60214825",
    "x69": "0.02168046",
    "x70": "0.12436805",
    "x71": "2.80831588",
    "x72": "0.48941937",
    "x73": "3.07847637",
    "x74": "-86.44286813",
    "x75": "0.4088527",
    "x76": "",
    "x77": "0.80646678",
    "x78": "14.02814387",
    "x79": "0.12779922",
    "x80": "3.25437849",
    "x81": "April",
    "x82": "Female",
    "x83": "0.460470644",
    "x84": "-1.129221693",
    "x85": "-0.124149454",
    "x86": "-1.650432198",
    "x87": "-1.295166064",
    "x88": "0.076903248",
    "x89": "-1.123881898",
    "x90": "0.323156018",
    "x91": "0.04191",
    "x92": "0.33889244",
    "x93": "3.52499912",
    "x94": "-97.7151381",
    "x95": "1.44463704",
    "x96": "2.72855326",
    "x97": "0.71872513",
    "x98": "-32.94590765",
    "x99": "2.55535888"
}

# Convert the sample data to a DataFrame
input_data = pd.DataFrame([sample_data])
#input_data = input_data.drop(columns=['x5', 'x31', 'x81', 'x82'])

# Load the model, variables, and scaler
model_path = 'models/model.pkl'
variables_path = 'models/variables.pkl'
scaler_path = 'models/scaler.pkl'

print(f"Loading model from {model_path}")
model = joblib.load(model_path)

print(f"Loading variables from {variables_path}")
variables = joblib.load(variables_path)

print(f"Loading scaler from {scaler_path}")
scaler = joblib.load(scaler_path)

print(f"Variables: {variables}")
print(f"Input data before cleaning: {input_data}")

# Clean the input data
input_data = clean_data(input_data)
print(f"Input data after cleaning: {input_data}")

input_data_std = input_data

# Create dummies for categorical variables
input_data_std = create_dummies(input_data, input_data_std)
print(f"Input data after creating dummies: {input_data_std}")

# Convert boolean columns to numeric
# input_data_std = convert_bool_to_numeric(input_data_std, variables)
# print(f"Input data after converting booleans: {input_data_std}")

# Ensure all expected columns are present
missing_columns = [col for col in variables if col not in input_data_std.columns]
if missing_columns:
    print(f"Missing columns: {missing_columns}")
    for col in missing_columns:
        input_data_std[col] = 0

input_data_std = input_data_std[variables]

# Convert all columns to numeric type to avoid the TypeError
input_data_std = input_data_std.apply(pd.to_numeric, errors='coerce')
print(f"Input data before scaling: {input_data_std}")

# Check for NaN values
if input_data_std.isnull().any().any():
    print("NaN values found in input data:")
    print(input_data_std.isnull().sum())

# Scale the input data
input_data_std = scaler.transform(input_data_std)
print(f"Input data after scaling: {input_data_std}")

# Making prediction
prob = model.predict(input_data_std)
predicted_class = (prob > 0.5).astype(int)

print("Predicted Class:", predicted_class.tolist())
print("Probability:", prob.tolist())
print("Model Features:", variables)
   