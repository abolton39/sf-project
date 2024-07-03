import pandas as pd
import joblib
import numpy as np
from src.data_prep import add_missing_columns, clean_data, impute_data, create_dummies, convert_bool_to_numeric
from src.model import initial_feature_selection, final_model, save_model
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


raw_test=pd.read_csv('data/exercise_26_test.csv')
#raw_test = raw_test.iloc[:1]
# Load the model and variables
model = joblib.load('models/model.pkl')
variables = joblib.load('models/variables.pkl')
scaler = joblib.load('models/scaler.pkl')

test_data = raw_test.copy(deep=True)
# Clean data by removing % and $
test_data = clean_data(test_data)

# Mean imputation
try:
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    train_all_imputed = pd.DataFrame(imputer.fit_transform(test_data.drop(columns=['x5', 'x31', 'x81', 'x82'])), columns=test_data.drop(columns=['x5', 'x31', 'x81', 'x82']).columns)
except ValueError as e:
    # For single row calls, mean doesn't work. Fill with 0s.
    train_all_imputed = test_data.drop(columns=['x5', 'x31', 'x81', 'x82']).fillna(0)

train_all_std = pd.DataFrame(scaler.transform(train_all_imputed))
#train_all_std, a = scale_data(train_all_imputed)

# Ceate dummies
input_data_imputed = create_dummies(test_data, train_all_std)

train_all = add_missing_columns(input_data_imputed, variables)
train_all = train_all[variables]
# Convert boolean columns to numeric
for col in variables:
    if train_all[col].dtype == 'bool':
        train_all[col] = train_all[col].astype(int)

# Making prediction
prob = model.predict(train_all)
predicted_class = (prob > 0.5).astype(int)

print("Predicted Class:", predicted_class.tolist())
print("Probability:", prob.tolist())
print("Model Features:", variables)
