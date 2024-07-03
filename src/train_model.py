import pandas as pd
from data_prep import clean_data, impute_data, scale_data, create_dummies, convert_bool_to_numeric
from model import initial_feature_selection, final_model, save_model

# Load training data
train_data = pd.read_csv('data/exercise_26_train.csv')

# Data prep
train_data = clean_data(train_data)
train_imputed, imputer = impute_data(train_data)
train_std, scaler = scale_data(train_imputed)
train_std = create_dummies(train_data, train_std)
train_all = pd.concat([train_std, train_data['y']], axis=1, sort=False)

# Initial feature selection
variables = initial_feature_selection(train_all)

# Convert boolean columns to numeric
train_all = convert_bool_to_numeric(train_all, variables)

# Train the model
final_result = final_model(train_all, variables)

# Save the model, variables, imputer, and scaler
save_model(final_result, variables, imputer, scaler)
