import pandas as pd
from data_prep import clean_data, impute_and_scale, create_dummies, save_var_reduced
from model import initial_feature_selection, convert_bool_to_numeric, final_model, save_model

# Load your data
train_data = pd.read_csv('data/exercise_26_train.csv')

# Data preparation
train_data = clean_data(train_data)
train_all_std = impute_and_scale(train_data)
train_all_std = create_dummies(train_data, train_all_std)
train_all = pd.concat([train_all_std, train_data['y']], axis=1, sort=False)

# Initial feature selection
variables = initial_feature_selection(train_all)

# Convert boolean columns to numeric
train_all = convert_bool_to_numeric(train_all, variables)

# Final model
final_result = final_model(train_all, variables)

# Save the model and variables
save_model(final_result, variables)
