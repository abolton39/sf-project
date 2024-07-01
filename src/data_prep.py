import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import yaml
import joblib

def clean_data(train_data):
    """
    Cleans and preprocesses the training data.
    """
    train_data['x12'] = train_data['x12'].str.replace('$','')
    train_data['x12'] = train_data['x12'].str.replace(',','')
    train_data['x12'] = train_data['x12'].str.replace(')','')
    train_data['x12'] = train_data['x12'].str.replace('(','-')
    train_data['x12'] = train_data['x12'].astype(float)
    train_data['x63'] = train_data['x63'].str.replace('%','')
    train_data['x63'] = train_data['x63'].astype(float)
    return train_data

def impute_data(train_data):
    """
    Imputes values absent in the data.
    """
    imputer = SimpleImputer(missing_values = pd.NA, strategy = 'mean')
    train_all_imputed = pd.DataFrame(imputer.fit_transform(train_data.drop(columns=['y', 'x5', 'x31', 'x81', 'x82'])), 
                                     columns=train_data.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
    return train_all_imputed

def scale_data(train_data):
    """
    Scales the data.
    """
    std_scaler = StandardScaler()
    train_all_std = pd.DataFrame(std_scaler.fit_transform(train_data), columns=train_data.columns)
    return train_all_std, std_scaler

def create_dummies(train_data, train_all_std):
    """
    Creates dummy variables for categorical columns.
    """
    for col in ['x5', 'x31', 'x81', 'x82']:
        dummies = pd.get_dummies(train_data[col], drop_first=True, prefix=col, prefix_sep='_', dummy_na=True)
        train_all_std = pd.concat([train_all_std, dummies], axis=1, sort=False)
    return train_all_std

def convert_bool_to_numeric(train_all, variables):
    """
    Converts boolean columns to numeric.
    """
    for col in variables:
        if train_all[col].dtype == 'bool':
            train_all[col] = train_all[col].astype(int)
    return train_all

def save_var_reduced(var_reduced, file_path='data/var_reduced.yml'):
    """
    Saves the reduced variables to a YAML file.
    """
    with open(file_path, 'w') as file:
        yaml.dump(var_reduced, file)

def load_var_reduced(file_path='data/var_reduced.yml'):
    """
    Loads the reduced variables from a YAML file.
    """
    with open(file_path, 'r') as file:
        var_reduced = yaml.load(file, Loader=yaml.SafeLoader)
    return pd.DataFrame(var_reduced)
