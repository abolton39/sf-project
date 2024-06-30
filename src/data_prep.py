import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import yaml

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

def impute_and_scale(train_data):
    """
    Imputes missing values and scales the data.
    """
    imputer = SimpleImputer(missing_values = pd.NA, strategy = 'mean')
    train_all_imputed = pd.DataFrame(imputer.fit_transform(train_data.drop(columns=['y', 'x5', 'x31', 'x81', 'x82'])), 
                                     columns=train_data.drop(columns=['y', 'x5', 'x31', 'x81', 'x82']).columns)
    std_scaler = StandardScaler()
    train_all_std = pd.DataFrame(std_scaler.fit_transform(train_all_imputed), columns=train_all_imputed.columns)
    return train_all_std

def create_dummies(train_data, train_all_std):
    """
    Creates dummy variables for categorical columns.
    """
    for col in ['x5', 'x31', 'x81', 'x82']:
        dummies = pd.get_dummies(train_data[col], drop_first=True, prefix=col, prefix_sep='_', dummy_na=True)
        train_all_std = pd.concat([train_all_std, dummies], axis=1, sort=False)
    return train_all_std

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
