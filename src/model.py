import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import joblib

def initial_feature_selection(train_all):
    """
    Performs initial feature selection using Logistic Regression with L1 penalty.
    """
    exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
    exploratory_LR.fit(train_all.drop(columns=['y']), train_all['y'])
    exploratory_results = pd.DataFrame(train_all.drop(columns=['y']).columns).rename(columns={0: 'name'})
    exploratory_results['coefs'] = exploratory_LR.coef_[0]
    exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
    var_reduced = exploratory_results.nlargest(25, 'coefs_squared')
    return var_reduced['name'].to_list()

def convert_bool_to_numeric(train_all, variables):
    """
    Converts boolean columns to numeric.
    """
    for col in variables:
        if train_all[col].dtype == 'bool':
            train_all[col] = train_all[col].astype(int)
    return train_all

def final_model(train_all, variables):
    """
    Fits the final logistic regression model.
    """
    final_logit = sm.Logit(train_all['y'], train_all[variables])
    final_result = final_logit.fit()
    return final_result

# Save the model and variables
def save_model(model, variables, model_path='models/model.pkl', variables_path='models/variables.pkl'):
    joblib.dump(model, model_path)
    joblib.dump(variables, variables_path)

# Load the model and variables
def load_model(model_path='models/model.pkl', variables_path='models/variables.pkl'):
    model = joblib.load(model_path)
    variables = joblib.load(variables_path)
    return model, variables