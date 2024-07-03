import pandas as pd
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import joblib

def initial_feature_selection(train_data):
    """
    Performs initial feature selection using Logistic Regression with L1 penalty.
    """
    exploratory_LR = LogisticRegression(penalty='l1', fit_intercept=False, solver='liblinear')
    exploratory_LR.fit(train_data.drop(columns=['y']), train_data['y'])
    exploratory_results = pd.DataFrame(train_data.drop(columns=['y']).columns).rename(columns={0: 'name'})
    exploratory_results['coefs'] = exploratory_LR.coef_[0]
    exploratory_results['coefs_squared'] = exploratory_results['coefs']**2
    var_reduced = exploratory_results.nlargest(25, 'coefs_squared')
    return var_reduced['name'].to_list()

def final_model(train_data, variables):
    """
    Fits the final logistic regression model.
    """
    final_logit = sm.Logit(train_data['y'], train_data[variables])
    final_result = final_logit.fit()
    return final_result

# Save the model, variables, imputer, and scaler
def save_model(model, variables, imputer, scaler, 
               model_path='models/model.pkl', variables_path='models/variables.pkl', 
               imputer_path='models/imputer.pkl', scaler_path='models/scaler.pkl'):
    """
    Saves the model, variables train on, and the scaler/imputer used on the training data
    """
    joblib.dump(model, model_path)
    joblib.dump(variables, variables_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(imputer, imputer_path)
