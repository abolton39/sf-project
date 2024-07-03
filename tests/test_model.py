import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
from model import initial_feature_selection, final_model, save_model

class TestModel(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'x1': [1, 2, 3, 4],
            'x2': [4, 3, 2, 1],
            'y': [0, 1, 0, 1]
        })

    def test_initial_feature_selection(self):
        selected_features = initial_feature_selection(self.data)
        self.assertEqual(len(selected_features), 2)

    def test_final_model(self):
        variables = ['x1', 'x2']
        model_result = final_model(self.data, variables)
        self.assertTrue(hasattr(model_result, 'params'))

    def test_save_model(self):
        variables = ['x1', 'x2']
        model_result = final_model(self.data, variables)
        save_model(model_result, variables, None, None, 'test_model.pkl', 'test_variables.pkl', 'test_imputer.pkl', 'test_scaler.pkl')
        self.assertTrue(joblib.load('test_model.pkl'))
        self.assertTrue(joblib.load('test_variables.pkl'))

if __name__ == '__main__':
    unittest.main()
