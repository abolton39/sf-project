import unittest
import pandas as pd
import numpy as np
from data_prep import clean_data, impute_data, scale_data, create_dummies, convert_bool_to_numeric, add_missing_columns

class TestDataPrep(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'x12': ['$1,200.00', '$3,400.00'],
            'x63': ['12%', '34%'],
            'x5': ['monday', 'tuesday'],
            'x31': ['germany', 'france'],
            'x81': ['March', 'April'],
            'x82': ['Male', 'Female'],
            'y': [1, 0]
        })

    def test_clean_data(self):
        cleaned_data = clean_data(self.data.copy())
        self.assertTrue(pd.api.types.is_float_dtype(cleaned_data['x12']))
        self.assertTrue(pd.api.types.is_float_dtype(cleaned_data['x63']))
        self.assertEqual(cleaned_data['x12'][0], 1200.00)
        self.assertEqual(cleaned_data['x63'][1], 34.0)

    def test_impute_data(self):
        self.data.loc[1, 'x12'] = np.nan
        imputed_data, _ = impute_data(self.data.copy())
        self.assertFalse(imputed_data.isnull().values.any())

    def test_scale_data(self):
        scaled_data, _ = scale_data(self.data[['x12', 'x63']].copy())
        self.assertAlmostEqual(scaled_data.mean().sum(), 0)
        self.assertAlmostEqual(scaled_data.std().sum(), 2)

    def test_create_dummies(self):
        dummy_data = create_dummies(self.data.copy(), self.data[['x12', 'x63']].copy())
        self.assertIn('x5_tuesday', dummy_data.columns)
        self.assertIn('x31_germany', dummy_data.columns)
        self.assertIn('x81_April', dummy_data.columns)
        self.assertIn('x82_Male', dummy_data.columns)

    def test_convert_bool_to_numeric(self):
        data = pd.DataFrame({'x_bool': [True, False], 'x_int': [1, 0]})
        variables = ['x_bool', 'x_int']
        converted_data = convert_bool_to_numeric(data.copy(), variables)
        self.assertTrue(pd.api.types.is_integer_dtype(converted_data['x_bool']))

    def test_add_missing_columns(self):
        columns = ['x1', 'x2', 'x3']
        input_data = pd.DataFrame({'x1': [1], 'x2': [2]})
        complete_data = add_missing_columns(input_data.copy(), columns)
        self.assertIn('x3', complete_data.columns)
        self.assertEqual(complete_data['x3'].iloc[0], 0)

if __name__ == '__main__':
    unittest.main()
