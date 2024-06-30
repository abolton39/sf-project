import unittest
import json
from fastapi.testclient import TestClient
from src.api import app

class ApiTestCase(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_predict(self):
        payload = [{
            "x0": -1.018506,
            "x1": -4.180869,
            "x2": 5.70305872366547,
            "x3": -0.522021597308617,
            "x99": 2.55535888
        }]
        response = self.client.post('/predict', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('predicted_class', data[0])
        self.assertIn('probability', data[0])

if __name__ == "__main__":
    unittest.main()
