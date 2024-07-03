# app.py
import streamlit as st
import requests
import json

def main():
    st.title("GLM Classification Model")

    # Upload JSON file
    uploaded_file = st.file_uploader("Choose a JSON file containing your data", type="json")
    
    if uploaded_file is not None:
        data = json.load(uploaded_file)
        
        # Button for prediction
        if st.button("Predict"):
            # Call the API
            response = requests.post("http://localhost:1313/predict", headers={"Content-Type": "application/json"}, data=json.dumps(data))
            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Class: {result['business_outcome']}")
                st.success(f"Probability: {result['phat']}")
            else:
                st.error(f"Error: {response.text}")

if __name__ == "__main__":
    main()
