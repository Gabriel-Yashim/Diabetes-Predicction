
import streamlit as st
import pandas as pd
import joblib
import pickle

# Load saved model and features
model = joblib.load('Diabetes_Subtype_RF_Model.pkl')

def preprocess_input(data):
    # Convert categorical inputs to numerical values
    family_history_mapping = {'No': 0, 'Yes': 1}

    fam_hist = data['Family history']
    if isinstance(fam_hist, list):
        fam_hist_value = fam_hist[0]
    else:
        fam_hist_value = fam_hist
    data['Family history'] = family_history_mapping.get(fam_hist_value, 1)

    df = pd.DataFrame(data)
    return df

def main():
    st.title("Diabetes Type Predictor")

    st.write("Enter patient information below: ")

    # Create form for user input
    with st.form(key='input_form'):
        age = st.number_input("Age", min_value=0, value=0)
        bmi = st.number_input("BMI", min_value=0.0, value=0.0)
        bp = st.number_input("Blood Pressure", min_value=0, value=0)
        glucose = st.number_input("Glucose", min_value=0, value=0)
        insulin = st.number_input("Insulin", min_value=0, value=0)
        hba1c = st.number_input("HbA1c", min_value=0.0, value=0.0)
        cholesterol = st.number_input("Cholesterol", min_value=0, value=0)
        family_history = st.selectbox("Family History", ['No', 'Yes'])

        submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Gather input data
        input_data = {
            'Age': [age],
            'BMI': [bmi],
            'Blood pressure': [bp],
            'Glucose': [glucose],
            'Insulin': [insulin],
            'HbA1c': [hba1c],
            'Cholesterol': [cholesterol],
            'Family history': [family_history]
        }

        # Preprocess input data
        input_df = preprocess_input(input_data)

        prediction = model.predict(input_df)

        predicted_label = 'Type I' if prediction[0] == 0 else 'Type II'

        st.success(f"Predicted Diabetes Type: {predicted_label}")
    else:
        st.info("Please fill in the form and click 'Predict'.")
    
if __name__ == '__main__':
    main()