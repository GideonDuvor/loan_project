import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open('models/model.pkl', 'rb'))

# Load dataset
df = pd.read_csv('data/credit.csv')
df.columns = df.columns.str.strip()

st.title("🏦 Loan Eligibility Prediction")

st.write("Enter applicant details:")

# Create input fields manually (SAFE VERSION)
Gender = st.selectbox("Gender", ["Male", "Female"])
Married = st.selectbox("Married", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
Education = st.selectbox("Education", ["Graduate", "Not Graduate"])
Self_Employed = st.selectbox("Self Employed", ["Yes", "No"])
ApplicantIncome = st.number_input("Applicant Income", value=5000)
CoapplicantIncome = st.number_input("Coapplicant Income", value=0)
LoanAmount = st.number_input("Loan Amount", value=100)
Loan_Amount_Term = st.number_input("Loan Term", value=360)
Credit_History = st.selectbox("Credit History", [0, 1])
Property_Area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Predict
if st.button("Check Eligibility"):

    input_data = {
        'Gender': Gender,
        'Married': Married,
        'Dependents': Dependents,
        'Education': Education,
        'Self_Employed': Self_Employed,
        'ApplicantIncome': ApplicantIncome,
        'CoapplicantIncome': CoapplicantIncome,
        'LoanAmount': LoanAmount,
        'Loan_Amount_Term': Loan_Amount_Term,
        'Credit_History': Credit_History,
        'Property_Area': Property_Area
    }

    input_df = pd.DataFrame([input_data])

    # SAME preprocessing as training
    input_df = input_df.ffill().fillna(0)

    input_df['Gender'] = input_df['Gender'].map({'Male': 1, 'Female': 0})
    input_df['Married'] = input_df['Married'].map({'Yes': 1, 'No': 0})
    input_df['Education'] = input_df['Education'].map({'Graduate': 1, 'Not Graduate': 0})
    input_df['Self_Employed'] = input_df['Self_Employed'].map({'Yes': 1, 'No': 0})

    input_df = pd.get_dummies(input_df)

    # Align with model columns
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Not Approved")