import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
# Load trained model, scaler, and encoders
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("label_encoders.pkl", "rb") as encoders_file:
    label_encoders = pickle.load(encoders_file)

# Define expected feature order
FEATURE_ORDER = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan', 'contact',
                 'month', 'day_of_week', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']

# 🌟 Set up Streamlit layout
st.set_page_config(page_title="Term Deposit Prediction", layout="wide")
st.title("Term Deposit Subscription Predictor")
st.write("Predict whether a customer is likely to subscribe to a term deposit based on their profile.")

# Use Sidebar for Inputs
st.sidebar.header("Customer Details")

# Personal & Financial Information
age = st.sidebar.slider("Age", min_value=18, max_value=100, value=35, help="Customer's age in years")
job = st.sidebar.selectbox("Job", label_encoders["job"].classes_, help="Customer's profession")
marital = st.sidebar.selectbox("Marital Status", label_encoders["marital"].classes_)
education = st.sidebar.selectbox("Education", label_encoders["education"].classes_)
default = st.sidebar.radio("Credit in Default?", label_encoders["default"].classes_)
housing = st.sidebar.radio("Housing Loan?", label_encoders["housing"].classes_)
loan = st.sidebar.radio("Personal Loan?", label_encoders["loan"].classes_)

# Contact Details
contact = st.sidebar.selectbox("Contact Type", label_encoders["contact"].classes_)
month = st.sidebar.selectbox("Last Contact Month", label_encoders["month"].classes_)
day_of_week = st.sidebar.selectbox("Last Contact Day", label_encoders["day_of_week"].classes_)
poutcome = st.sidebar.selectbox("Previous Campaign Outcome", label_encoders["poutcome"].classes_)

# Campaign History
duration = st.sidebar.number_input("Last Contact Duration (seconds)", min_value=0, value=150, help="Duration of last call")
campaign = st.sidebar.number_input("Number of Contacts in Campaign", min_value=1, value=2, help="Total contacts in this campaign")
pdays = st.sidebar.number_input("Days Since Last Contact (-1 if never contacted)", min_value=-1, value=-1, help="Days since last contact (-1 = never)")
previous = st.sidebar.number_input("Previous Campaign Contacts", min_value=0, value=0, help="Number of previous contacts")

# Encode categorical variables
input_data = pd.DataFrame([[age, job, marital, education, default, housing, loan, contact, month, day_of_week, duration, campaign, pdays, previous, poutcome]],
                          columns=FEATURE_ORDER)

for col in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']:
    if col in label_encoders:
        try:
            input_data[col] = label_encoders[col].transform(input_data[col])
        except ValueError:
            input_data[col] = label_encoders[col].transform([label_encoders[col].classes_[0]])[0]

# Standardize numerical features
numeric_features = ['age', 'duration', 'campaign', 'pdays', 'previous']
input_data[numeric_features] = scaler.transform(input_data[numeric_features])

# Ensure feature order
input_data = input_data[FEATURE_ORDER]

# Make Prediction
if st.sidebar.button("Predict Subscription Likelihood"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    # Display Result
    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"✅ The customer is **likely** to subscribe to the term deposit.")
    else:
        st.warning(f"❌ The customer is **unlikely** to subscribe.")

    # Show Probability Score
    #st.progress(int(probability * 100))
    #st.write(f"**Prediction Probability: {probability:.2f}**")

    # Feature Importance 
    #try:
     #   importances = model.feature_importances_
     #   feature_importance_df = pd.DataFrame({'Feature': FEATURE_ORDER, 'Importance': importances})
     #   feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

      #  st.subheader("Feature Importance")
      # st.bar_chart(feature_importance_df.set_index("Feature"))
    #except AttributeError:
       # st.info("Feature importance not available for this model.")




