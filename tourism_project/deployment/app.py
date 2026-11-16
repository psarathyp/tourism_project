from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# Load the trained model committed to the repo by the GitHub Actions pipeline
MODEL_PATH = Path(__file__).parent / "best_tourism_model_v1.joblib"
CLASSIFICATION_THRESHOLD = 0.45  # same threshold used during training (train.py)


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


model = load_model()

# Streamlit UI
st.title("Wellness Tourism Package Prediction")
st.write(
    "This app predicts whether a customer is likely to purchase the "
    "**Wellness Tourism Package** offered by *Visit with Us*. "
    "Enter the customer details below and click **Predict**."
)

st.subheader("Customer Details")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=0.0, value=25000.0, step=500.0)
with col2:
    number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    number_of_children_visiting = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0)
    preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
    number_of_trips = st.number_input("Number of Trips per Year", min_value=0, max_value=30, value=3)
    passport = st.selectbox("Has Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])

st.subheader("Customer Interaction Data")
col3, col4 = st.columns(2)
with col3:
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
with col4:
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0, max_value=200, value=15)

# Collect the user inputs into a dataframe with the same columns used in training
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": type_of_contact,
    "CityTier": city_tier,
    "DurationOfPitch": duration_of_pitch,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "NumberOfFollowups": number_of_followups,
    "ProductPitched": product_pitched,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "PitchSatisfactionScore": pitch_satisfaction_score,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict"):
    probability = model.predict_proba(input_data)[0, 1]
    prediction = int(probability >= CLASSIFICATION_THRESHOLD)

    if prediction == 1:
        st.success(f"The customer is **likely to purchase** the Wellness Tourism Package (probability: {probability:.2%}).")
    else:
        st.warning(f"The customer is **unlikely to purchase** the Wellness Tourism Package (probability: {probability:.2%}).")
