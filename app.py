
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠"
)

models = joblib.load("california_models.pkl")

st.title("🏠 California House Price Prediction")
st.write("Predict a house value using your trained ML models.")

model_name = st.selectbox(
    "Choose a Machine Learning Model",
    list(models.keys())
)

st.subheader("Enter House Details")

MedInc = st.number_input(
    "Median Income (units of $10,000)",
    min_value=0.0, value=3.5
)
HouseAge = st.number_input(
    "House Age (years)", min_value=1.0, value=30.0
)
AveRooms = st.number_input(
    "Average Rooms", min_value=0.1, value=5.0
)
AveBedrms = st.number_input(
    "Average Bedrooms", min_value=0.1, value=1.0
)
Population = st.number_input(
    "Neighborhood Population", min_value=1.0, value=1000.0
)
AveOccup = st.number_input(
    "Average Occupancy", min_value=0.1, value=3.0
)
Latitude = st.number_input(
    "Latitude", min_value=32.0, max_value=42.0, value=34.0
)
Longitude = st.number_input(
    "Longitude", min_value=-125.0, max_value=-114.0, value=-118.0
)

if st.button("Predict House Price"):
    input_data = pd.DataFrame([{
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude
    }])

    try:
        prediction = models[model_name].predict(input_data)[0]
        price = prediction * 100000

        st.success("Prediction completed!")
        st.metric("Estimated House Value", f"${price:,.0f}")
        st.caption(f"Model used: {model_name}")
    except Exception as e:
        st.error(f"Prediction error: {e}")
