import pandas as pd
import numpy as np
import sklearn
import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
import pickle
import datetime

st.title("Flight Ticket Price Prediction App ✈")

st.subheader("by Vijay Sada")

airlines = ['Air India', 'Vistara', 'SpiceJet', 'AirAsia', 'GO FIRST', 'Indigo', 'Trujet', 'StarAir']

# Select box for choosing the airline
airline = st.selectbox(
    "Which airline would you like to fly?",
    airlines,
    index=None,
    placeholder="Select an airline...",
)

# Display the selected airline
st.write("You selected:", airline)

if airline in ["Air India", "Vistara"]:
    flight_class_options = ["Economy", "Business"]
else:
    flight_class_options = ["Economy"]

classs = st.radio(
    "Select the flight class:",
    flight_class_options,
)

# Display the selected class
st.write("You selected:", classs)

duration = st.slider(
    "How many hours would you like to spend in the flight?",
    min_value=0.8,
    max_value=49.8,
    value=3.0,  # Default value
    step=0.1
)

# Display the selected flight duration
st.write("You selected a flight duration of:", duration, "hours")

current_date = datetime.date.today()

# Calculate the maximum allowed date (current date + 49 days)
max_allowed_date = current_date + datetime.timedelta(days=49)

# Calculate the minimum allowed date (current date + 1 day)
min_allowed_date = current_date + datetime.timedelta(days=1)

departure_date = st.date_input(
    "Choose your departure date:",
    min_value=min_allowed_date,
    max_value=max_allowed_date,
)

days_left = (departure_date - current_date).days  # Calculate days left as an integer

st.write(f"Days left before your trip: {days_left} days")

stops_options = ['zero', 'one', 'multiple']

# Select box for choosing the number of stops
stops = st.selectbox(
    "Choose the number of stops:",
    stops_options,
)

# Display the selected number of stops
st.write("You selected:", stops, "stops")

model_path = r"C:\Users\vijay\Desktop\ML\PROJECTS\FIINAL_PROJECT\streamlit\price_dt_pred.pkl"

if st.button("Submit"):
    try:
        data = pd.DataFrame([[airline, classs, duration, days_left, stops]], 
                            columns=['airline', 'class', 'duration', 'days_left', 'stops'])
        
        model = pickle.load(open(model_path, "rb"))
        ticket_price = model.predict(data)[0]
        
        st.success(f"Your flight ticket costs you ₹ {ticket_price:.2f}")
    except Exception as e:
        st.error("Error predicting ticket price. Please check your inputs.")
        st.error(e)
