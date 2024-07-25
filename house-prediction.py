import streamlit as st
import pandas as pd
import joblib

# Pre-trained model and preprocessing objects
model_path = "model_rf.pkl"
model = joblib.load(model_path)

scaler_path = "scaler.pkl"
scaler = joblib.load(scaler_path)

# Page config
st.set_page_config(
    page_title='House Price Prediction',
    page_icon='🏠',
    layout='wide',
    initial_sidebar_state='expanded'
)


# Function to display the main area
def main_area():
    st.title("🏠 House Price Prediction App")
    st.subheader("This app predicts the **house price** with its features!")


# Function to display the sidebar
def sidebar_area():
    st.sidebar.header('User Input Parameters')


# Function to get user input from the sidebar
def user_input_features():
    bedrooms = st.sidebar.number_input("Bedrooms", min_value=0, step=1)
    bathrooms = st.sidebar.number_input("Bathrooms", min_value=0.0, step=0.25)
    sqft_living = st.sidebar.number_input("Square Feet Living", min_value=0,
                                          step=1)
    sqft_lot = st.sidebar.number_input("Square Feet Lot", min_value=0, step=1)
    floors = st.sidebar.number_input("Floors", min_value=0.0, step=0.5)
    waterfront = st.sidebar.selectbox("Waterfront", ["No", "Yes"])
    waterfront = 1 if waterfront == "Yes" else 0  # Convert to numeric
    view = st.sidebar.number_input("View", min_value=0, max_value=4, step=1)
    condition = st.sidebar.number_input("Condition", min_value=1, max_value=5,
                                        step=1)
    grade = st.sidebar.number_input("Grade", min_value=1, max_value=13, step=1)
    sqft_above = st.sidebar.number_input("Square Feet Above", min_value=0,
                                         step=1)
    sqft_basement = st.sidebar.number_input("Square Feet Basement",
                                            min_value=0, step=1)
    latitude = st.sidebar.number_input("Latitude", min_value=47.0,
                                       max_value=48.0, step=0.000001,
                                       format="%.6f")
    longitude = st.sidebar.number_input("Longitude", min_value=-123.0,
                                        max_value=-121.0, step=0.000001,
                                        format="%.6f")

    lat_long_interaction = latitude * longitude
    age = st.sidebar.number_input("Age of House", min_value=0, step=1)
    renov_age = st.sidebar.number_input("Years Since Last Renovation",
                                        min_value=0, step=1)

    data = {
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft_living': sqft_living,
        'sqft_lot': sqft_lot,
        'floors': floors,
        'waterfront': waterfront,
        'view': view,
        'condition': condition,
        'grade': grade,
        'sqft_above': sqft_above,
        'sqft_basement': sqft_basement,
        'lat_long_interaction': lat_long_interaction,
        'age': age,
        'renov_age': renov_age,
    }
    features = pd.DataFrame(data, index=[0])
    return features


# Set the design
main_area()
sidebar_area()

# Get user input
df = user_input_features()

# Display user input
st.subheader('Please Input the parameters')
st.write(df)

# Add a custom CSS style for the submit button
st.markdown(
    """
    <style>
    .stButton button {
        background-color: #4CAF50; /* Green background */
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add a submit button with the custom style
if st.sidebar.button('Submit'):
    # Impute and standardize the input data
    input_scaled = scaler.transform(df)

    # Apply model to make predictions
    prediction = model.predict(input_scaled)

    # Display the prediction in a highlighted box
    st.markdown(
        f"""
        <div style="margin-top: 30px; border:3px solid #000000; padding: 10px;
        border-radius: 10px; text-align: center;">
            <h2 style="font-size: 30px;">The predicted house price is:
            ${prediction[0]:,.2f}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
