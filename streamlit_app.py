import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Set up the layout of the UI
st.set_page_config(page_title="House Price Predictor", layout="centered", page_icon="🏡")

st.title("🏡 Advanced House Price Predictor")
st.write("Enter the details of the house below to quickly predict its expected sale price based on our trained XGBoost model!")

# Cache the models so they only load once per session reliably
@st.cache_resource
def load_models():
    preprocessor = joblib.load('preprocessor.pkl')
    model = joblib.load('model.pkl')
    return preprocessor, model

preprocessor, model = load_models()
EXPECTED_COLUMNS = preprocessor.feature_names_in_

# Build the layout for input fields
st.header("House Features")
col1, col2 = st.columns(2)

with col1:
    year_built = st.number_input("Year Built", min_value=1800, max_value=2030, value=2000)
    yr_sold = st.number_input("Year Sold", min_value=2000, max_value=2030, value=2010)
    overall_qual = st.slider("Overall Quality (1-10)", min_value=1, max_value=10, value=7)
    
st.header("Location & Surroundings")
col3, col4 = st.columns(2)

with col3:
    neighborhoods = ['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer', 'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV', 'Edwards', 'Timber', 'Gilbert', 'StoneBr', 'ClearCr', 'NPkVill', 'Blmngtn', 'BrDale', 'SWISU', 'Blueste']
    neighborhood = st.selectbox("Neighborhood", sorted(neighborhoods), index=sorted(neighborhoods).index("CollgCr"))

with col4:
    conditions = ['Norm', 'Feedr', 'PosN', 'Artery', 'RRAe', 'RRNn', 'RRAn', 'PosA', 'RRNe']
    condition1 = st.selectbox("Proximity Condition 1", sorted(conditions), index=sorted(conditions).index("Norm"))
    condition2 = st.selectbox("Proximity Condition 2", sorted(conditions), index=sorted(conditions).index("Norm"))


with col2:
    gr_liv_area = st.number_input("Ground Living Area (sq ft)", min_value=300, max_value=10000, value=1500)
    garage_area = st.number_input("Garage Area (sq ft)", min_value=0.0, max_value=3000.0, value=400.0)

# The predict button logic
if st.button("Predict Price", type="primary"):
    # Scaffold an empty dataset containing the columns the preprocessor expects
    row = {col: [None] for col in EXPECTED_COLUMNS}
    df = pd.DataFrame(row)
    
    # Variables to be inserted into the dataframe
    input_data = {
        'YearBuilt': year_built,
        'YrSold': yr_sold,
        'OverallQual': overall_qual,
        'GrLivArea': gr_liv_area,
        'GarageArea': garage_area,
        'Neighborhood': neighborhood,
        'Condition1': condition1,
        'Condition2': condition2
    }
    
    # Insert modified variables into their respective columns in the dataframe
    for key, val in input_data.items():
        if key in df.columns:
            df.at[0, key] = val
            
    # Apply our simplified Feature Engineering
    if 'YrSold' in df.columns and 'YearBuilt' in df.columns:
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    try:
        # Preprocess features into numbers, and pass to XGBoost
        X_processed = preprocessor.transform(df)
        log_pred = model.predict(X_processed)
        
        # Exponential transformation to return to normal $ values
        price = np.expm1(log_pred[0])
        
        st.success(f"### Predicted Price: ${price:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
