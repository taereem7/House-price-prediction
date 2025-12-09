# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="House Price Classification App", layout="wide")
st.title("House Price Classification using Logistic Regression")

# Step 1: Load preprocessed data and create binary target
@st.cache_data
def load_data():
    data = pd.read_pickle('house_price_processed.pkl')
    median_price = data['Price'].median()
    data['Target'] = (data['Price'] > median_price).astype(int)
    return data

data = load_data()

# Step 2: Separate features and target
X = data.drop('Target', axis=1)
y = data['Target']

# Step 3: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Train Logistic Regression on full data
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# Step 5: Predict on new input
st.subheader("Predict for New House Data")
input_dict = {}
for col in X.columns:
    input_dict[col] = st.number_input(f"{col}", value=float(data[col].median()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    result = "Price > Median" if prediction == 1 else "Price <= Median"
    st.success(f"The predicted class for the house is: **{result}**")
