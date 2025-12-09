# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="House Price Classification", layout="wide")
st.title("House Price Classification using Logistic Regression")

# Load preprocessed data and create binary target
@st.cache_data
def load_data():
    data = pd.read_pickle('house_price_processed.pkl')
    median_price = data['Price'].median()
    data['Target'] = (data['Price'] > median_price).astype(int)
    return data

data = load_data()

# Dynamically select top 5 numeric features for user input
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols.remove('Price')  # remove original target if present
numeric_cols.remove('Target') # remove binary target
important_features = numeric_cols[:5]  # pick first 5 numeric columns

X = data[important_features]
y = data['Target']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression on full data
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_scaled, y)

# User input section
st.subheader("Enter House Details")
cols = st.columns(len(important_features))
input_dict = {}
for i, col_name in enumerate(important_features):
    median_val = float(data[col_name].median())
    input_dict[col_name] = cols[i].number_input(col_name, value=median_val)

# Prediction button
if st.button("Predict"):
    input_df = pd.DataFrame([input_dict])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)[0]
    result = "Price > Median" if prediction == 1 else "Price <= Median"
    st.success(f"The predicted class for the house is: **{result}**")
