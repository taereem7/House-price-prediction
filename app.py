# app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="House Price Classification App", layout="wide")
st.title("House Price Classification using Logistic Regression")

# Step 1: Load preprocessed data
@st.cache_data
def load_data():
    data = pd.read_pickle('house_price_processed.pkl')
    return data

data = load_data()

st.subheader("Dataset Preview")
st.dataframe(data.head())

# Step 2: Separate features and target
X = data.drop('Target', axis=1)
y = data['Target']

# Step 3: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split into train/test (for demo purposes, we use all data here)
X_train, X_test, y_train, y_test = X_scaled, X_scaled, y, y  # Using full data

# Step 5: Train Logistic Regression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluation
st.subheader("Model Evaluation")
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, output_dict=True)

st.write(f"**Accuracy:** {accuracy:.4f}")
st.write("**Confusion Matrix:**")
st.write(conf_matrix)

st.write("**Classification Report:**")
st.dataframe(pd.DataFrame(class_report).transpose())

# Step 8: Predict on new input
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
