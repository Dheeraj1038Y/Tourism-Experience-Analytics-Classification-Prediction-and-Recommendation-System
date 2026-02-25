import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load models
reg_model = joblib.load("regression_model.pkl")
clf_model = joblib.load("classification_model.pkl")
rec_model = joblib.load("recommendation_model.pkl")
user_item_matrix = joblib.load("user_item_matrix.pkl")

st.title("Tourism Prediction & Recommendation System")

# --- Regression Section ---
st.header("Regression: Predict Attraction Rating")

# Use the shape of X_train to determine feature count
n_features = reg_model.n_features_in_

input_values = []
for i in range(n_features):
    val = st.number_input(f"Feature {i+1}", value=0.0)
    input_values.append(val)

if st.button("Predict Rating"):
    features = np.array(input_values).reshape(1, -1)
    prediction = reg_model.predict(features)
    st.success(f"Predicted Rating: {prediction[0]:.2f}")

# --- Classification Section ---
st.header("Classification: Satisfied vs Unsatisfied")
n_features_clf = clf_model.n_features_in_

clf_inputs = []
for i in range(n_features_clf):
    val = st.number_input(f"Classification Feature {i+1}", value=0.0)
    clf_inputs.append(val)

if st.button("Classify Satisfaction"):
    features_class = np.array(clf_inputs).reshape(1, -1)
    prediction_class = clf_model.predict(features_class)
    label = "Satisfied" if prediction_class[0] == 1 else "Unsatisfied"
    st.success(f"Classification Result: {label}")


# --- Recommendation Section ---
st.header("Recommendation: Attractions for a User")
user_id = st.number_input("Enter User ID", min_value=1, max_value=user_item_matrix.shape[0])
if st.button("Recommend Attractions"):
    try:
        user_vector = user_item_matrix.iloc[user_id-1].values.reshape(1, -1)
        distances, indices = rec_model.kneighbors(user_vector, n_neighbors=5)
        st.success(f"Recommended Attraction IDs: {indices[0].tolist()}")
    except Exception as e:
        st.error(f"Error: {e}")
