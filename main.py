# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

# -----------------------------
# Function to load the model and label encoder
# -----------------------------
@st.cache_resource
def load_resources(model_path='poi_model.h5', encoder_path='label_encoder.joblib', feature_columns_path='feature_columns.npy'):
    if not os.path.exists(model_path):
        st.error(f"Model file '{model_path}' not found.")
        return None, None, None
    if not os.path.exists(encoder_path):
        st.error(f"Label Encoder file '{encoder_path}' not found.")
        return None, None, None
    if not os.path.exists(feature_columns_path):
        st.error(f"Feature Columns file '{feature_columns_path}' not found.")
        return None, None, None
    
    model = tf.keras.models.load_model(model_path)
    label_encoder = joblib.load(encoder_path)
    feature_columns = np.load(feature_columns_path, allow_pickle=True).tolist()
    return model, label_encoder, feature_columns

# -----------------------------
# Function to preprocess user input
# -----------------------------
def preprocess_input(user_priorities, feature_columns):
    # Initialize a DataFrame with zeros
    user_feature_vector = pd.DataFrame(0, index=[0], columns=feature_columns)
    
    # Set the appropriate priority columns to 1 based on user input
    for i, priority in enumerate(user_priorities, start=1):
        if priority != 'Unknown':
            column_name = f'PRIORITY_{i}_{priority}'
            if column_name in user_feature_vector.columns:
                user_feature_vector.at[0, column_name] = 1
    
    # If all priorities are 'Unknown', set 'PRIORITY_Unknown' to 1
    if all(priority == 'Unknown' for priority in user_priorities):
        if 'PRIORITY_Unknown' in user_feature_vector.columns:
            user_feature_vector.at[0, 'PRIORITY_Unknown'] = 1
    
    # Ensure all feature columns are present
    for col in feature_columns:
        if col not in user_feature_vector.columns:
            user_feature_vector[col] = 0
    
    # Reorder columns to match feature_columns
    user_feature_vector = user_feature_vector[feature_columns]
    
    return user_feature_vector.astype(float).values

# -----------------------------
# Streamlit App Layout
# -----------------------------
def main():
    st.title("POI Recommendation System")
    st.write("Enter your priorities to get top recommended Points of Interest (POIs).")
    
    # Load model, label encoder, and feature columns
    model, label_encoder, feature_columns = load_resources()
    
    if model is None or label_encoder is None or feature_columns is None:
        st.stop()
    
    # Define the priority options
    # Update this list based on your dataset's unique priority values
    priority_options = [
        'Adventure',
        'Scenic',
        'Relaxing',
        'Cultural',
        'Historical',
        'Unknown'  # Ensure 'Unknown' is an option
    ]
    
    st.header("Define Your Priorities")
    
    # Collect user priorities for PRIORITY_1 to PRIORITY_5
    user_priorities = []
    for i in range(1, 6):
        priority = st.selectbox(f"Priority {i}", options=priority_options, key=f'priority_{i}')
        user_priorities.append(priority)
    
    if st.button("Get Recommendations"):
        with st.spinner("Processing your input and generating recommendations..."):
            # Preprocess the input
            user_features = preprocess_input(user_priorities, feature_columns)
            
            # Make predictions
            predictions = model.predict(user_features)
            
            # Define the number of top recommendations
            N = 5  # You can make this configurable
            
            # Get the indices of the top N recommended POIs
            top_indices = np.argsort(predictions[0])[::-1][:N]
            
            # Decode the recommended POIs using the label encoder
            try:
                recommended_pois = label_encoder.inverse_transform(top_indices)
            except Exception as e:
                st.error(f"Error in decoding POIs: {e}")
                return
            
            # Get the corresponding probabilities
            recommended_probabilities = predictions[0][top_indices]
            
            # Display the recommended POIs with their probabilities
            st.success("Top Recommended POIs:")
            for idx, (poi, prob) in enumerate(zip(recommended_pois, recommended_probabilities), start=1):
                st.write(f"{idx}. **{poi}** (Probability: {prob:.2f})")
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application recommends Points of Interest (POIs) based on your defined priorities using a trained neural network model.
    
    **Developed by:** Your Name
    
    **Contact:** your.email@example.com
    """)

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    main()
