import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

st.set_page_config(page_title="Internet Usage Anomaly Detection", layout="centered")

st.title("Intelligent Web Application for Detecting Abnormal Internet Usage Patterns")

st.write("Upload internet usage data to detect abnormal usage patterns using Machine Learning.")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Internet Usage CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(data)

    # Select numeric columns
    features = data[['Data_Used_MB', 'Session_Duration_Min']]

    # Train Isolation Forest model
    model = IsolationForest(contamination=0.2, random_state=42)
    model.fit(features)

    # Predict anomalies
    data['Anomaly'] = model.predict(features)
    data['Status'] = data['Anomaly'].apply(lambda x: "Abnormal" if x == -1 else "Normal")

    st.subheader("Detected Abnormal Internet Usage Patterns")
    st.dataframe(data)

    # Visualization
    st.subheader("Visualization of Usage Patterns")
    fig, ax = plt.subplots()
    scatter = ax.scatter(
        data['Data_Used_MB'],
        data['Session_Duration_Min'],
        c=data['Anomaly']
    )
    ax.set_xlabel("Data Used (MB)")
    ax.set_ylabel("Session Duration (Minutes)")
    ax.set_title("Internet Usage Anomaly Detection")
    st.pyplot(fig)