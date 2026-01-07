import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

# =========================
# PWA SUPPORT (IMPORTANT)
# =========================
st.markdown("""
<link rel="manifest" href="/static/manifest.json">
<script>
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('/static/service-worker.js');
}
</script>
""", unsafe_allow_html=True)

# =========================
# APP CONFIG
# =========================
st.set_page_config(
    page_title="Internet Usage Anomaly Detector",
    page_icon="📶",
    layout="wide"
)

st.title("📶 Intelligent Web App for Detecting Abnormal Internet Usage Patterns")
st.write(
    "This system uses **machine learning (Isolation Forest)** to detect abnormal internet usage behavior."
)

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload Internet Usage CSV File", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.subheader("📊 Uploaded Data")
    st.dataframe(data)

    # =========================
    # FEATURE SELECTION
    # =========================
    if "Data_Used_MB" in data.columns and "Session_Duration_Min" in data.columns:

        features = data[["Data_Used_MB", "Session_Duration_Min"]]

        # =========================
        # MACHINE LEARNING MODEL
        # =========================
        model = IsolationForest(
            n_estimators=100,
            contamination=0.1,
            random_state=42
        )
        data["Anomaly"] = model.fit_predict(features)

        # Convert -1 and 1 to labels
        data["Anomaly"] = data["Anomaly"].map({1: "Normal", -1: "Abnormal"})

        st.subheader("🚨 Detection Results")
        st.dataframe(data)

        # =========================
        # VISUALIZATION
        # =========================
        st.subheader("📈 Visualization of Anomalies")

        fig, ax = plt.subplots()
        colors = data["Anomaly"].map({"Normal": "blue", "Abnormal": "red"})

        ax.scatter(
            data["Session_Duration_Min"],
            data["Data_Used_MB"],
            c=colors
        )

        ax.set_xlabel("Session Duration (Minutes)")
        ax.set_ylabel("Data Used (MB)")
        ax.set_title("Abnormal Internet Usage Detection")

        st.pyplot(fig)

        st.success("Analysis completed successfully!")

    else:
        st.error(
            "CSV must contain 'Data_Used_MB' and 'Session_Duration_Min' columns."
        )

else:
    st.info("Please upload a CSV file to begin analysis.")