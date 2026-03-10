import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(page_title="Customer Segmentation", layout="centered")

st.title("Customer Segmentation System")
st.write(
    "This application predicts the **customer segment** using a K-Means clustering model "
    "based on customer **Annual Income and Spending Score**."
)

st.markdown("---")

# Sidebar inputs
st.sidebar.header("Customer Information")

annual_income = st.sidebar.slider(
    "Annual Income (k$)", min_value=10, max_value=150, value=60
)

spending_score = st.sidebar.slider(
    "Spending Score (1-100)", min_value=1, max_value=100, value=50
)

# Load trained model
model = pickle.load(open("kmeans_model.pkl", "rb"))

# Input dataframe
input_data = pd.DataFrame(
    {
        "Annual Income (k$)": [annual_income],
        "Spending Score (1-100)": [spending_score],
    }
)

st.subheader("Customer Input Data")
st.write(input_data)

# Prediction
if st.button("Predict Customer Segment"):

    cluster = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if cluster == 0:
        st.success("Segment: High Income – High Spending (Premium Customers)")
    elif cluster == 1:
        st.info("Segment: High Income – Low Spending")
    elif cluster == 2:
        st.warning("Segment: Low Income – High Spending")
    elif cluster == 3:
        st.error("Segment: Low Income – Low Spending")
    else:
        st.write(f"Segment Group: {cluster}")

st.markdown("---")
st.markdown("Built with ❤️ using **Streamlit & Machine Learning**")