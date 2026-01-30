import streamlit as st
import pandas as pd
import zipfile
import os

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# ================================
# LOAD ORDERS FROM ZIP
# ================================
@st.cache_data
def load_orders():
    zip_path = "orders_cleaned.zip"
    extract_path = "orders_data"

    # Extract only once
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    # Find CSV inside extracted folder
    for file in os.listdir(extract_path):
        if file.endswith(".csv"):
            return pd.read_csv(os.path.join(extract_path, file))

    st.error("❌ CSV file not found inside ZIP")
    return None


@st.cache_data
def load_other_files():
    rfm = pd.read_csv("rfm_segments.csv")
    products = pd.read_csv("product_master.csv")
    return rfm, products


# ================================
# LOAD DATA
# ================================
st.success("Loading data...")

orders = load_orders()
rfm, product_master = load_other_files()

st.success("Data loaded successfully ✅")

# ================================
# DASHBOARD
# ================================
st.title("📊 E-Commerce Analytics Dashboard")

st.subheader("📦 Orders Data")
st.dataframe(orders.head())

st.subheader("📊 RFM Segments")
st.dataframe(rfm.head())

st.subheader("🛒 Product Master")
st.dataframe(product_master.head())

st.success("🚀 Dashboard Ready")
