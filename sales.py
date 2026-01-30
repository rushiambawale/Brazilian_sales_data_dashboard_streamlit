import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# ================================
# LOAD ORDERS FROM ZIP
# ================================
@st.cache_data
def load_orders(zip_path="orders_cleaned.zip", extract_path="orders_data"):
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    for file in os.listdir(extract_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(extract_path, file))
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
            return df

    st.error("❌ CSV file not found inside ZIP")
    return None

# ================================
# LOAD RFM AND PRODUCT CSVs
# ================================
@st.cache_data
def load_other_data(rfm_path="rfm_segments.csv", product_path="product_master.csv"):
    if not os.path.exists(rfm_path) or not os.path.exists(product_path):
        st.error("❌ One or more CSV files not found")
        return None, None

    rfm_df = pd.read_csv(rfm_path)
    product_df
