import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from datetime import timedelta

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# ================================
# LOAD ORDERS FROM ZIP
# ================================
@st.cache_data
def load_orders(zip_path="orders_cleaned.zip", extract_path="orders_data"):
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

# ================================
# LOAD OTHER CSV FILES
# ================================
@st.cache_data
def load_other_data(rfm_path="rfm_segments.csv", product_path="product_master.csv"):
    if not os.path.exists(rfm_path):
        st.error(f"❌ {rfm_path} not found")
        return None, None
    if not os.path.exists(product_path):
        st.error(f"❌ {product_path} not found")
        return None, None

    rfm_df = pd.read_csv(rfm_path)
    product_df = pd.read_csv(product_path)
    return rfm_df, product_df

# Load datasets
orders = load_orders()
rfm_segments, products = load_other_data()

if orders is None or rfm_segments is None or products is None:
    st.stop()  # Stop app if any file is missing

# --------------------------
# Preprocessing
# --------------------------
orders_full = orders.copy()
# Ensure order_purchase_timestamp is datetime
orders_full['order_purchase_timestamp'] = pd.to_datetime(orders_full['order_purchase_timestamp'], errors='coerce')

# --------------------------
# Streamlit App Layout
# --------------------------
st.title("E-Commerce Analytics Dashboard")
tabs = st.tabs(["Forecasting", "RFM Table", "Recommendations"])

# --------------------------
# 1️⃣ Forecasting Tab
# --------------------------
with tabs[0]:
    st.header("Sales Forecasting")

    # Prepare monthly sales data
    monthly_sales = (
        orders_full.groupby([pd.Grouper(key='order_purchase_timestamp', freq='M'), 'product_id'])
        .agg({'total_price': 'sum'})
        .reset_index()
        .rename(columns={'total_price': 'sales'})
    )

    # Forecast horizon input
    forecast_periods = st.slider("Select months to forecast", 1, 12, 3)

    # Select product for forecast
    product_list = monthly_sales['product_id'].unique()
    selected_product = st.selectbox("Select Product ID", product_list)

    # Filter data for product
    product_sales = monthly_sales[monthly_sales['product_id'] == selected_product].copy()
    product_sales['month'] = product_sales['order_purchase_timestamp'].dt.month
    product_sales['year'] = product_sales['order_purchase_timestamp'].dt.year

    # Lag & rolling features
    product_sales['lag_1'] = product_sales['sales'].shift(1)
    product_sales['lag_2'] = product_sales['sales'].shift(2)
    product_sales['lag_3'] = product_sales['sales'].shift(3)
    product_sales['rolling_mean_3'] = product_sales['sales'].rolling(3).mean()
    product_sales['rolling_mean_6'] = product_sales['sales'].rolling(6).mean()
    product_sales.dropna(inplace=True)

    # Feature & target
    X = product_sales[['month','year','lag_1','lag_2','lag_3','rolling_mean_3','rolling_mean_6']]
    y = product_sales['sales']

    # Random Forest Forecast
    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    product_sales['Predicted_Sales'] = preds

    st.line_chart(product_sales[['sales','Predicted_Sales']].rename(columns={'sales':'Actual','Predicted_Sales':'Predicted'}))

# --------------------------
# 2️⃣ RFM Table Tab
# --------------------------
with tabs[1]:
    st.header("Customer RFM Analysis")

    # Use uploaded RFM
