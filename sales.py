# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from datetime import timedelta

# --------------------------
# Streamlit Page Config
# --------------------------
st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

# ================================
# 1️⃣ Load Orders from ZIP
# ================================
@st.cache_data
def load_orders_from_zip(zip_path="orders_cleaned.zip", extract_path="orders_data"):
    # Extract ZIP if folder does not exist
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    
    # Find CSV inside extracted folder
    for file in os.listdir(extract_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(extract_path, file), parse_dates=['order_purchase_timestamp'])
            return df
    st.error("❌ CSV file not found inside ZIP")
    return None

orders = load_orders_from_zip()

# ================================
# 2️⃣ Load Other CSVs
# ================================
@st.cache_data
def load_other_data():
    order_items = pd.read_csv("order_items.csv")
    products = pd.read_csv("products.csv")
    return order_items, products

order_items, products = load_other_data()

# ================================
# 3️⃣ Merge Full Dataset
# ================================
orders_full = orders.merge(order_items, on='order_id', how='left')
orders_full = orders_full.merge(products, on='product_id', how='left')

# ================================
# Streamlit Layout
# ================================
st.title("E-Commerce Analytics Dashboard")
tabs = st.tabs(["Forecasting", "RFM Table", "Recommendations"])

# --------------------------
# 4️⃣ Forecasting Tab
# --------------------------
with tabs[0]:
    st.header("Sales Forecasting")

    # Prepare monthly sales
    monthly_sales = (
        orders_full.groupby([pd.Grouper(key='order_purchase_timestamp', freq='M'), 'product_id'])
        .agg({'total_price': 'sum'})
        .reset_index()
        .rename(columns={'total_price': 'sales'})
    )

    # Forecast inputs
    forecast_periods = st.slider("Select months to forecast", 1, 12, 3)
    product_list = monthly_sales['product_id'].unique()
    selected_product = st.selectbox("Select Product ID", product_list)

    # Filter data
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

    # Features & target
    X = product_sales[['month','year','lag_1','lag_2','lag_3','rolling_mean_3','rolling_mean_6']]
    y = product_sales['sales']

    # Random Forest Model
    model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    product_sales['Predicted_Sales'] = preds

    st.subheader("Actual vs Predicted Sales")
    st.line_chart(product_sales[['sales','Predicted_Sales']].rename(columns={'sales':'Actual','Predicted_Sales':'Predicted'}))

# --------------------------
# 5️⃣ RFM Table Tab
# --------------------------
with tabs[1]:
    st.header("Customer RFM Analysis")

    reference_date = orders_full['order_purchase_timestamp'].max() + timedelta(days=1)
    rfm = orders_full.groupby('customer_id').agg({
        'order_purchase_timestamp': lambda x: (reference_date - x.max()).days,
        'order_id': 'nunique',
        'total_price': 'sum'
    }).reset_index()

    rfm.columns = ['customer_id','Recency','Frequency','Monetary']

    # RFM scoring
    rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
    rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'),5,labels=[1,2,3,4,5])
    rfm['M_score'] = pd.qcut(rfm['Monetary'],5,labels=[1,2,3,4,5])
    rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

    # Segment mapping
    def segment_customer(row):
        score = int(row['R_score']) + int(row['F_score']) + int(row['M_score'])
        if score >= 12:
            return 'Champions'
        elif score >= 9:
            return 'Loyal Customers'
        elif score >= 6:
            return 'Potential Loyalist'
        elif score >= 3:
            return 'At Risk'
        else:
            return 'Lost Customers'

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    st.dataframe(rfm[['customer_id','Recency','Frequency','Monetary','Segment']])

# --------------------------
# 6️⃣ Recommendation Tab
# --------------------------
with tabs[2]:
    st.header("Product Recommendation")

    # Prepare product features
    prod_df = products.drop_duplicates(subset='product_id').copy()
    prod_df = pd.get_dummies(prod_df, columns=['product_category_name'])
    num_cols = ['price','product_weight_g','product_photos_qty','product_length_cm','product_height_cm','product_width_cm']
    scaler = StandardScaler()
    prod_df[num_cols] = scaler.fit_transform(prod_df[num_cols])
    prod_df.set_index('product_id', inplace=True)

    # Nearest Neighbors
    knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    knn.fit(prod_df)

    # Recommendation function
    def get_recommendation(product_id, top_n=5):
        prod_vector = prod_df.loc[product_id].values.reshape(1,-1)
        distances, indices = knn.kneighbors(prod_vector, n_neighbors=top_n+1)
        rec_products = prod_df.index[indices[0][1:]]
        return rec_products

    selected_prod = st.selectbox("Select Product for Recommendation", prod_df.index)
    top_n = st.slider("Number of Recommendations", 1, 10, 5)
    recommendations = get_recommendation(selected_prod, top_n)
    st.dataframe(pd.DataFrame(recommendations, columns=["Recommended Products"]))
