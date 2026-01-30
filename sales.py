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
    if not os.path.exists(zip_path):
        return None
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    for file in os.listdir(extract_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(extract_path, file))
            df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'], errors='coerce')
            return df
    return None

# ================================
# LOAD RFM AND PRODUCT CSVs
# ================================
@st.cache_data
def load_other_data(rfm_path="rfm_segments.csv", product_path="product_master.csv"):
    if not os.path.exists(rfm_path) or not os.path.exists(product_path):
        return None, None
    rfm_df = pd.read_csv(rfm_path)
    product_df = pd.read_csv(product_path)
    return rfm_df, product_df

# --------------------------
# Load data
# --------------------------
orders = load_orders()
rfm_segments, products = load_other_data()

# Fallback dummy data if missing
if orders is None:
    st.warning("⚠️ Orders data not found. Using dummy data.")
    dates = pd.date_range("2021-01-01", periods=12, freq="M")
    orders = pd.DataFrame({
        "order_purchase_timestamp": dates,
        "product_id": [101]*12,
        "total_price": np.random.randint(100, 500, size=12)
    })

if rfm_segments is None or products is None:
    st.warning("⚠️ RFM/Product data not found. Using dummy data.")
    rfm_segments = pd.DataFrame({
        "customer_id": range(1,6),
        "Recency": [10,20,5,30,40],
        "Frequency": [5,3,10,2,1],
        "Monetary": [500,300,1000,200,100]
    })
    products = pd.DataFrame({
        "product_id": [101,102,103,104,105],
        "product_category_name": ["electronics","fashion","fashion","books","electronics"],
        "price": [200,50,80,30,400],
        "product_weight_g":[500,200,300,100,800],
        "product_photos_qty":[1,2,1,3,2],
        "product_length_cm":[20,10,15,5,30],
        "product_height_cm":[10,5,8,2,15],
        "product_width_cm":[5,3,4,1,10]
    })

st.success("✅ Data loaded successfully!")

# --------------------------
# Unified Layout
# --------------------------
st.title("E-Commerce Analytics Dashboard")
tabs = st.tabs(["Forecasting", "RFM Table", "Recommendations"])

# ==========================
# 1️⃣ Forecasting Tab
# ==========================
with tabs[0]:
    st.subheader("📈 Sales Forecasting")

    orders['order_purchase_timestamp'] = pd.to_datetime(
        orders['order_purchase_timestamp'], errors='coerce'
    )

    # Aggregate monthly sales
    monthly_sales = (
        orders.groupby([pd.Grouper(key='order_purchase_timestamp', freq='M'), 'product_id'])
        .agg({'total_price': 'sum'})
        .reset_index()
        .rename(columns={'total_price': 'sales'})
    )

    product_list = monthly_sales['product_id'].unique()
    selected_product = st.selectbox("Select Product ID", product_list)
    forecast_periods = st.slider("Select months to forecast", 1, 12, 3)

    product_sales = monthly_sales[monthly_sales['product_id'] == selected_product].copy()
    product_sales = product_sales.sort_values('order_purchase_timestamp')

    # Ensure enough data
    if len(product_sales) < 6:
        st.warning("Not enough data to forecast this product (need at least 6 months).")
    else:
        product_sales['month'] = product_sales['order_purchase_timestamp'].dt.month
        product_sales['year'] = product_sales['order_purchase_timestamp'].dt.year

        # Lag & rolling features
        product_sales['lag_1'] = product_sales['sales'].shift(1)
        product_sales['lag_2'] = product_sales['sales'].shift(2)
        product_sales['lag_3'] = product_sales['sales'].shift(3)
        product_sales['rolling_mean_3'] = product_sales['sales'].rolling(3).mean()
        product_sales['rolling_mean_6'] = product_sales['sales'].rolling(6).mean()

        # Clean NaN / inf values
        product_sales = product_sales.dropna()
        product_sales = product_sales.replace([np.inf, -np.inf], np.nan).dropna()

        if len(product_sales) < 3:
            st.warning("Not enough clean data after lag/rolling features to forecast.")
        else:
            X = product_sales[['month','year','lag_1','lag_2','lag_3','rolling_mean_3','rolling_mean_6']]
            y = product_sales['sales']

            model = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=42)
            model.fit(X, y)
            product_sales['Predicted_Sales'] = model.predict(X)

            st.line_chart(
                product_sales[['sales','Predicted_Sales']].rename(
                    columns={'sales':'Actual','Predicted_Sales':'Predicted'}
                )
            )
            st.dataframe(product_sales[['order_purchase_timestamp','sales','Predicted_Sales']])

# ==========================
# 2️⃣ RFM Table Tab
# ==========================
with tabs[1]:
    st.subheader("Customer RFM Analysis")

    rfm = rfm_segments.copy()
    if 'RFM_Score' not in rfm.columns:
        rfm['R_score'] = pd.qcut(rfm['Recency'], 5, labels=[5,4,3,2,1])
        rfm['F_score'] = pd.qcut(rfm['Frequency'].rank(method='first'),5,labels=[1,2,3,4,5])
        rfm['M_score'] = pd.qcut(rfm['Monetary'],5,labels=[1,2,3,4,5])
        rfm['RFM_Score'] = rfm['R_score'].astype(str) + rfm['F_score'].astype(str) + rfm['M_score'].astype(str)

    def segment_customer(row):
        score_str = str(row['RFM_Score'])
        try:
            r, f, m = int(score_str[0]), int(score_str[1]), int(score_str[2])
        except:
            return 'Unknown'
        if r >= 4 and f >= 4 and m >= 4:
            return 'Champions'
        elif f >= 3 and m >= 3:
            return 'Loyal Customers'
        elif r >= 2:
            return 'Potential Loyalist'
        elif r <= 2:
            return 'At Risk'
        else:
            return 'Lost Customers'

    rfm['Segment'] = rfm.apply(segment_customer, axis=1)
    st.dataframe(rfm[['customer_id','Recency','Frequency','Monetary','Segment']])

# ==========================
# 3️⃣ Recommendation Tab
# ==========================
with tabs[2]:
    st.subheader("Product Recommendation")

    prod_df = products.drop_duplicates(subset='product_id').copy()
    prod_df = pd.get_dummies(prod_df, columns=['product_category_name'])
    num_cols = ['price','product_weight_g','product_photos_qty','product_length_cm','product_height_cm','product_width_cm']
    scaler = StandardScaler()
    prod_df[num_cols] = scaler.fit_transform(prod_df[num_cols])
    prod_df.set_index('product_id', inplace=True)

    knn = NearestNeighbors(n_neighbors=10, metric='cosine', algorithm='brute')
    knn.fit(prod_df)

    def get_recommendation(product_id, top_n=5):
        prod_vector = prod_df.loc[product_id].values.reshape(1,-1)
        distances, indices = knn.kneighbors(prod_vector, n_neighbors=top_n+1)
        rec_products = prod_df.index[indices[0][1:]]
        rec_df = products[products['product_id'].isin(rec_products)][['product_id','product_category_name','price']]
        return rec_df

    selected_prod = st.selectbox("Select Product for Recommendation", prod_df.index)
    top_n = st.slider("Number of Recommendations", 1, 10, 5)

    recommendations = get_recommendation(selected_prod, top_n)
    st.dataframe(recommendations)
