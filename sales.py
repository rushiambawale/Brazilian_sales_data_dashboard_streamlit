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
    if not os.path.exists(zip_path):
        return None
    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    for file in os.listdir(extract_path):
        if file.endswith(".csv"):
            df = pd.read_csv(os.path.join(extract_path
