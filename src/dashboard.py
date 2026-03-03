import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import plotly.express as px
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__) , '..')))
from src.databases.db_loader import get_engine

st.set_page_config(page_title='Silent Churn Dashboard' , layout="wide")

@st.cache_resource
def load_models():

    # load XGBoost
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model("models/silent_churn_v1.json")

    # load random forest
    rf_model = joblib.load("models/rf_silent_churn_v1.joblib")

    return xgb_model , rf_model


def get_data(limit):
    engine = get_engine()
    query = f"SELECT * FROM processed_features ORDER BY usage_date DESC LIMIT {limit}"
    return pd.read_sql(query,engine)



# UI Header 
st.title("Silent churn prediction dashboard")
st.markdown("Monitor user engagement and predict potentail churn risks in real-time")

# sidebar
model_type = st.sidebar.selectbox("Select Model", ("XGBoost", "Random Forest"))
threshold = st.sidebar.slider("Risk Threshold", 0.0, 1.0, 0.85, 0.05)
data_limit = st.sidebar.number_input("Users to Analyze", 100, 10000, 1000)

# load data and models
xgb_model, rf_model = load_models()
df = get_data(data_limit)

# Inference Logic
features = ['daily_usage', 'usage_7d_avg', 'usage_30d_avg', 'usage_drop_rate']
X = df[features]

if model_type == "XGBoost":
    probs = xgb_model.predict_proba(X)[:, 1]
else:
    probs = rf_model.predict_proba(X)[:, 1]


df['risk_score'] = probs
df['is_risk'] = (df['risk_score'] >= threshold).astype(int)
at_risk_df = df[df['is_risk'] == 1].sort_values(by='risk_score', ascending=False)


# Dashboard Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Total Analyzed", len(df))
col2.metric("High Risk Users", len(at_risk_df))
col3.metric("Avg Risk Score", f"{df['risk_score'].mean():.2f}")


# Visualizations
st.subheader("Risk Distribution Analysis")
tab1, tab2 = st.tabs(["Global View (Log Scale)", "High Risk Focus"])

with tab1:
    # Logaritmik görünüm tüm resmi ama dengeli gösterir
    fig_log = px.histogram(df, x="risk_score", nbins=50, 
                           log_y=True,
                           color_discrete_sequence=['#ff4b4b'])
    st.plotly_chart(fig_log, width='stretch')

with tab2:
    # Sadece 0.3'ten büyükleri göstererek "mikro" analize odaklanır
    zoom_df = df[df['risk_score'] > 0.3]
    if not zoom_df.empty:
        fig_zoom = px.histogram(zoom_df, x="risk_score", nbins=20,
                               color_discrete_sequence=['#ffa500'])
        st.plotly_chart(fig_zoom, width='stretch')
    else:
        st.info("No users found with risk score > 0.3")

# Risk Table
st.subheader(f"High Risk Users (Threshold > {threshold})")
if not at_risk_df.empty:
    st.dataframe(at_risk_df[['account_id', 'usage_date', 'risk_score', 'usage_drop_rate']], use_container_width=True)
else:
    st.success("No high-risk users detected at this threshold.")

# Individual User Lookup
st.sidebar.markdown("---")
search_id = st.sidebar.text_input("Search Account ID (e.g. A-432483)") #

if search_id:
    user_data = df[df['account_id'] == search_id]
    
    if not user_data.empty:
        # Sort by date for chronological trend
        user_data = user_data.sort_values("usage_date")
        
        st.subheader(f"Historical Analysis for: {search_id}")
        
        # Trend Chart
        fig_trend = px.line(user_data, x="usage_date", y="usage_drop_rate",
                            title=f"Usage Drop Rate Trend - {search_id}",
                            markers=True,
                            color_discrete_sequence=['#ffa500'])
        
        # Technical fix: stretch width for modern Streamlit versions
        st.plotly_chart(fig_trend, width='stretch')
        
        # Show raw data table
        st.dataframe(user_data[['usage_date', 'daily_usage', 'usage_drop_rate', 'risk_score']], width='stretch')
    else:
        st.error("Account ID not found in the current analyzed batch.")