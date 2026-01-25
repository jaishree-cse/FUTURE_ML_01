import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Sales Forecasting Dashboard",
    layout="wide"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>

/* Main background */
[data-testid="stAppViewContainer"] {
    background-color: #eef2f7;
}

/* Remove default padding */
.block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
}

/* KPI Cards */
.kpi-card {
    padding: 18px;
    border-radius: 14px;
    color: white;
    text-align: center;
    box-shadow: 0 6px 14px rgba(0,0,0,0.15);
}

.kpi-title {
    font-size: 16px;
    opacity: 0.9;
}

.kpi-value {
    font-size: 32px;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

# ================= TITLE =================
st.markdown(
    "<h1 style='text-align:center;'>📊 AI-Powered Sales Forecasting Dashboard</h1>",
    unsafe_allow_html=True
)

# ================= LOAD DATA =================
df = pd.read_csv("sales_data.csv")
df["OrderDate"] = pd.to_datetime(df["OrderDate"])
df["Month"] = df["OrderDate"].dt.month
df["Year"] = df["OrderDate"].dt.year

# ================= SIDEBAR FILTERS =================
st.sidebar.header("🔎 Filters")

state_filter = st.sidebar.multiselect(
    "Select State",
    df["State"].unique(),
    default=df["State"].unique()
)

category_filter = st.sidebar.multiselect(
    "Select Category",
    df["Category"].unique(),
    default=df["Category"].unique()
)

filtered_df = df[
    (df["State"].isin(state_filter)) &
    (df["Category"].isin(category_filter))
]

# ================= KPI CALCULATIONS =================
total_sales = filtered_df["Sales"].sum()
total_quantity = filtered_df["Quantity"].sum()
avg_sales = filtered_df["Sales"].mean()

k1, k2, k3 = st.columns(3)

with k1:
    st.markdown(f"""
    <div class="kpi-card" style="background:linear-gradient(135deg,#4f8cff,#6fb1ff);">
        <div class="kpi-title">💰 Total Sales</div>
        <div class="kpi-value">{total_sales/1e6:.2f} M</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card" style="background:linear-gradient(135deg,#ff7a18,#ffb347);">
        <div class="kpi-title">📦 Total Quantity</div>
        <div class="kpi-value">{int(total_quantity)}</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card" style="background:linear-gradient(135deg,#00c9a7,#92fe9d);">
        <div class="kpi-title">📊 Avg Sales</div>
        <div class="kpi-value">{avg_sales:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

# ================= PAST SALES =================
past_sales = filtered_df.groupby("OrderDate")["Sales"].sum().reset_index()

fig1 = px.line(
    past_sales,
    x="OrderDate",
    y="Sales",
    title="📈 Past Sales Trend"
)
fig1.update_layout(height=260, margin=dict(t=40, b=10))

# ================= FORECAST MODEL =================
X = np.arange(len(past_sales)).reshape(-1, 1)
y = past_sales["Sales"].values

model = LinearRegression()
model.fit(X, y)

future_months = 6
future_X = np.arange(len(past_sales) + future_months).reshape(-1, 1)
forecast = model.predict(future_X)

forecast_df = pd.DataFrame({
    "Month Index": range(len(forecast)),
    "Sales": forecast
})

fig2 = px.line(
    forecast_df,
    x="Month Index",
    y="Sales",
    title="🔮 Sales Forecast"
)
fig2.update_layout(height=260, margin=dict(t=40, b=10))

# ================= SIDE-BY-SIDE CHARTS =================
c1, c2 = st.columns(2)
c1.plotly_chart(fig1, use_container_width=True)
c2.plotly_chart(fig2, use_container_width=True)

# ================= SALES BY CATEGORY =================
category_sales = (
    filtered_df.groupby("Category")["Quantity"]
    .sum()
    .reset_index()
)

fig3 = px.pie(
    category_sales,
    values="Quantity",
    names="Category",
    hole=0.45,
    title="🧩 Sales by Category"
)
fig3.update_layout(height=240, margin=dict(t=40, b=10))

st.plotly_chart(fig3, use_container_width=True)