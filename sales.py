import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Sales Forecasting Dashboard",
    layout="wide"
)

# ================= TITLE =================
st.title("📊 AI Sales Forecasting Dashboard")
st.write("Analyze historical sales and forecast future trends using Machine Learning.")

# ================= LOAD DATA =================
df = pd.read_csv("Sales/sales_data.csv")

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

forecast_months = st.sidebar.slider(
    "Select Forecast Months",
    3,
    12,
    6
)

filtered_df = df[
    (df["State"].isin(state_filter)) &
    (df["Category"].isin(category_filter))
]

# ================= KPI METRICS =================
total_sales = filtered_df["Sales"].sum()
total_quantity = filtered_df["Quantity"].sum()
avg_sales = filtered_df["Sales"].mean()

k1, k2, k3 = st.columns(3)

k1.metric("Total Sales", f"{total_sales:,.0f}")
k2.metric("Total Quantity", f"{total_quantity}")
k3.metric("Average Sales", f"{avg_sales:,.0f}")

# ================= MONTHLY SALES =================
monthly_sales = (
    filtered_df
    .groupby(pd.Grouper(key="OrderDate", freq="M"))["Sales"]
    .sum()
    .reset_index()
)

# ================= PAST SALES TREND =================
fig1 = px.line(
    monthly_sales,
    x="OrderDate",
    y="Sales",
    title="📈 Monthly Sales Trend"
)

st.plotly_chart(fig1, use_container_width=True)

# ================= FORECAST MODEL =================

monthly_sales["time_index"] = np.arange(len(monthly_sales))

X = monthly_sales[["time_index"]]
y = monthly_sales["Sales"]

model = LinearRegression()
model.fit(X, y)

# predictions for existing data
predictions = model.predict(X)

# evaluation
mae = mean_absolute_error(y, predictions)
rmse = np.sqrt(mean_squared_error(y, predictions))
r2 = r2_score(y, predictions)

st.subheader("📊 Model Evaluation")

m1, m2, m3 = st.columns(3)

m1.metric("MAE", f"{mae:,.2f}")
m2.metric("RMSE", f"{rmse:,.2f}")
m3.metric("R² Score", f"{r2:.2f}")

# ================= FUTURE FORECAST =================

future_index = np.arange(len(monthly_sales) + forecast_months)

future_predictions = model.predict(future_index.reshape(-1, 1))

future_dates = pd.date_range(
    start=monthly_sales["OrderDate"].iloc[0],
    periods=len(future_index),
    freq="M"
)

forecast_df = pd.DataFrame({
    "Date": future_dates,
    "Forecasted Sales": future_predictions
})

# ================= FORECAST VISUALIZATION =================

fig2 = px.line(
    forecast_df,
    x="Date",
    y="Forecasted Sales",
    title="🔮 Sales Forecast"
)

fig2.add_scatter(
    x=monthly_sales["OrderDate"],
    y=monthly_sales["Sales"],
    mode="lines",
    name="Actual Sales"
)

st.plotly_chart(fig2, use_container_width=True)

# ================= CATEGORY SALES =================

category_sales = (
    filtered_df
    .groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

fig3 = px.pie(
    category_sales,
    names="Category",
    values="Sales",
    title="🧩 Sales Distribution by Category",
    hole=0.4
)

st.plotly_chart(fig3, use_container_width=True)

# ================= STATE SALES =================

state_sales = (
    filtered_df
    .groupby("State")["Sales"]
    .sum()
    .reset_index()
)

fig4 = px.bar(
    state_sales,
    x="State",
    y="Sales",
    title="🏙 Sales by State"
)

st.plotly_chart(fig4, use_container_width=True)

# ================= DATA TABLE =================
st.subheader("📄 Filtered Data")
st.dataframe(filtered_df)
