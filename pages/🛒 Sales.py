import pandas as pd
#import redshift_connector
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import matplotlib.ticker as ticker
import squarify
import os
import re
from datetime import datetime
from glob import glob
import plotly.express as px
import pycountry
#import psycopg2
#import psycopg2.extras
st.set_page_config(layout="wide") 
def classify_beverage(drink):
    if drink in coffee_based:
        return "Coffee-Based"
    elif drink in chocolate_based:
        return "Chocolate-Based"
    else:
        return "Other"
    
def line_chart_streamlit_plotly(data, x, y, title="Line Chart", x_label=None, y_label=None, smooth=False, window=3):
    df = data.copy()

    # Apply smoothing if enabled
    if smooth:
        df[y] = df[y].rolling(window=window, center=True).mean()

    fig = px.line(df, x=x, y=y)
    fig.update_layout(
        xaxis_title=x_label if x_label else x,
        yaxis_title=y_label if y_label else y,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

def plot_vertical_barchart_from_df(df, x_column, y_column, bar_width=0.6, figsize=(12, 8), font_size=14):
    # Filter out rows where y_column is 0 or NaN
    df = df[df[y_column].fillna(0) != 0]

    if df.empty:
        st.warning("No data to display in the chart.")
        return None

    # df_sorted = df.sort_values(by=y_column, ascending=False)
    xdata = df[x_column].astype(str)
    ydata = df[y_column]

    min_val = ydata.min()
    max_val = ydata.max()

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    bars = ax.bar(xdata, ydata, width=bar_width, color="crimson")

    for bar in bars:
        height = bar.get_height()
        label = f"{height:,.0f}"

        if height > max_val * 0.1:
            ax.text(bar.get_x() + bar.get_width() / 2, height - (max_val * 0.01),
                    label, ha='center', va='top', color='white', fontsize=font_size-2)
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, height + (max_val * 0.01),
                    label, ha='center', va='bottom', color='white', fontsize=font_size-2)

    ax.set_ylabel(y_column, fontsize=font_size, color='white')
    ax.set_xlabel(x_column, fontsize=font_size, color='white')
    ax.tick_params(axis='both', labelsize=font_size - 2, colors='white')

    # Set y-axis limits using actual min and max
    ax.set_ylim(min_val * 0.95 if min_val > 0 else min_val - 1, max_val * 1.05)

    for spine in ax.spines.values():
        spine.set_visible(False)

    return fig  


coffee_based = [
    "Espresso",
    "Americano",
    "Americano with Milk",
    "Cortado",
    "Cappuccino",
    "Latte"
]

chocolate_based = [
    "Hot Chocolate",
    "Cocoa"
]


sales_df = pd.read_csv('pages/index_1.csv', delimiter=",", on_bad_lines='skip')
sales_df["date"] = pd.to_datetime(sales_df["date"]).dt.date
sales_df["datetime"] = pd.to_datetime(sales_df["datetime"])
sales_df["hour"] = sales_df["datetime"].dt.hour
sales_df["minute"] = sales_df["datetime"].dt.minute
sales_df["second"] = sales_df["datetime"].dt.second
sales_df["weekday"] = pd.to_datetime(sales_df["date"]).dt.day_name()

sales_df['beveragetype'] = sales_df['coffee_name'].apply(classify_beverage)

beverage_types = list(sales_df["beveragetype"].unique())
min_date = sales_df['date'].min()
max_date = sales_df['date'].max()

col1, col2 = st.columns([1, 4])  # Adjust the ratio to your layout

with col1:
    st.image("pages/diginergyconsulting_logo.jpeg") 

with col2:
    st.markdown(
        "<h1 style='margin-top: 40px; font-size: 60px;'>Diginergy</h1>",
        unsafe_allow_html=True
    )

st.markdown(
        "<br><br>",
        unsafe_allow_html=True)

st.header("Sales", divider="gray")

st.sidebar.header("Filter Options")

# print(sales_df['coffee_name'].unique())

selected_bev_type = st.sidebar.multiselect("Select beverage types", beverage_types, default=beverage_types)
selected_dates = st.sidebar.date_input("Pick a date range", (min_date, max_date))

if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    sales_df_selected = sales_df[
        sales_df["beveragetype"].isin(selected_bev_type) &
        (sales_df["date"] >= start_date) &
        (sales_df["date"] <= end_date)
    ]
else:
    st.warning("Please select a valid date range.")

col = st.columns((2, 3, 3), gap='medium')

with col[0]:
  st.subheader("Metric KPIs", divider="gray")
  coffee_sold = sales_df_selected.shape[0]
  average_sold = sales_df_selected['money'].mean()
  total_sold = sales_df_selected['money'].sum()
  diff_customers = sales_df_selected['card'].unique().shape[0]
  st.metric(label="Total Beverages Sold", value=coffee_sold, delta= 15)
  st.metric(label="Total Revenue", value=f"€ {total_sold}", delta= 1234)
  st.metric(label="Average Price Sold", value=f"€ {round(average_sold,2)}", delta= 0.4)

with col[1]:
    st.subheader("Revenue Over Time", divider="gray")
    sakes_df_grouped = sales_df_selected.groupby("date", as_index=False)["money"].sum()
    line_chart_streamlit_plotly(sakes_df_grouped, x='date', y='money', title='Revenue Over Time')

with col[2]:
    st.subheader("Weekday Revenue Average", divider="gray")
    week_df_grouped = sales_df_selected.groupby("weekday", as_index=False).size()
    week_df_grouped.rename(columns={"size": "count"}, inplace=True)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]  
    week_df_grouped['weekday'] = pd.Categorical(week_df_grouped['weekday'] , categories=weekday_order, ordered=True)
    df_sorted = week_df_grouped.sort_values(by='weekday')
    histo = plot_vertical_barchart_from_df(df_sorted, x_column="weekday", y_column="count")
    st.pyplot(histo)


st.header("Consumer Analysis", divider="gray")

col = st.columns((3, 5), gap='medium')

with col[0]:
    st.subheader("Metric KPIs", divider="gray")
    personal_df_grouped = sales_df_selected.groupby("card", as_index=False)["money"].sum()
    personal_df_grouped_count = sales_df_selected.groupby("card", as_index=False).size()
    personal_df_grouped_count.rename(columns={"size": "count"}, inplace=True)
    st.metric(label="Total Unique Customers", value=diff_customers, delta=122)  
    st.metric(label="Total Average Spend", value=f"€ {round(personal_df_grouped['money'].mean(),2)}", delta=2) 
    st.metric(label="Total Average Returns", value=round(personal_df_grouped_count['count'].mean(),2)}, delta=-1) 

with col[1]:
    sales_df_group = sales_df_selected.groupby("coffee_name", as_index=False).size()
    sales_df_group.rename(columns={"size": "count"}, inplace=True)
    sales_df_group["percent"] = round(100 * sales_df_group["count"] / sales_df_group["count"].sum(),2)
    sales_df_group_sorted = sales_df_group.sort_values(by="percent", ascending=False)
    st.subheader("Beverage Popularity", divider="gray")
    st.dataframe(
        sales_df_group_sorted,
        column_order=("coffee_name", "percent"),
        hide_index=True,
        column_config={
            "coffee_name": st.column_config.TextColumn("Beverage"),
            "percent": st.column_config.ProgressColumn(
                "Sold Percentage",
                help="Sales Percentage",
                format="%.2f%%",
                min_value=0,
                max_value=50,
            )
        }
        )
