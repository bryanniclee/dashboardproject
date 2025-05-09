import pandas as pd
import redshift_connector
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
import psycopg2
import psycopg2.extras
st.set_page_config(layout="wide") 


hr_df = pd.read_csv('pages/Employee.csv', delimiter=",", on_bad_lines='skip')


education_types = list(hr_df["Education"].unique())
cities = list(hr_df["City"].unique())
genders = list(hr_df["Gender"].unique())
st.sidebar.header("Filter Options")
selected_types = st.sidebar.multiselect("Select education types", education_types, default=education_types)
selected_cities = st.sidebar.multiselect("Select Cities", cities, default=cities)
selected_gender = st.sidebar.multiselect("Select Genders", genders, default=genders)

hr_df_filter = hr_df[
        hr_df["Education"].isin(selected_types) &
        hr_df["City"].isin(selected_cities) &
        hr_df["Gender"].isin(selected_gender)
    ]

def donut_chart(df, column):
    # Set up the figure and axes with smaller size
    fig, ax = plt.subplots(figsize=(4, 3))  # Adjusted smaller size
    
    # Get the data
    data = df[column].value_counts()
    labels = data.index
    sizes = data.values

    # Update color palette to use shades of blue/green for Streamlit
    colors = sns.color_palette('Reds', n_colors=len(labels))  # You can choose other palettes like 'BuGn', etc.

    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140
    )

    # Draw a white circle in the center to create a "donut" with the dark background color
    centre_circle = plt.Circle((0, 0), 0.70, fc='#0e1117')  # Dark background for the center
    fig.gca().add_artist(centre_circle)


    # Set the background color to match Streamlit's dark theme
    fig.patch.set_facecolor('#0e1117')  # Dark background
    ax.set_facecolor('#0e1117')  # Axes background same as figure background

    # Customize text color to make it lighter (white) for contrast against the dark background
    for text in texts + autotexts:
        text.set_color('#ffffff')  # Light text for better readability

    # Return the figure
    return fig


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

def plot_vertical_barchart_from_df(df, x_column, y_column, bar_width=0.7, figsize=(12, 10), font_size=14):
    # Filter out rows where y_column is 0 or NaN
    df = df[df[y_column].fillna(0) != 0]

    if df.empty:
        st.warning("No data to display in the chart.")
        return None

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


col1, col2 = st.columns([ 1, 4]) 

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

st.header("Human Resources", divider="gray")


total_employed = hr_df_filter.shape[0]
average_age = round(hr_df_filter['Age'].mean(),2)
average_experience = round(hr_df_filter['ExperienceInCurrentDomain'].mean(),2)

col1, col2, col3 = st.columns(3)


with col1:
    st.markdown(f"""
        <div style="background-color:#1f77b4;padding:20px;border-radius:10px;text-align:center">
            <h4 style="color:white;">Total Employee Count</h4>
            <h2 style="color:white;">{total_employed}</h2>
        </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
        <div style="background-color:#2ca02c;padding:20px;border-radius:10px;text-align:center">
            <h4 style="color:white;">Average Age</h4>
            <h2 style="color:white;">{average_age}</h2>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
        <div style="background-color:#ff7f0e;padding:20px;border-radius:10px;text-align:center">
            <h4 style="color:white;">Average Work Experience</h4>
            <h2 style="color:white;">{average_experience}</h2>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Employee Diversity", divider="gray")
    city_df_grouped = hr_df_filter.groupby("City", as_index=False).size()
    city_df_grouped.rename(columns={"size": "count"}, inplace=True)
    histo = plot_vertical_barchart_from_df(city_df_grouped, x_column="City", y_column="count")
    st.pyplot(histo)
with col2:
    st.subheader("Gender Diversity", divider="gray")
    histo = donut_chart(hr_df_filter, "Gender")
    st.pyplot(histo)
with col3:
    st.subheader("Education Background", divider="gray")
    histo = donut_chart(hr_df_filter, "Education")
    st.pyplot(histo)