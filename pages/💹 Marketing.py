import pandas as pd
# import redshift_connector
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
# import psycopg2
# import psycopg2.extras

st.set_page_config(layout="wide") 

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

market_df = pd.read_csv('pages/advertising.csv', delimiter=",", on_bad_lines='skip')
market_df['Timestamp'] = pd.to_datetime(market_df['Timestamp'])
market_df['Hour'] = market_df['Timestamp'].dt.hour
market_df["Timestamp"] = pd.to_datetime(market_df["Timestamp"]).dt.date

market_df["Gender"] = market_df["Male"].replace({1: "Male", 0: "Female"})


Country_types = list(market_df["Country"].unique())
genders = list(market_df["Gender"].unique())
min_date = market_df['Timestamp'].min()
max_date = market_df['Timestamp'].max()

st.sidebar.header("Filter Options")
with st.sidebar.expander("Select Countries", expanded=True):
    selected_country = st.multiselect(
        "Choose one or more countries",
        Country_types,
        default=Country_types
    )
selected_gender = st.sidebar.multiselect("Select Genders ", genders, default=genders)
selected_dates = st.sidebar.date_input("Pick a date range", (min_date, max_date))

if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    market_df_selected = market_df[
        market_df["Country"].isin(selected_country) &
        market_df["Gender"].isin(selected_gender) &
        (market_df["Timestamp"] >= start_date) &
        (market_df["Timestamp"] <= end_date)
    ]
else:
    st.warning("Please select a valid date range.")


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

st.header("Marketing", divider="gray")


col = st.columns((2, 4), gap='medium')

with col[0]:
    st.subheader("Metric KPIs", divider="gray")
    total_people = market_df_selected.shape[0]
    total_clicked = market_df_selected[market_df_selected["Clicked on Ad"] == 1].shape[0]
    total_time = market_df_selected['Daily Time Spent on Site'].mean()
    st.metric(label="Total People Visited", value=total_people, delta= 15)
    st.metric(label="Total Directed From Ads", value=total_clicked, delta= 1234)
    st.metric(label="Average Minutes in the Site", value=round(total_time,2), delta= 0.4)
with col[1]:
    st.subheader("Age Distribution by Ad Click", divider="gray")
    fig1, ax1 = plt.subplots(figsize=(10, 4))

    # Apply dark background
    fig1.patch.set_facecolor("#0e1117")
    ax1.set_facecolor("#0e1117")

    # Seaborn plot
    sns.histplot(
        data=market_df_selected,
        x="Age",
        hue="Clicked on Ad",
        multiple="stack",
        palette=["#1f77b4", "#ff7f0e"],  # brighter colors for dark bg
        edgecolor="white",
        ax=ax1,
    )

    # Title and labels
    ax1.set_xlabel("Age", fontsize=14, color="white")
    ax1.set_ylabel("Count", fontsize=14, color="white")

    # Ticks
    ax1.tick_params(axis='both', colors='white', labelsize=12)

    # Remove spines
    for spine in ax1.spines.values():
        spine.set_visible(False)

    st.pyplot(fig1)



col = st.columns((3, 3), gap='medium')

with col[0]:
    st.subheader("Income Distribution by Click Status", divider="gray")

    # Create the plot with dark background
    fig3, ax3 = plt.subplots(figsize=(10, 6))  # Adjust the size for better fit

    # Apply dark background
    fig3.patch.set_facecolor("#0e1117")
    ax3.set_facecolor("#0e1117")

    # Seaborn KDE plot with bright colors for dark background
    sns.kdeplot(
        data=market_df_selected, 
        x="Area Income", 
        hue="Clicked on Ad", 
        fill=True, 
        common_norm=False, 
        palette=["#1f77b4", "#ff7f0e"],  # Brighter colors for dark bg
        ax=ax3
    )

    # Title and labels with white font
    ax3.set_xlabel("Area Income", fontsize=12, color="white")
    ax3.set_ylabel("Density", fontsize=12, color="white")

    # Ticks with white color
    ax3.tick_params(axis='both', colors='white', labelsize=12)

    # Remove spines
    for spine in ax3.spines.values():
        spine.set_visible(False)

    # Display the plot with Streamlit's responsive layout
    st.pyplot(fig3)

with col[1]:
    st.subheader("Visitors Per Hour", divider="gray")
    market_df_grouped = market_df_selected.groupby("Hour", as_index=False).size()
    market_df_grouped.rename(columns={"size": "Count"}, inplace=True)

    line_chart_streamlit_plotly(market_df_grouped, x='Hour', y='Count', title='Clicks Over the Day')

st.subheader("Visitors based on Country", divider="gray")
country_code_mapping = {
    "Tunisia": "TUN",
    "Nauru": "NRU",
    "San Marino": "SMR",
    "Italy": "ITA",
    "Iceland": "ISL",
    "Norway": "NOR",
    "Myanmar": "MMR",
    "Australia": "AUS",
    "Grenada": "GRD",
    "Ghana": "GHA",
    "Qatar": "QAT",
    "Burundi": "BDI",
    "Egypt": "EGY",
    "Bosnia and Herzegovina": "BIH",
    "Barbados": "BRB",
    "Spain": "ESP",
    "Palestinian Territory": "PSE",
    "Afghanistan": "AFG",
    "British Indian Ocean Territory (Chagos Archipelago)": "IOT",
    "Russian Federation": "RUS",
    "Cameroon": "CMR",
    "Korea": "PRK",  # North Korea
    "Tokelau": "TKL",
    "Monaco": "MCO",
    "Tuvalu": "TUV",
    "Greece": "GRC",
    "British Virgin Islands": "VGB",
    "Bouvet Island (Bouvetoya)": "BVT",
    "Peru": "PER",
    "Aruba": "ABW",
    "Maldives": "MDV",
    "Senegal": "SEN",
    "Dominica": "DMA",
    "Luxembourg": "LUX",
    "Montenegro": "MNE",
    "Ukraine": "UKR",
    "Saint Helena": "SHN",
    "Liberia": "LBR",
    "Turkmenistan": "TKM",
    "Niger": "NER",
    "Sri Lanka": "LKA",
    "Trinidad and Tobago": "TTO",
    "United Kingdom": "GBR",
    "Guinea-Bissau": "GNB",
    "Micronesia": "FSM",
    "Turkey": "TUR",
    "Croatia": "HRV",
    "Israel": "ISR",
    "Svalbard & Jan Mayen Islands": "SJM",
    "Azerbaijan": "AZE",
    "Iran": "IRN",
    "Saint Vincent and the Grenadines": "VCT",
    "Bulgaria": "BGR",
    "Christmas Island": "CXR",
    "Canada": "CAN",
    "Rwanda": "RWA",
    "Turks and Caicos Islands": "TCA",
    "Norfolk Island": "NFK",
    "Cook Islands": "COK",
    "Guatemala": "GTM",
    "Cote d'Ivoire": "CIV",
    "Faroe Islands": "FRO",
    "Ireland": "IRL",
    "Moldova": "MDA",
    "Nicaragua": "NIC",
    "Montserrat": "MSR",
    "Timor-Leste": "TLS",
    "Puerto Rico": "PRI",
    "Central African Republic": "CAF",
    "Venezuela": "VEN",
    "Wallis and Futuna": "WLF",
    "Jersey": "JEY",
    "Samoa": "WSM",
    "Antarctica (the territory South of 60 deg S)": "ATA",
    "Albania": "ALB",
    "Hong Kong": "HKG",
    "Lithuania": "LTU",
    "Bangladesh": "BGD",
    "Western Sahara": "ESH",
    "Serbia": "SRB",
    "Czech Republic": "CZE",
    "Guernsey": "GGY",
    "Tanzania": "TZA",
    "Bhutan": "BTN",
    "Guinea": "GIN",
    "Madagascar": "MDG",
    "Lebanon": "LBN",
    "Eritrea": "ERI",
    "Guyana": "GUY",
    "United Arab Emirates": "ARE",
    "Martinique": "MTQ",
    "Somalia": "SOM",
    "Benin": "BEN",
    "Papua New Guinea": "PNG",
    "Uzbekistan": "UZB",
    "South Africa": "ZAF",
    "Hungary": "HUN",
    "Falkland Islands (Malvinas)": "FLK",
    "Saint Martin": "MAF",
    "Cuba": "CUB",
    "United States Minor Outlying Islands": "UMI",
    "Belize": "BLZ",
    "Kuwait": "KWT",
    "Thailand": "THA",
    "Gibraltar": "GIB",
    "Holy See (Vatican City State)": "VAT",
    "Netherlands": "NLD",
    "Belarus": "BLR",
    "New Zealand": "NZL",
    "Togo": "TGO",
    "Kenya": "KEN",
    "Palau": "PLW",
    "Cambodia": "KHM",
    "Costa Rica": "CRI",
    "Liechtenstein": "LIE",
    "Angola": "AGO",
    "Equatorial Guinea": "GNQ",
    "Mongolia": "MNG",
    "Brazil": "BRA",
    "Chad": "TCD",
    "Portugal": "PRT",
    "Malawi": "MWI",
    "Singapore": "SGP",
    "Kazakhstan": "KAZ",
    "China": "CHN",
    "Vietnam": "VNM",
    "Mayotte": "MYT",
    "Jamaica": "JAM",
    "Bahamas": "BHS",
    "Algeria": "DZA",
    "Fiji": "FJI",
    "Argentina": "ARG",
    "Philippines": "PHL",
    "Suriname": "SUR",
    "Guam": "GUM",
    "Antigua and Barbuda": "ATG",
    "Georgia": "GEO",
    "Jordan": "JOR",
    "Saudi Arabia": "SAU",
    "Sao Tome and Principe": "STP",
    "Cyprus": "CYP",
    "Kyrgyz Republic": "KGZ",
    "Pakistan": "PAK",
    "Seychelles": "SYC",
    "Mauritania": "MRT",
    "Chile": "CHL",
    "Poland": "POL",
    "Estonia": "EST",
    "Latvia": "LVA",
    "Bahrain": "BHR",
    "Colombia": "COL",
    "Brunei Darussalam": "BRN",
    "Taiwan": "TWN",
    "Saint Pierre and Miquelon": "SPM",
    "Finland": "FIN",
    "French Southern Territories": "ATF",
    "Sierra Leone": "SLE",
    "Tajikistan": "TJK"}

country_grouped = market_df_selected.groupby("Country", as_index=False).size()
country_grouped.rename(columns={"size": "Count"}, inplace=True)

country_grouped["Country"] = country_grouped["Country"].replace(country_code_mapping)
country_grouped = country_grouped.dropna(subset=["Country"])

max = country_grouped["Count"].max()
min = country_grouped["Count"].min()
print(country_grouped)
fig = px.choropleth(country_grouped,locations="Country",
                color='Count',
                hover_name="Country",
                range_color=(min,max),
                projection="natural earth",
                color_continuous_scale=px.colors.sequential.Reds)
fig.update_layout(

autosize=False,  # Automatically adjust the size based on container
height=800,     # You can adjust the height here as needed
width=1400       # You can adjust the width here as needed
)
st.plotly_chart(fig)
