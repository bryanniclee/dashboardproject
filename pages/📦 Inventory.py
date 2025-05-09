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
import numpy as np
# Function to fetch data from Redshift
# def fetch_data():
#     conn = redshift_connector.connect(
#         host="daas-test.650251725395.eu-central-1.redshift-serverless.amazonaws.com",
#         database="dev",
#         port=5439,
#         user="admin",
#         password="$St123456"
#     )
#     cursor = conn.cursor()
#     cursor.execute("SELECT * FROM public.transactions LIMIT 100")  # Limit rows for performance
#     rows = cursor.fetchall()
#     columns = [desc[0] for desc in cursor.description]
#     df = pd.DataFrame(rows, columns=columns)
#     cursor.close()
#     conn.close()
#     return df
st.set_page_config(layout="wide") 
# dbhost = "daastest1.c7aawgim4aqb.eu-central-1.rds.amazonaws.com"
# dbname = "transactiontest"
# dbuser = "postgres"
# dbpass = "$St770877"

# def append_receipts_to_postgres(df, dbhost2, dbname2, dbuser2, dbpass2):
#     try:
#         connection = psycopg2.connect(
#             host=dbhost2,
#             database=dbname2,
#             user=dbuser2,
#             password=dbpass2
#         )
#         cursor = connection.cursor()

#         # Quote column names to avoid issues with spaces or special characters
#         columns = ', '.join([f'"{col}"' for col in df.columns])
#         insert_query = f"INSERT INTO valuationdemo ({columns}) VALUES %s"

#         # Convert DataFrame rows to list of tuples
#         data = [tuple(x) for x in df.to_numpy()]

#         # Efficient bulk insert
#         psycopg2.extras.execute_values(cursor, insert_query, data)

#         connection.commit()
#         st.success(f"{len(data)} rows appended to valuationdemo.")

#     except Exception as e:
#         st.error(f"Error appending data: {e}")
#     finally:
#         if 'cursor' in locals():
#             cursor.close()
#         if 'connection' in locals():
#             connection.close()
# EVALUATIONS

filenames = glob('pages/Inventory_Valuation*.csv')
print(filenames)

column_rename_map_reversed = {
    'Item': 'itemnumber',
    'Description': 'description',
    'Rotating': 'rotating',
    'Item Status': 'itemstatus',
    'Quality inspection required?': 'qualityinspectionrequired',
    'Safety-related field': 'safetyrelatedfield',
    'Type': 'type',
    'Order Unit': 'orderunit',
    'Issue Unit': 'issueunit',
    'Date of last issue': 'dateoflastissue',
    'Shelflife': 'shelflife',
    'Locomotive Range': 'locomotiverange',
    'Commodity': 'commodity',
    'Commodity Group': 'commoditygroup',
    'Storeroom': 'storeroom',
    'Storeroom description': 'storeroomdescription',
    'Legal entity of the storeroom': 'legalentityofthestoreroom',
    'Sage company': 'sagecompany',
    'Street number': 'streetnumber',
    'Street name': 'streetname',
    'Postal Code': 'postalcode',
    'City': 'city',
    'Country': 'country',
    'Avg. Price': 'avgprice',
    'Last Price': 'lastprice',
    'Last vendor': 'lastvendor',
    'Vendor': 'vendor',
    'Asset': 'asset',
    'Asset inventory cost': 'assetinventorycost',
    'Asset status': 'assetstatus',
    'Asset condition code': 'assetconditioncode',
    'End of life (EOL)': 'endoflifeeol',
    'Balance': 'balance',
    'Stock Value (Fifo)': 'stockvaluefifo',
    'Received quantity': 'receivedquantity',
    'Received value': 'receivedvalue',
    'Shipped quantity': 'shippedquantity',
    'Shipped value': 'shippedvalue',
    'Staged quantity': 'stagedquantity',
    'Staged value': 'stagedvalue',
    'Total quantity': 'totalquantity',
    'Total value': 'totalvalue'
}

columns_to_fix = [
    'avgprice', 'lastprice', 'assetinventorycost', 'stockvaluefifo',
    'receivedvalue', 'shippedvalue', 'stagedvalue', 'totalvalue', 'shippedquantity', 'balance'
]

dataframes = []

for fname in filenames:
    # full_path = os.path.join('pages/', fname)
    full_path = fname
    # Match date in filename: 8-digit YYYYMMDD
    match = re.search(r'(\d{8})', fname)
    if match:
        date_str = match.group(1)
        try:
            file_date = datetime.strptime(date_str, '%Y%m%d').date()
        except ValueError as e:
            print(f"Error parsing date from {fname}: {e}")
            continue

        # Load and clean
        df = pd.read_csv(full_path, encoding="latin1", delimiter=";", on_bad_lines='skip')
        df.columns = df.columns.str.replace('ï»¿', '', regex=False)
        df.rename(columns=column_rename_map_reversed, inplace=True)
        for col in columns_to_fix:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace(',', '.', regex=False)
                    .str.replace(' ', '', regex=False)
                    .replace('', '0')
                    .astype(float)
                )
                

        df['date'] = file_date

        dataframes.append(df)
    else:
        print(f"Skipping: {fname} - no valid date found in filename")

# Combine all
if dataframes:
    evaluations_df = pd.concat(dataframes, ignore_index=True)
    evaluations_df = evaluations_df.dropna(subset=['itemnumber'])
else:
    print("⚠️ No files processed.")





# def load_data():
#     connection = psycopg2.connect(host=dbhost, database=dbname, user=dbuser, password=dbpass)
#     cursor = connection.cursor()
#     tables = ['transactiondemo', 'valuationdemo']
#     dataframes = {}

#     for table in tables:
#         cursor.execute(f"SELECT * FROM {table} LIMIT 4000;")
#         columns = [desc[0] for desc in cursor.description]
#         rows = cursor.fetchall()
#         df = pd.DataFrame(rows, columns=columns)
#         dataframes[table] = df

#     cursor.close()
#     connection.close()
#     return dataframes

# dataframes = load_data()
# transaction_df = dataframes['transactiondemo'].copy()
csv_files = glob("pages/Inventory_Transactions*.csv")
print(csv_files)
transaction_df = pd.concat(
    [pd.read_csv(file, delimiter=';') for file in csv_files],
    ignore_index=True
)
transaction_df['Date'] = pd.to_datetime(transaction_df['Date'], errors='coerce', dayfirst=True)


transaction_df['Item'] = transaction_df['Item'].apply(
    lambda x: x.replace('ITM', 'ITEM') if isinstance(x, str) and 'ITM' in x else x
)
transaction_df['Item'] = transaction_df['Item'].apply(
    lambda x: f"{x}_ITEM" if isinstance(x, str) and x.isdigit() else x
)

transaction_df['Date'] = pd.to_datetime(transaction_df['Date'], format='%Y-%m-%d', errors='coerce').dt.date
transaction_df = transaction_df.dropna(subset=['Date'])

# EVALUATIONS PREPROCESS

# evaluations_df = dataframes['valuationdemo'].copy()
evaluations_df['date'] = pd.to_datetime(
    evaluations_df['date'],
    format='%Y-%m-%d',
    errors='coerce'
).dt.date

evaluations_df = evaluations_df.dropna(subset=['date'])
# evaluations_df['itemnumber'] = evaluations_df['itemnumber'].apply(lambda x: x.replace('ITM', 'ITEM') if 'ITM' in x else x)
# evaluations_df['itemnumber'] = evaluations_df['itemnumber'].apply(lambda x: f"{x}_ITEM" if x.isdigit() else x)

# evaluations_df["type"] = evaluations_df["type"].fillna("Unknown")
# evaluations_df["country"] = evaluations_df["country"].fillna("Unknown")

# if st.button("Home"):
#     st.switch_page("kpi_housee.py")
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

st.header("Inventory", divider="gray")

# Move filters to the sidebar
st.sidebar.header("Filter Options")
country_name_mapping = {
    "Deutschland": "Germany",
    "Italia": "Italy",
    "Polen": "Poland",
    "Unknown": None  # Optional: drop unknowns later
    }
evaluations_df["country"] = evaluations_df["country"].replace(country_name_mapping)
item_types = list(evaluations_df["type"].unique())
countries = list(evaluations_df["country"].unique())
min_date = evaluations_df['date'].min()
max_date = evaluations_df['date'].max()

selected_types = st.sidebar.multiselect("Select transaction types", item_types, default=item_types)
selected_countries = st.sidebar.multiselect("Select countries", countries, default=countries)
selected_dates = st.sidebar.date_input("Pick a date range", (min_date, max_date))

# Filter the DataFrame
if len(selected_dates) == 2:
    start_date, end_date = selected_dates
    df_selected = evaluations_df[
        evaluations_df["type"].isin(selected_types) &
        evaluations_df["country"].isin(selected_countries) &
        (evaluations_df["date"] >= start_date) &
        (evaluations_df["date"] <= end_date)
    ]
else:
    st.warning("Please select a valid date range.")
evaluations_df_sorted = df_selected.sort_values(by="type", ascending=False)


def histplot(df, column, bins=30):
    fig, ax = plt.subplots(figsize=(8, 6))  # Set size to match the pie chart
    sns.histplot(data=df, x=column, bins=bins, ax=ax, element="bars")
    ax.set_title("Histogram")
    ax.set_xlabel(column)
    ax.set_ylabel("Count")
    
    plt.xticks(rotation=45)

    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.text(p.get_x() + p.get_width() / 2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom')
    
    return fig


def line_chart_streamlit_plotly(data, x, y, title="Line Chart", x_label=None, y_label=None):

    fig = px.line(data, x=x, y=y)
    fig.update_layout(
        xaxis_title=x_label if x_label else x,
        yaxis_title=y_label if y_label else y,
        template="plotly_white",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)
    

def plot_barchart_from_df(df, x_column, y_column, bar_height=0.6, figsize=(12, 12), font_size=14):
    # Filter out rows where y_column is 0 or NaN
    df = df[df[y_column].fillna(0) != 0]

    # Ensure there's data left to plot
    if df.empty:
        st.warning("No data to display in the chart.")
        return None

    # Compute max value
    max_val = df[y_column].max()

    df_sorted = df.sort_values(by=y_column, ascending=True)
    fig, ax = plt.subplots(figsize=figsize)

    # Set dark background
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    xdata = df_sorted[x_column].astype(str)
    ydata = df_sorted[y_column]

    bars = ax.barh(xdata, ydata, height=bar_height, color="crimson")

    for bar in bars:
        width = bar.get_width()
        label = f"{width:,.0f}"

        if width > max_val * 0.1:
            ax.text(width - max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                    label, va='center', ha='right', color='white', fontsize=font_size-2)
        else:
            ax.text(width + max_val * 0.01, bar.get_y() + bar.get_height() / 2,
                    label, va='center', ha='left', color='white', fontsize=font_size-2)

    # Set all text to white
    ax.set_xlabel(y_column, fontsize=font_size, color='white')
    ax.set_ylabel(x_column, fontsize=font_size, color='white')

    ax.tick_params(axis='both', labelsize=font_size - 2, colors='white')

    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig

def plot_treemap(df, label_col, value_col, title="Treemap", figsize=(12, 8)):
    fig, ax = plt.subplots(figsize=figsize)

    # Sort data by value for better layout
    df_sorted = df.sort_values(by=value_col, ascending=False)

    sizes = df_sorted[value_col]
    labels = [
        f"{label}\n{value:,.0f}" for label, value in zip(df_sorted[label_col], sizes)
    ]

    squarify.plot(
        sizes=sizes,
        label=labels,
        alpha=0.8,
        color=plt.cm.tab20.colors,
        text_kwargs={'fontsize': 10}
    )

    ax.set_title(title, fontsize=16)
    ax.axis('off')

    return fig

def pie_chart(df, column):
    fig, ax = plt.subplots(figsize=(8, 6))  # Set the same size as the histogram
    data = df[column].value_counts()
    labels = data.index
    sizes = data.values
    colors = sns.color_palette('pastel')[0:len(labels)]
    
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
    ax.set_title(f'Pie Chart of {column}')
    
    return fig

def donut_chart(df, column):
    fig, ax = plt.subplots(figsize=(8, 6))  # Same size as the histogram
    data = df[column].value_counts()
    labels = data.index
    sizes = data.values
    colors = sns.color_palette('pastel')[0:len(labels)]

    # Create the pie chart
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140
    )

    # Draw a white circle in the center to create a "donut"
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    ax.set_title(f'Donut Chart of {column}')
    return fig
restock_df = evaluations_df_sorted[evaluations_df_sorted['balance'] == 0]
col = st.columns((2, 3, 3), gap='medium')


with col[0]:
  st.subheader("Metric KPIs", divider="gray")
  item_shift = float((-1 * evaluations_df_sorted['shippedquantity']).sum())
  total_balance = int(evaluations_df_sorted['balance'].sum())
  quantity_shift = int(item_shift)
  st.metric(label="Total Item Balance", value=f"€ {total_balance}", delta=f"{quantity_shift:+}")
  st.metric(label="Quantity Shift", value=quantity_shift, delta=f"{quantity_shift:+}")
  total_values = evaluations_df_sorted['totalvalue'].sum()
  st.metric(label="Total Value", value=f"€ {total_values:,.2f}", delta="+120,423")
  st.metric(label="Items With 0 Stock", value=restock_df.shape[0], delta="+12")


with col[1]:
    st.subheader("Inventory Stock Over Time", divider="gray")
    # st.bar_chart(evaluations_df_sorted, x="type", y="balance", stack=False, height= 600, horizontal = True)
    # df_grouped = evaluations_df_sorted.groupby("type", as_index=False)["balance"].sum()
    # histo = plot_barchart_from_df(df_grouped, x_column="type", y_column="balance")
    df_grouped = evaluations_df_sorted.groupby("date", as_index=False)["balance"].sum()
    line_chart_streamlit_plotly(df_grouped, x='date', y='balance', title='Balance Over Time')
#   st.pyplot(histo)

with col[2]:
#   pie = donut_chart(evaluations_df_sorted,"country")
#   st.pyplot(pie)
    st.subheader("Quantity Stock Grouped by Item Type", divider="gray")
    df_grouped = evaluations_df_sorted.groupby("type", as_index=False)["balance"].sum()
    histo = plot_barchart_from_df(df_grouped, x_column="type", y_column="balance")
    st.pyplot(histo)
    # top_5_df = evaluations_df_sorted.sort_values(by='balance', ascending=False).head(35)
    # st.subheader("Items with Largest Stock", divider="gray")
    # st.dataframe(
    #     top_5_df,
    #     column_order=("itemnumber", "balance"),
    #     hide_index=True,
    #     column_config={
    #         "itemnumber": st.column_config.TextColumn("Item Number"),
    #         "balance": st.column_config.ProgressColumn(
    #             "Balance",
    #             format="%.2f",
    #             min_value=0,
    #             max_value=float(evaluations_df_sorted['balance'].max()),  # 🔧 cast to float
    #         ),
    #     }
    # )

col = st.columns((5, 3), gap='medium')

with col[0]:
    st.subheader("Item Quantity based on Country", divider="gray")

    country_code_mapping = {
    "Denmark": "DNK",
    "France": "FRA",
    "Germany": "DEU",
    "Hungary": "HUN",
    "Italy": "ITA",
    "Norway": "NOR",
    "Poland": "POL",
    "Sweden": "SWE",
    "Switzerland": "CHE",
    "Deutschland": "DEU",  # Germany's alternate name
    "Italia": "ITA",       # Italy's alternate name
    "Polen": "POL",        # Poland's alternate name
    "Unknown": None        # Remove rows with 'Unknown'
}

    # Replace and clean up
    country_grouped = evaluations_df_sorted.groupby("country", as_index=False)["balance"].sum()
    country_grouped["country"] = country_grouped["country"].replace(country_code_mapping)
    country_grouped = country_grouped.dropna(subset=["country"])

    max = country_grouped["balance"].max()
    min = country_grouped["balance"].min()
    print(country_grouped)
    fig = px.choropleth(country_grouped,locations="country",
                    color='balance',
                    hover_name="country",
                    range_color=(min,max),
                    scope= 'europe',
                    projection="natural earth",
                    color_continuous_scale=px.colors.sequential.Reds)
    fig.update_layout(
    
    autosize=False,  # Automatically adjust the size based on container
    height=600,     # You can adjust the height here as needed
    width=900       # You can adjust the width here as needed
    )
    st.plotly_chart(fig)

with col[1]:
    st.subheader("Items in need of Restocking!", divider="gray")
    st.dataframe(
    restock_df,
    column_order=("itemnumber", "description"),
    hide_index=True,
    column_config={
        "itemnumber": st.column_config.TextColumn("Item Number"),
        "description": st.column_config.TextColumn("Full Description")
    },
    height=550  # adjust this number (pixels) to your preferred height
    )

st.header("Transactions Today", divider="gray")\

Titem_types = list(transaction_df["ItemType"].unique())
Tmin_date = transaction_df['Date'].min()
Tmax_date = transaction_df['Date'].max()

Tselected_types = st.multiselect("Select transaction types", Titem_types, default=Titem_types)
Tselected_dates = st.date_input("Pick a date range", (Tmin_date, Tmax_date), key="12")

if len(Tselected_dates) == 2:
    tstart_date, tend_date = Tselected_dates
    Tdf_selected = transaction_df[
        transaction_df["ItemType"].isin(Tselected_types) &
        (transaction_df["Date"] >= tstart_date) &
        (transaction_df["Date"] <= tend_date)
    ]
else:
    st.warning("Please select a valid date range.")



Tdf_selected['ActualCost'] = pd.to_numeric(Tdf_selected['ActualCost'], errors='coerce')
Tdf_selected['Qty'] = pd.to_numeric(Tdf_selected['Qty'], errors='coerce')
col = st.columns((2, 3, 3), gap='medium')

with col[0]:
    st.subheader("Metric KPIs", divider="gray")
    st.metric(label="Total Transactions for today", value= Tdf_selected.shape[0 ], delta= 12) 
    st.metric(label="Total Cost", value= f"€ {float(Tdf_selected['ActualCost'].sum())}", delta="-323")
    st.metric(label="Quantity Shift", value=int(Tdf_selected['Qty'].sum()), delta="-122")

with col[1]:
    st.subheader("Quantity Shift Grouped by Item Type", divider="gray")
    Tdf_grouped = Tdf_selected.groupby("TransType", as_index=False)["Qty"].sum()
    Thisto = plot_barchart_from_df(Tdf_grouped, x_column="TransType", y_column="Qty")
    st.pyplot(Thisto)

with col[2]:
    st.subheader("Most Lost Quantity", divider="gray")
    inverse_T_df = Tdf_selected.groupby("Item", as_index=False)["Qty"].sum()
    inverse_T_df['Qty'] = -inverse_T_df['Qty']
    Tdfgroup = inverse_T_df
    Ttop_5_df = Tdfgroup.sort_values(by='Qty', ascending=False).head(35)
    st.dataframe(
        Ttop_5_df,
        column_order=("Item", "Qty"),
        hide_index=True,
        column_config={
            "Item": st.column_config.TextColumn("Item Number"),
            "Qty": st.column_config.ProgressColumn(
                "Quantity Shift",
                format="%.2f",
                min_value=0,
                max_value=float(Tdfgroup['Qty'].max()),  # 🔧 cast to float
            ),
        }
    )

col = st.columns((4, 4), gap='medium')

with col[0]:
    st.subheader("Item Shift Over Time", divider="gray")
    trans_group = Tdf_selected.groupby("Date", as_index=False)["Qty"].sum()
    line_chart_streamlit_plotly(trans_group, x='Date', y='Qty')

with col[1]:
    st.subheader("Transactions Over Time", divider="gray")
    trans_group = Tdf_selected.groupby("Date").size().reset_index(name="Count")
    line_chart_streamlit_plotly(trans_group, x='Date', y='Count')


col = st.columns((4, 4), gap='medium')

with col[0]:
    st.subheader("Inventory and Transaction Difference Over Time", divider="gray")
    trans_group = Tdf_selected.groupby("Date", as_index=False)["Qty"].sum()
    trans_group["Date"] = pd.to_datetime(trans_group["Date"])

    # Step 2: Group balance sum by date from evaluations
    df_grouped = evaluations_df_sorted.groupby("date", as_index=False)["balance"].sum()
    df_grouped["date"] = pd.to_datetime(df_grouped["date"])
    df_grouped = df_grouped.sort_values("date")

    # Step 3: Calculate daily change in balance
    df_grouped["balance_change"] = df_grouped["balance"].diff()

    # Step 4: Merge with transactions on the same date
    merged = pd.merge(trans_group, df_grouped, left_on="Date", right_on="date", how="inner")

    # Step 5: Calculate difference between Qty and balance change
    merged["difference"] = merged["Qty"] - merged["balance_change"]

    # Step 6: Plot
    line_chart_streamlit_plotly(merged, x="Date", y="difference")


transaction_df = transaction_df.loc[:, ~transaction_df.columns.str.contains('Unnamed: 24')]

for col in ['balance', 'totalquantity']:
    evaluations_df[col] = (
        evaluations_df[col]    # Select the column
        .astype(str)           # Convert to string (if not already)
        .str.replace(',', '.', regex=False)  # Handle European commas
        .astype(float)         # Convert to float (just in case it's like '2.0')
        .round(0)              # Round values to the nearest integer
        .astype(int)           # Convert to integer
        .fillna(0)             # Fill missing values with 0
    )

for col in ['avgprice', 'lastprice', 'stockvaluefifo', 'receivedquantity', 'receivedvalue', 'shippedquantity', 'shippedvalue', 'stagedquantity', 'stagedvalue', 'totalvalue']:
    evaluations_df[col] = (
        evaluations_df[col]
        .astype(str)
        .str.replace(',', '.', regex=False)
        .astype(float)
        .fillna(0.0)
)

evaluations_df.columns = evaluations_df.columns.str.strip()

# append_receipts_to_postgres(df, dbhost2, dbname2, dbuser2, dbpass2)

print("test succesful")
evaluations_df.to_csv('evals_full.csv', sep = ';',index=False)


transaction_df["TransType"] = transaction_df["TransType"].replace("NaN", np.nan)
transaction_df["ItemType"] = transaction_df["ItemType"].replace("NaN", np.nan)

# Then fill the real NaNs with "Unknown"
transaction_df["TransType"] = transaction_df["TransType"].fillna("Unknown")
transaction_df["ItemType"] = transaction_df["ItemType"].fillna("Unknown")

transaction_df.to_csv('transaction_full.csv', sep = ';', index=False)
