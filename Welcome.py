# import pandas as pd
# import redshift_connector
# import seaborn as sns
# import matplotlib.pyplot as plt
# import streamlit as st
# import pathlib
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


# Load data once when the app starts

import streamlit as st
import pathlib
st.set_page_config(layout="wide") 
def load_css(file_path):
    with open(file_path) as f:
        st.html(f"<style>{f.read()}</style>")

css_path = pathlib.Path("assets/styles.css")
load_css(css_path)

# Header layout
col1, col2 = st.columns([1, 3])
with col1:
    st.image("pages/diginergyconsulting_logo.jpeg")

with col2:
    st.markdown(
        "<h1 style='margin-top: 25px; font-size: 60px;'>Diginergy</h1>",
        unsafe_allow_html=True
    )

st.markdown(
        "<br><br>",
        unsafe_allow_html=True)

st.markdown("""<hr style="border: 1px solid #6c6c6c;">""", unsafe_allow_html=True)

# Welcome Box
st.markdown("""
<div style='padding: 30px; background-color: #1f1f28; border-radius: 12px; color: white; font-size: 17px;'>
    <h3 style='color: rgb(245, 189, 230);'>👋 Hello, Diginergy Admin!</h3>
    <p>Welcome to your <strong>KPI House</strong> dashboard, where key performance metrics for your departments are just a click away.</p>
    <p><strong>📌 To get started:</strong></p>
    <ul style='line-height: 1.7;'>
        <li>Use the sidebar on the left to switch between <strong>Inventory</strong>, <strong>Labor</strong>, <strong>Sales</strong>, or <strong>Accounting</strong>.</li>
        <li>Each section contains visuals, metrics, and trend analysis curated for that business function.</li>
        <li>All your insights update automatically from the backend data source.</li>
    </ul>
    <p style='margin-top: 20px;'>Need help? Contact the data team at <a href="mailto:support@diginergy.com">support@diginergy.com</a>.</p>
</div>
""", unsafe_allow_html=True)

# Optional: Tips Section
with st.expander("💡 Dashboard Tips"):
    st.markdown("""

    - 📊 Hover over charts for exact values.
    - 📥 Export any section using the download buttons (if available).
    - 🌓 Use Dark Mode for better contrast (already active).
    """)

