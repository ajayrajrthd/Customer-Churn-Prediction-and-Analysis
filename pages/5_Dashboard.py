import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from streamlit_metrics import metric, metric_row
import pygal
import leather
import plotly.express as px


# Redirect if not logged in
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("You must login first to access this page.")
    st.stop()  # Stops execution here

# Load the dataset
dataset_path = 'dataset/Churn Prediction Dataset.csv'
df = pd.read_csv(dataset_path)

# Convert 'TotalCharges' column to numerical values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Set page title
st.set_page_config(page_title="Visualization Dashboard")

# Title for the page
st.title("Visualization Dashboard")

# Sidebar navigation
option = st.sidebar.selectbox(
    'Select:',
    ('Analytics Dashboard', 'Sales Trend Analysis', 'Key Performance Indicators')
)

if option == 'Analytics Dashboard':
    # Research question 1: Distribution of churn for different Internet service types
    st.header("Research question 1: Distribution of churn for different Internet service types")

    # Using Plotly Express
    fig = px.bar(df, x='InternetService', color='Churn', barmode='group',
                title='Churn Distribution for Internet Service Types (Plotly Express)',
                category_orders={'InternetService': ['DSL', 'Fiber optic', 'No']},
                color_discrete_map={'No': 'lightgreen', 'Yes': 'yellow'})
    fig.update_xaxes(title="Internet Service Type")
    fig.update_yaxes(title="Count")
    st.plotly_chart(fig)

    # Research question 2: Impact of having a partner or dependents on customer churn
    st.header("Research question 2: Impact of having a partner or dependents on customer churn")

    # Using Altair
    partner_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Partner:O', title='Partner Status'),
        y=alt.Y('count():Q', title='Count'),
        color='Churn:N'
    ).properties(
        title="Churn Distribution for Partner Status (Altair)"
    )
    st.altair_chart(partner_chart, use_container_width=True)

    dependents_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Dependents:O', title='Dependents Status'),
        y=alt.Y('count():Q', title='Count'),
        color='Churn:N'
    ).properties(
        title="Churn Distribution for Dependents Status (Altair)"
    )
    st.altair_chart(dependents_chart, use_container_width=True)

    # Research question 3: Influence of contract type on customer churn
    st.header("Research question 3: Influence of contract type on customer churn")

    # Using Plotly Express
    fig2 = px.histogram(df, x='Contract', color='Churn', barmode='group')
    fig2.update_layout(title="Churn Distribution for Contract Type (Plotly Express)")
    st.plotly_chart(fig2, use_container_width=True)

    # Research question 4: Impact of billing preference on customer churn
    st.header("Research question 4: Impact of billing preference on customer churn")

    # Convert 'Churn' column to boolean (0 for No, 1 for Yes)
    df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

    # Group data by Billing Preference and calculate churn
    billing_churn = df.groupby('PaperlessBilling')['Churn'].sum().reset_index()

    # Plot using Plotly Express
    fig = px.bar(billing_churn, x='PaperlessBilling', y='Churn', 
                labels={'PaperlessBilling': 'Billing Preference', 'Churn': 'Churn Count'},
                title='Churn Distribution for Billing Preference (Plotly Express)')
    st.plotly_chart(fig)



    # Using Altair
    gender_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('gender', title='Gender'),
        y=alt.Y('count()', title='Count'),
        color='Churn:N'
    ).properties(
        title="Churn Distribution by Gender (Altair)"
    )
    st.altair_chart(gender_chart, use_container_width=True)

    # Additional research questions
    st.header("Additional Research Questions")

    # Research question 6: Impact of tenure on customer churn
    st.header("Research question 6: Impact of tenure on customer churn")

    # Plot using Plotly Express
    fig = px.histogram(df, x='tenure', color='Churn', nbins=20,
                    labels={'tenure': 'Tenure', 'Churn': 'Churn'},
                    title='Impact of Tenure on Customer Churn')
    st.plotly_chart(fig)

    # Research question 7: Relationship between total charges and churn
    st.subheader("Research question 7: Relationship between total charges and churn")
    charges_churn_scatter = alt.Chart(df).mark_circle(size=60).encode(
        x='TotalCharges',
        y='Churn',
        color='Churn:N',
        tooltip=['TotalCharges', 'Churn']
    ).properties(
        title="Churn vs Total Charges (Altair)"
    ).interactive()
    st.altair_chart(charges_churn_scatter, use_container_width=True)


elif option == 'Sales Trend Analysis':
    st.header("Sales Trend Analysis (Approximate)")

    if 'tenure' in df.columns and 'MonthlyCharges' in df.columns:
        df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})
        # Approximate each customer's start month (assume tenure in months)
        df['StartMonth'] = pd.to_datetime('2025-01-01') - pd.to_timedelta(df['tenure']*30, unit='d')

        # Create a monthly revenue DataFrame
        revenue_data = []

        for i, row in df.iterrows():
            start_month = row['StartMonth']
            for month_offset in range(int(row['tenure'])):
                month = start_month + pd.DateOffset(months=month_offset)
                revenue_data.append({'Month': month.to_period('M').to_timestamp(), 
                                     'MonthlyRevenue': row['MonthlyCharges'], 
                                     'Churn': row['Churn']})

        revenue_df = pd.DataFrame(revenue_data)

        # Group by month to calculate total revenue and churn count
        monthly_summary = revenue_df.groupby('Month').agg(
            TotalRevenue=('MonthlyRevenue', 'sum'),
            ChurnedCustomers=('Churn', 'sum')
        ).reset_index()

        # Plot monthly revenue
        st.subheader("Monthly Revenue")
        fig = px.line(monthly_summary, x='Month', y='TotalRevenue', 
                      title='Approximate Monthly Revenue')
        st.plotly_chart(fig, use_container_width=True)

        # Plot churn impact
        st.subheader("Monthly Churn Impact")
        fig2 = px.bar(monthly_summary, x='Month', y='ChurnedCustomers', 
                      title='Number of Churned Customers per Month', 
                      labels={'ChurnedCustomers': 'Churned Customers'})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Dataset does not have 'tenure' and/or 'MonthlyCharges' columns for sales analysis.")

    
elif option == 'Key Performance Indicators for Churn Prediction':
    # Key Performance Indicators (KPIs)
    st.header("Key Performance Indicators (KPIs)")

    # Example values (replace with actual calculations)
    gross_mrr_churn = 0.05
    net_mrr_churn = 0.03
    net_change_customers = 100
    revenue_growth_rate = 0.10
    activation_rate = 0.75
    dau_mau_ratio = 0.65
    nps = 75
    csat = 85
    clv = 1500

    # Function to create colored KPI cards
    def metric_card(title, value, color="#000000"):
        st.markdown(
            f"""
            <div style="
                background-color: black;
                border-radius: 10px;
                padding: 15px;
                margin-bottom: 10px;
                box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
            ">
                <h4 style='margin:0'>{title}</h4>
                <p style='font-size:22px; color:{color}; font-weight:bold; margin:5px 0 0 0'>{value}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Layout: two columns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Financial Metrics")
        metric_card("Gross MRR Churn", gross_mrr_churn, "green")
        metric_card("Net MRR Churn", net_mrr_churn, "orange")
        metric_card("Net Change in Customers", net_change_customers, "blue")
        metric_card("Revenue Growth Rate", revenue_growth_rate, "purple")

    with col2:
        st.subheader("Product Metrics")
        metric_card("Activation Rate", activation_rate, "purple")
        metric_card("DAU/MAU Ratio", dau_mau_ratio, "orange")
        metric_card("Net Promoter Score (NPS)", nps, "green")
        metric_card("Customer Satisfaction (CSAT)", csat, "yellow")
        metric_card("Customer Lifetime Value (LTV)", clv, "blue")
