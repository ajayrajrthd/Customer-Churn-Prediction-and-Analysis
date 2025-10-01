import pandas as pd
import streamlit as st
import os

# Get the current working directory
current_dir = os.getcwd()

# Relative path to the dataset within the "dataset" folder
dataset_path = os.path.join(current_dir, 'dataset', 'Churn Prediction Dataset.csv')

# Load the dataset
df = pd.read_csv(dataset_path)

st.title("Data Analysis")

st.header("Basic Information about the Dataset")

# Numeric info first
basic_info_numeric = pd.DataFrame({
    "Property": ["Number of Rows", "Number of Columns"],
    "Value": [len(df), len(df.columns)]
})
st.table(basic_info_numeric)  # safe with st.table

# Column names separately
st.header("Column Names")
columns_table = pd.DataFrame({
    "All Columns": [", ".join(df.columns)]
})
st.dataframe(columns_table, hide_index=True)


# --- Summary Statistics ---
st.header("Summary Statistics of Numerical Variables")
summary_table = df.describe().T.reset_index().rename(columns={'index': 'Variable'})
st.dataframe(summary_table, width=1000)

# --- First Few Rows ---
st.header("First Few Rows of the Dataset")
st.dataframe(df.head(), width=1000)

# --- Missing Values ---
st.header("Missing Values")
missing_values = pd.DataFrame({
    "Column": df.columns,
    "Missing Values": df.isnull().sum()
})
st.table(missing_values)

# --- Duplicate Rows ---
st.header("Duplicate Rows")
duplicate_count = df.duplicated().sum()
st.write(f"Number of duplicate rows: {duplicate_count}")

# --- Univariate Analysis ---
st.header("Univariate Analysis - Churn Distribution")
churn_distribution = pd.DataFrame(df['Churn'].value_counts()).reset_index()
churn_distribution.columns = ["Churn", "Count"]
st.table(churn_distribution)

# --- Bivariate Analysis ---
st.header("Bivariate Analysis - Monthly Charges vs Churn")
bivariate_table = df.groupby('Churn')['MonthlyCharges'].mean().reset_index()
bivariate_table.columns = ["Churn", "Average Monthly Charges"]
st.table(bivariate_table)
