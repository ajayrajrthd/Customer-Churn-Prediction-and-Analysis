import streamlit as st
import pandas as pd
import os
import joblib
import pickle
import numpy as np
import warnings
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow import keras
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

warnings.filterwarnings("ignore")


# Redirect if not logged in
if 'logged_in' not in st.session_state or not st.session_state['logged_in']:
    st.warning("You must login first to access this page.")
    st.stop()  # Stops execution here

# ------------------ Load existing models ------------------
models_dir = 'models'
dt_model_path = os.path.join(models_dir, 'dt_model.pkl')
rf_model_path = os.path.join(models_dir, 'rf_model.pkl')
preprocessor_path = os.path.join(models_dir, 'pipeline_preprocessor.pkl')

with open(preprocessor_path, 'rb') as file:
    preprocessor = joblib.load(file)

with open(dt_model_path, 'rb') as file:
    dt_model = joblib.load(file)

with open(rf_model_path, 'rb') as file:
    rf_model = joblib.load(file)

# ------------------ Batch Prediction ------------------
def predict_batch():
    uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.drop('customerID', axis=1)
        
        # Handle NaNs
        imputer = SimpleImputer(strategy='most_frequent')
        df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        df_filled['TotalCharges'] = pd.to_numeric(df_filled['TotalCharges'], errors='coerce')

        if 'Churn' in df_filled.columns:
            df_filled['Churn'] = df_filled['Churn'].map({'Yes': 1, 'No': 0})

            # Categorical encoding
            categorical_cols = df_filled.select_dtypes(include=['object']).columns
            df_encoded = pd.get_dummies(df_filled, columns=categorical_cols, drop_first=True)

            X = df_encoded.drop(columns=['Churn'])
            y = df_encoded['Churn']
            numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
            X[numerical_cols] = StandardScaler().fit_transform(X[numerical_cols])

            model_choice = st.selectbox("Select Model", ["SVM", "XGBoost", "ANN", "RNN"])

            if st.button("Predict"):
                if model_choice == "SVM":
                    model = SVC(random_state=42, probability=True)
                    model.fit(X, y)
                    predictions = model.predict(X)
                    churn_percentage = (predictions.sum() / len(predictions)) * 100

                elif model_choice == "XGBoost":
                    model = XGBClassifier(random_state=42)
                    model.fit(X, y)
                    predictions = model.predict(X)
                    churn_percentage = (predictions.sum() / len(predictions)) * 100

                elif model_choice == "ANN":
                    ann = Sequential([
                        Dense(64, input_dim=X.shape[1], activation='relu'),
                        Dense(32, activation='relu'),
                        Dense(1, activation='sigmoid')
                    ])
                    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    ann.fit(X, y, epochs=10, batch_size=16, verbose=0)
                    predictions = (ann.predict(X) > 0.5).astype(int)
                    churn_percentage = (predictions.sum() / len(predictions)) * 100

                elif model_choice == "RNN":
                    # Convert all features to float and handle NaNs
                    X_numeric = X.astype(np.float32).fillna(0)

                    # Reshape for LSTM: (samples, timesteps=1, features)
                    X_seq = np.expand_dims(X_numeric.values, axis=1)

                    # Build a simple LSTM
                    rnn = Sequential([
                        LSTM(32, input_shape=(X_seq.shape[1], X_seq.shape[2])),
                        Dense(16, activation='relu'),
                        Dense(1, activation='sigmoid')
                    ])
                    rnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    rnn.fit(X_seq, y.values, epochs=10, batch_size=16, verbose=0)
                    predictions = (rnn.predict(X_seq) > 0.5).astype(int)
                    churn_percentage = (predictions.sum() / len(predictions)) * 100

                # Display churn percentage if applicable
                if churn_percentage is not None:
                    st.write(f"Churn Percentage ({model_choice}): {churn_percentage:.2f}%")

                    # Churn risk gauge
                    thresholds = [20, 40]
                    colors = ['#8A2BE2', '#FFFF00', '#FFA500']
                    risk_level = np.digitize(churn_percentage, thresholds, right=True)
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=churn_percentage,
                        title={'text': "Churn Risk"},
                        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': colors[risk_level]},
                               'steps': [{'range':[0, thresholds[0]], 'color': colors[0]},
                                         {'range':[thresholds[0], thresholds[1]], 'color': colors[1]},
                                         {'range':[thresholds[1],100],'color': colors[2]}]}
                    ))
                    st.plotly_chart(fig)
        else:
            st.error("Churn column not found.")

# ------------------ Online Prediction ------------------
def predict_online():
    st.subheader("Enter Customer Details")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.header('Demographics')
        gender = st.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.selectbox('Senior Citizen', ['No', 'Yes'])
        partner = st.selectbox('Partner', ['No', 'Yes'])
        dependents = st.selectbox('Dependents', ['No', 'Yes'])

    with col2:
        st.header('Services')
        phone_service = st.selectbox('Phone Service', ['No', 'Yes'])
        multiple_lines = st.selectbox('Multiple Lines', ['No', 'Yes'])
        internet_service = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.selectbox('Online Security', ['No', 'Yes', 'No phone service'])
        online_backup = st.selectbox('Online Backup', ['No', 'Yes', 'No phone service'])
        device_protection = st.selectbox('Device Protection', ['No', 'Yes', 'No phone service'])
        tech_support = st.selectbox('Tech Support', ['No', 'Yes', 'No phone service'])
        streaming_tv = st.selectbox('Streaming TV', ['No', 'Yes', 'No phone service'])
        streaming_movies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No phone service'])

    with col3:
        st.header('Payments')
        contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.selectbox('Paperless Billing', ['No', 'Yes'])
        payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        monthly_charges = st.number_input('Monthly Charges', min_value=0)
        total_charges = st.number_input('Total Charges', min_value=0)
        tenure = st.number_input('Tenure', min_value=0)

    if st.button('Predict'):
        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        # Preprocess data
        preprocessed_data = preprocessor.transform(input_data)

        selected_model = st.session_state.get('model', 'DecisionTree')
        if selected_model == 'DecisionTree':
            model = dt_model
        else:
            model = rf_model

        prediction = model.predict_proba(preprocessed_data)
        churn_percentage = prediction[0][1] * 100
        st.success(f'Churn Percentage ({selected_model} Model): {churn_percentage:.2f}%')

        # Visualize churn risk
        st.subheader("Churn Risk Meter")
        colors = ['#8A2BE2', '#FFFF00', '#FFA500']  # Violet, Yellow, Orange
        thresholds = [20, 40]
        levels = ['Low Churn Risk', 'Medium Churn Risk', 'High Churn Risk']
        risk_level = np.digitize(churn_percentage, thresholds, right=True)

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=churn_percentage,
            title={'text': "Churn Risk"},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': colors[risk_level]},
                'steps': [
                    {'range': [0, thresholds[0]], 'color': colors[0]},
                    {'range': [thresholds[0], thresholds[1]], 'color': colors[1]},
                    {'range': [thresholds[1], 100], 'color': colors[2]}
                ],
            }
        ))

        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig)

        st.write(f"Churn Percentage: {churn_percentage:.2f}%")

# ------------------ Main ------------------
def main():
    st.title("Churn Prediction Application with Advanced Models")
    option = st.radio("Select Prediction Option", ["Online", "Batch"])
    if option == "Online":
        st.session_state['model'] = st.selectbox('Select Model', ['DecisionTree', 'RandomForest'])
        predict_online()
    else:
        predict_batch()

if __name__ == "__main__":
    main()
