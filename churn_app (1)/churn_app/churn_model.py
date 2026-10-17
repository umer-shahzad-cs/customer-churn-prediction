"""Shared preprocessing so training and prediction always use the same steps."""
import pandas as pd

SERVICE_COLS = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
COLS_INTERNET = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']
BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
DUMMY_COLS = ['InternetService', 'Contract', 'PaymentMethod']


def encode(df, drop_first):
    """Clean, engineer and encode raw customer rows (without Churn)."""
    df = df.copy()
    for c in COLS_INTERNET:
        df[c] = df[c].replace('No internet service', 'No')
    df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
    df['num_services'] = (df[SERVICE_COLS] == 'Yes').sum(axis=1)
    df['auto_pay'] = df['PaymentMethod'].str.contains('automatic').astype(int)
    for c in BINARY_COLS:
        df[c] = df[c].map({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0})
    return pd.get_dummies(df, columns=DUMMY_COLS, drop_first=drop_first, dtype=int)
