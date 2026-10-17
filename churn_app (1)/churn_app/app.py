"""Flask backend for the Customer Churn Prediction web app."""
import os
import joblib
import pandas as pd
from flask import Flask, render_template, request
from churn_model import encode

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# Saved in train_model.py: the Random Forest, the exact training column order, and test metrics
bundle = joblib.load(os.path.join(BASE, 'model.pkl'))
model, columns, metrics = bundle['model'], bundle['columns'], bundle['metrics']

YES_NO = ['Yes', 'No']
SERVICE = ['No', 'Yes', 'No internet service']

# Dropdown fields: name -> (label, options)
CHOICES = {
    'gender': ('Gender', ['Female', 'Male']),
    'SeniorCitizen': ('Senior citizen', ['No', 'Yes']),
    'Partner': ('Has a partner', YES_NO),
    'Dependents': ('Has dependents', YES_NO),
    'PhoneService': ('Phone service', YES_NO),
    'MultipleLines': ('Multiple phone lines', ['No', 'Yes', 'No phone service']),
    'InternetService': ('Internet service', ['DSL', 'Fiber optic', 'No']),
    'OnlineSecurity': ('Online security', SERVICE),
    'OnlineBackup': ('Online backup', SERVICE),
    'DeviceProtection': ('Device protection', SERVICE),
    'TechSupport': ('Tech support', SERVICE),
    'StreamingTV': ('Streaming TV', SERVICE),
    'StreamingMovies': ('Streaming movies', SERVICE),
    'Contract': ('Contract type', ['Month-to-month', 'One year', 'Two year']),
    'PaperlessBilling': ('Paperless billing', YES_NO),
    'PaymentMethod': ('Payment method', ['Electronic check', 'Mailed check',
                                         'Bank transfer (automatic)', 'Credit card (automatic)']),
}
# Number fields: name -> label
NUMBERS = {
    'tenure': 'Tenure (months with the company)',
    'MonthlyCharges': 'Monthly charges ($)',
    'TotalCharges': 'Total charges ($) - optional',
}
# Form layout: section title -> field names
SECTIONS = [
    ('Account', ['tenure', 'Contract', 'PaymentMethod', 'PaperlessBilling', 'MonthlyCharges', 'TotalCharges']),
    ('Customer', ['gender', 'SeniorCitizen', 'Partner', 'Dependents']),
    ('Services', ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']),
]
DEFAULTS = {'tenure': '12', 'MonthlyCharges': '70', 'TotalCharges': '', 'gender': 'Female',
            'SeniorCitizen': 'No', 'Partner': 'No', 'Dependents': 'No', 'PhoneService': 'Yes',
            'MultipleLines': 'No', 'InternetService': 'Fiber optic', 'OnlineSecurity': 'No',
            'OnlineBackup': 'No', 'DeviceProtection': 'No', 'TechSupport': 'No', 'StreamingTV': 'No',
            'StreamingMovies': 'No', 'Contract': 'Month-to-month', 'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'}


def render(values, result=None, errors=None):
    return render_template('index.html', sections=SECTIONS, choices=CHOICES, numbers=NUMBERS,
                           values=values, result=result, errors=errors or {}, metrics=metrics)


def read_number(text, name, low, high, errors, required=True):
    """Return a float, or None (and record an error message) if the text is not valid."""
    text = (text or '').strip()
    if not text:
        if required:
            errors[name] = 'This field is required.'
        return None
    try:
        number = float(text)
    except ValueError:
        errors[name] = 'Please enter a number.'
        return None
    if not low <= number <= high:
        errors[name] = f'Please enter a value between {low} and {high}.'
        return None
    return number


@app.route('/')
def index():
    return render(DEFAULTS)


@app.route('/predict', methods=['POST'])
def predict():
    values = {k: request.form.get(k, '').strip() for k in list(CHOICES) + list(NUMBERS)}
    errors = {}

    # 1. validate dropdowns (value must be one of the allowed options)
    for name, (label, options) in CHOICES.items():
        if values[name] not in options:
            errors[name] = 'Please choose a valid option.'

    # 2. validate numbers
    tenure = read_number(values['tenure'], 'tenure', 0, 100, errors)
    monthly = read_number(values['MonthlyCharges'], 'MonthlyCharges', 0, 500, errors)
    total = read_number(values['TotalCharges'], 'TotalCharges', 0, 100000, errors, required=False)

    if errors:
        return render(values, errors=errors)
    if total is None:                       # blank total charges -> estimate from tenure x monthly
        total = tenure * monthly

    # 3. build one raw customer row (same columns as the original dataset)
    row = {name: values[name] for name in CHOICES}
    row.update(SeniorCitizen=1 if values['SeniorCitizen'] == 'Yes' else 0,
               tenure=int(tenure), MonthlyCharges=monthly, TotalCharges=total)

    # 4. same preprocessing as training, then align columns to the training order
    X = encode(pd.DataFrame([row]), drop_first=False).reindex(columns=columns, fill_value=0)

    # 5. predict
    probability = float(model.predict_proba(X)[0, 1])
    churn = probability >= 0.5
    result = {
        'label': 'Churn' if churn else 'No Churn',
        'css': 'churn' if churn else 'stay',
        'percent': round(probability * 100, 1),
        'message': ('This customer is likely to leave. A retention offer is recommended.' if churn
                    else 'This customer is likely to stay with the company.'),
    }
    return render(values, result=result)


if __name__ == '__main__':
    app.run(debug=True)
