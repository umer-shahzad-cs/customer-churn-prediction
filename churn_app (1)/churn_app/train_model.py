"""Run once: trains the Random Forest and saves model.pkl."""
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from churn_model import encode

df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
df = df.drop(columns=['customerID'])
y = df['Churn'].map({'Yes': 1, 'No': 0})
X = encode(df.drop(columns=['Churn']), drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                               class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

pred = model.predict(X_test)
metrics = {'accuracy': accuracy_score(y_test, pred),
           'recall': recall_score(y_test, pred),
           'f1': f1_score(y_test, pred)}
joblib.dump({'model': model, 'columns': list(X.columns), 'metrics': metrics}, 'model.pkl')
print('Saved model.pkl', metrics)
