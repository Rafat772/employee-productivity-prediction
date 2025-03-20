import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('dataset/garments_worker_productivity.csv')


# Preprocessing
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df.drop(['date'], axis=1, inplace=True)
df.fillna(0, inplace=True)

# Encode categorical columns
encoder = LabelEncoder()
for col in ['quarter', 'department', 'day']:
    df[col] = encoder.fit_transform(df[col])

# Define features and target
X = df.drop('actual_productivity', axis=1)
y = df['actual_productivity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Model
model_xgb = XGBRegressor()
model_xgb.fit(X_train, y_train)

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Save the model in the model folder
with open('model/gwp.pkl', 'wb') as file:
    pickle.dump(model_xgb, file)

print("✅ Model saved successfully in 'model/gwp.pkl'")
