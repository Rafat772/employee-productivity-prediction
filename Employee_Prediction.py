# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Step 1: Load Dataset
df = pd.read_csv('dataset/garment_worker_productivity.csv')

# Step 2: Basic Exploration
print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Step 3: Handling Missing Values
df = df.dropna()  # You can also fillna() if you prefer

# Step 4: Convert 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month

# Step 5: Drop unnecessary columns
df = df.drop(columns=['date'])

# Step 6: Handle Categorical Data (Department, day, quarter)
from sklearn.preprocessing import LabelEncoder

cols = ['quarter', 'department', 'day']
le = LabelEncoder()
for col in cols:
    df[col] = le.fit_transform(df[col])

# Step 7: Define features (X) and target (y)
X = df.drop(columns=['actual_productivity'])
y = df['actual_productivity']

# Step 8: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 9: Train Models
# Linear Regression
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)
pred_lr = model_lr.predict(X_test)

# Random Forest
model_rf = RandomForestRegressor()
model_rf.fit(X_train, y_train)
pred_rf = model_rf.predict(X_test)

# XGBoost
model_xgb = XGBRegressor()
model_xgb.fit(X_train, y_train)
pred_xgb = model_xgb.predict(X_test)

# Step 10: Evaluate Models
def evaluate(y_test, pred):
    print("MAE:", mean_absolute_error(y_test, pred))
    print("MSE:", mean_squared_error(y_test, pred))
    print("R2 Score:", r2_score(y_test, pred))

print("Linear Regression:")
evaluate(y_test, pred_lr)

print("Random Forest:")
evaluate(y_test, pred_rf)

print("XGBoost:")
evaluate(y_test, pred_xgb)

# Step 11: Save the best model (Let's assume XGBoost performs best)
with open('model/gwp.pkl', 'wb') as file:
    pickle.dump(model_xgb, file)

print("Model saved successfully!")
