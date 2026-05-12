# -*- coding: utf-8 -*-
"""
Created on Tue Feb 10 13:06:48 2026

@author: Amol Gaikwad
"""

# LINEAR REGRESSION ON KAGGLE HOUSING PRICES DATASET
# Dataset: House Prices - Advanced Regression Techniques (train.csv)

# Step 1: Import required libraries

import pandas as pd
import numpy as np

# For visualization

import matplotlib.pyplot as plt
import seaborn as sns

# For model building

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# For evaluation

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 2: Load dataset
# Replace path with your actual file location

data = pd.read_csv(r"C:\Users\Amol Gaikwad\Data Science\house_rates\House_Price_Predictions.csv")

# Step 3: Display basic information
print(data.head())
print(data.shape)

# Step 4: Select only numerical features for simplicity
# (Linear regression requires numeric input)

num_data = data.select_dtypes(include=[np.number])

# Step 5: Handle missing values
# Fill missing values with mean

 
# Step 6: Define target variable (what we want to predict)
# SalePrice is the house price

X = num_data.drop("Saleprice", axis=1)   # Features
y = num_data["Saleprice"]                # Target

# Step 7: Split data into train and test
# 80% training, 20% testing

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Create Linear Regression model

model = LinearRegression()

# Step 9: Train model
model.fit(X_train, y_train)

# Step 10: Make predictions
y_pred = model.predict(X_test)

# Step 11: Model Evaluation

# R2 Score (How well model explains variance)
r2 = r2_score(y_test, y_pred)

# Mean Absolute Error (Average prediction error)
mae = mean_absolute_error(y_test, y_pred)

# Mean Squared Error
mse = mean_squared_error(y_test, y_pred)

# Root Mean Squared Error
rmse = np.sqrt(mse)

print("\nMODEL EVALUATION RESULTS")
print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# Step 12: Actual vs Predicted comparison
comparison = pd.DataFrame({
    "Actual Price": y_test,
    "Predicted Price": y_pred
})

print(comparison.head(10))

# Step 13: Plot Actual vs Predicted values
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()

# Step 14: Residuals (Errors)
residuals = y_test - y_pred

plt.figure(figsize=(6,6))
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution")
plt.show()
#Shreyas V. Aundhekar