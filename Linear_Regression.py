# housing_regression.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load and preprocess the dataset
df = pd.read_csv("Housing.csv")

# Check for missing values
print("\n Missing Values \n")
print(df.isnull().sum())

# Encode 'furnishingstatus', 'mainroad', 'guestroom' etc. if they exist
df = pd.get_dummies(df, drop_first=True)

#  Define features and target
X = df.drop("price", axis=1)
y = df["price"]

# Split into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Fit linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.2f}")

# Plotting actual vs predicted prices
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linewidth=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Housing Prices")
plt.tight_layout()
plt.show()

# Coefficients
print("\n Model Coefficients")
coef = pd.Series(model.coef_, index=X.columns)
print(coef)
print(f"\nIntercept: {model.intercept_:.2f}")
