# ==========================================================
# TASK 01: House Price Prediction using Linear Regression
# Dataset: House Prices - Advanced Regression Techniques
# ==========================================================

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==========================================================
# Load Dataset
# ==========================================================

data = pd.read_csv("train.csv")

print("="*50)
print("DATASET INFORMATION")
print("="*50)

print("\nDataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())

print("\nDataset Info:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())

# ==========================================================
# Select Required Features
# ==========================================================

data = data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']]

print("\nSelected Features:")
print(data.head())

# ==========================================================
# Check Missing Values
# ==========================================================

print("\nMissing Values:")
print(data.isnull().sum())

# ==========================================================
# Distribution of House Prices
# ==========================================================

plt.figure(figsize=(8,5))
plt.hist(data['SalePrice'], bins=30)
plt.title('Distribution of House Prices')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.savefig('distribution.png')
plt.show()

# ==========================================================
# Living Area vs Sale Price
# ==========================================================

plt.figure(figsize=(8,5))
plt.scatter(data['GrLivArea'], data['SalePrice'])
plt.title('Living Area vs Sale Price')
plt.xlabel('Living Area (Square Feet)')
plt.ylabel('Sale Price')
plt.savefig('livingarea_vs_price.png')
plt.show()

# ==========================================================
# Correlation Heatmap
# ==========================================================

plt.figure(figsize=(8,6))
sns.heatmap(
    data.corr(),
    annot=True,
    cmap='coolwarm'
)
plt.title('Correlation Heatmap')
plt.savefig('heatmap.png')
plt.show()

# ==========================================================
# Feature Selection
# ==========================================================

X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

# ==========================================================
# Train Test Split
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# ==========================================================
# Build Linear Regression Model
# ==========================================================

model = LinearRegression()

# ==========================================================
# Train Model
# ==========================================================

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# ==========================================================
# Predictions
# ==========================================================

y_pred = model.predict(X_test)

# ==========================================================
# Evaluation Metrics
# ==========================================================

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*50)
print("MODEL PERFORMANCE")
print("="*50)

print(f"Mean Absolute Error (MAE)  : {mae:.2f}")
print(f"Mean Squared Error (MSE)   : {mse:.2f}")
print(f"Root Mean Squared Error    : {rmse:.2f}")
print(f"R² Score                   : {r2:.4f}")

# ==========================================================
# Model Coefficients
# ==========================================================

print("\n" + "="*50)
print("MODEL COEFFICIENTS")
print("="*50)

coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print(coefficients)

print("\nIntercept:")
print(model.intercept_)

# ==========================================================
# Actual vs Predicted Plot
# ==========================================================

plt.figure(figsize=(8,6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")

plt.savefig("actual_vs_predicted.png")
plt.show()

# ==========================================================
# Residual Plot
# ==========================================================

residuals = y_test - y_pred

plt.figure(figsize=(8,6))

plt.scatter(y_pred, residuals)

plt.axhline(y=0, linestyle='--')

plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.title("Residual Error Plot")

plt.savefig("residual_plot.png")
plt.show()

# ==========================================================
# Predict New House Price
# ==========================================================

print("\n" + "="*50)
print("NEW HOUSE PRICE PREDICTION")
print("="*50)

new_house = pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'FullBath': [2]
})

predicted_price = model.predict(new_house)

print(f"Predicted House Price: ${predicted_price[0]:,.2f}")

# ==========================================================
# Conclusion
# ==========================================================

print("\n" + "="*50)
print("CONCLUSION")
print("="*50)

print("""
Linear Regression model was successfully built to predict
house prices using:

1. GrLivArea (Square Footage)
2. BedroomAbvGr (Number of Bedrooms)
3. FullBath (Number of Bathrooms)

Evaluation Metrics Used:
- MAE
- MSE
- RMSE
- R² Score

Visualizations Generated:
- House Price Distribution
- Living Area vs Sale Price
- Correlation Heatmap
- Actual vs Predicted Plot
- Residual Plot

All graphs have been saved automatically as PNG files.
""")