import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # Seaborn is often useful even if not strictly required

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure settings
%matplotlib inline
plt.style.use('seaborn-v0_8-darkgrid') # Use a nice style for plots
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('House_data.csv') 

# Display basic info
print("Dataset Info:\n")
df.info()

print("\nFirst 5 Rows:")
display(df.head())

print("\nSummary Statistics:")
display(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

print(f"\nDataset Shape: {df.shape}")

feature_simple = ['area']
target = 'price'

# Prepare data for Simple Linear Regression
X_simple = df[feature_simple]
y = df[target]

print(f"Selected feature for Simple Regression: {feature_simple}")
print(f"Target variable: {target}")
print("\nShape of X_simple:", X_simple.shape)
print("Shape of y:", y.shape)

features_multiple = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking']

X_multiple = df[features_multiple]
# y remains the same (df[target])

print(f"\nSelected features for Multiple Regression: {features_multiple}")
print("\nShape of X_multiple:", X_multiple.shape)
print("Shape of y:", y.shape)

print("\nNote: Categorical features (e.g., 'mainroad', 'guestroom') were not included in this basic model.")
print("\nThey would require encoding (like mapping 'yes'/'no' to 1/0 or One-Hot Encoding) to be used.")

# --- Splitting for SIMPLE Linear Regression ---
# Uses X_simple (containing 'area') and y ('price') defined in the previous cell
X_train_simple, X_test_simple, y_train, y_test =  train_test_split( X_simple, y, test_size=0.2, random_state=42 )

print("--- Simple Regression Shapes ---\n")
print("X_train_simple:", X_train_simple.shape)
print("X_test_simple:", X_test_simple.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

# --- Splitting for MULTIPLE Linear Regression ---
# Uses X_multiple (containing 'area', 'bedrooms', etc.) and y ('price') defined in the previous cell
X_train_multiple, X_test_multiple, y_train_multi, y_test_multi = train_test_split( X_multiple, y, test_size=0.2, random_state=42 )

# Let's reuse y_train and y_test for consistency, as they come from the same split of y
# y_train = y_train_multi <--- No, use the original y_train/y_test from the simple split for consistency
# y_test = y_test_multi  <--- as long as random_state is the same, they are identical

print("--- Multiple Regression Shapes ---\n")
print("X_train_multiple:", X_train_multiple.shape)
print("X_test_multiple:", X_test_multiple.shape)
print("y_train:", y_train.shape) # Reusing y_train from simple split
print("y_test:", y_test.shape)   # Reusing y_test from simple split

# --- Simple Linear Regression ---
print("--- Training Simple Linear Regression ---\n")
model_simple = LinearRegression()
# Fit using the training data created in the previous cell
model_simple.fit(X_train_simple, y_train)
print("Simple Linear Regression model trained successfully.")

# --- Multiple Linear Regression ---
print("--- Training Multiple Linear Regression ---\n")
model_multiple = LinearRegression()
# Fit using the multiple features training set created previously
model_multiple.fit(X_train_multiple, y_train)
print("Multiple Linear Regression model trained successfully.")

# --- Evaluate Simple Linear Regression ---
print("--- Simple Linear Regression Evaluation (using 'area') ---\n")
# Predict on the simple test set
y_pred_simple = model_simple.predict(X_test_simple)

# Calculate metrics comparing actual test prices (y_test) vs predicted prices (y_pred_simple)
mae_simple = mean_absolute_error(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)
rmse_simple = np.sqrt(mse_simple) # Root Mean Squared Error
r2_simple = r2_score(y_test, y_pred_simple)

print(f"Mean Absolute Error (MAE): {mae_simple:,.2f}")
print(f"Mean Squared Error (MSE): {mse_simple:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_simple:,.2f}")
print(f"R-squared (R²): {r2_simple:.4f}")

# --- Evaluate Multiple Linear Regression ---
print("--- Multiple Linear Regression Evaluation ---\n")
# Predict using the multiple regression model on its corresponding test set
y_pred_multiple = model_multiple.predict(X_test_multiple)

# Calculate metrics comparing actual test prices (y_test) vs predicted prices (y_pred_multiple)
mae_multiple = mean_absolute_error(y_test, y_pred_multiple) # Compare against the same y_test
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
rmse_multiple = np.sqrt(mse_multiple)
r2_multiple = r2_score(y_test, y_pred_multiple)

print(f"Mean Absolute Error (MAE): {mae_multiple:,.2f}")
print(f"Mean Squared Error (MSE): {mse_multiple:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_multiple:,.2f}")
print(f"R-squared (R²): {r2_multiple:.4f}")

# Compare R² scores
print(f"Comparison: \n Simple R² = {r2_simple:.4f}, \n Multiple R² = {r2_multiple:.4f}")
if r2_multiple > r2_simple:
    print("Multiple Regression explains more variance in the price.")
else:
    print("Multiple Regression does not explain significantly more variance than Simple Regression (using only 'area').")

# --- Interpret Simple Linear Regression (using 'area') ---
print("--- Simple Linear Regression Interpretation ('area') ---\n ")
simple_intercept = model_simple.intercept_
simple_coefficient = model_simple.coef_[0] # Get the coefficient for 'area'

print(f"Intercept (Estimated base price): ${simple_intercept:,.2f}")
print(f"Coefficient for '{feature_simple[0]}': {simple_coefficient:,.2f}") # Use feature_simple[0] which is 'area'
print(f"Interpretation: For each 1 unit increase in '{feature_simple[0]}', the predicted price increases by approximately ${simple_coefficient:,.2f}.")

# --- Plot Simple Linear Regression ---
plt.figure(figsize=(10, 6))
# Scatter plot of actual test data points (area vs price)
plt.scatter(X_test_simple, y_test, alpha=0.5, label='Actual Prices')
# Plot the regression line (predicted prices for the test areas)
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Regression Line')
plt.title(f'Simple Linear Regression: Price vs. {feature_simple[0]}') # Updated title
plt.xlabel(feature_simple[0]) # Updated x-axis label
plt.ylabel('Price')
plt.ticklabel_format(style='plain', axis='y') # Prevent scientific notation on y-axis
plt.ticklabel_format(style='plain', axis='x')
plt.legend()
plt.show()

# --- Interpret Multiple Linear Regression ---
print("--- Multiple Linear Regression Interpretation ---\n")
multiple_intercept = model_multiple.intercept_
multiple_coefficients = model_multiple.coef_

print(f"Intercept (Estimated base price): ${multiple_intercept:,.2f}")
# Create a DataFrame using the correct feature names from 'features_multiple'
coeffs = pd.DataFrame(multiple_coefficients, index=features_multiple, columns=['Coefficient'])
print("\nCoefficients for Features:")
display(coeffs.sort_values('Coefficient', ascending=False))

print("\nInterpretation Examples:")
# Use .loc to access coefficients by name, ensure names match 'features_multiple'
if 'area' in coeffs.index:
    print(f"- For each 1 unit increase in 'area', the predicted price changes by ${coeffs.loc['area'].values[0]:,.2f}, holding all other listed features constant.")
if 'bedrooms' in coeffs.index:
    print(f"- For each 1 unit increase in 'bedrooms', the predicted price changes by ${coeffs.loc['bedrooms'].values[0]:,.2f}, holding all other listed features constant.")
if 'bathrooms' in coeffs.index:
    print(f"- For each 1 unit increase in 'bathrooms', the predicted price changes by ${coeffs.loc['bathrooms'].values[0]:,.2f}, holding all other listed features constant.")
# Add interpretations for 'stories' and 'parking' if desired.

# --- Plot Multiple Linear Regression Results ---
# Plot Actual vs. Predicted values. This plot works regardless of the number of features.
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_multiple, alpha=0.5)
# Plot the ideal line where Actual = Predicted
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', lw=2, color='red', label='Ideal Fit Line (Actual=Predicted)')
plt.title('Multiple Linear Regression: Actual vs. Predicted Prices')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
plt.legend()
# Add R² score text to the plot
plt.text(0.05, 0.9, f'$R^2 = {r2_multiple:.4f}$', transform=plt.gca().transAxes, fontsize=12,
         bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.5))
plt.show()

# Plot Residuals (Errors = Actual - Predicted) vs. Predicted Values
residuals = y_test - y_pred_multiple
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_multiple, residuals, alpha=0.5)
# Add a horizontal line at zero error
plt.hlines(0, xmin=y_pred_multiple.min(), xmax=y_pred_multiple.max(), colors='red', linestyles='--', lw=2)
plt.title('Residuals vs. Predicted Values')
plt.xlabel('Predicted Price')
plt.ylabel('Residuals (Actual - Predicted)')
plt.ticklabel_format(style='plain', axis='y')
plt.ticklabel_format(style='plain', axis='x')
plt.show()
