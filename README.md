# House Price Prediction using Linear Regression

This repository contains the code and data : Linear Regression, focused on predicting house prices.

## Objective
To implement and evaluate simple and multiple linear regression models for house price prediction using Scikit-learn.

## Dataset
The dataset used is `House_Data.csv`, containing information about houses such as price, area, number of bedrooms, bathrooms, etc.

## Files
- `House_Price_Regression.ipynb`: Jupyter Notebook containing the Python code for data loading, preprocessing, model training (Simple and Multiple Linear Regression), evaluation (MAE, MSE, RMSE, R²), and visualization.
- `House_Data.csv`: The dataset used for the analysis.
- `README.md`: This explanatory file.
- `.gitignore`: Specifies intentionally untracked files that Git should ignore (if added).

## Steps Taken
1. Loaded the dataset using Pandas.
2. Performed initial data inspection (info, describe, null check).
3. Selected 'area' as the feature for Simple Linear Regression and ['area', 'bedrooms', 'bathrooms', 'stories', 'parking'] for Multiple Linear Regression.
4. Split the data into training (80%) and testing (20%) sets.
5. Trained `sklearn.linear_model.LinearRegression` models.
6. Evaluated models using MAE, MSE, RMSE, and R-squared metrics on the test set.
7. Interpreted the coefficients of both models.
8. Visualized the simple regression line and plotted Actual vs. Predicted values & Residuals for the multiple regression model.

## Key Results 
- **Simple Linear Regression (using 'area'):**
  - R² score: 0.2729
  - Interpretation: For each unit increase in area, the price is predicted to increase by $425.73.
- **Multiple Linear Regression (using 'area', 'bedrooms', 'bathrooms', 'stories', 'parking'):**
  - R² score: 5464
  - This model explains more variance in price compared to the simple model.
    
## How to Run
1. Clone this repository.
2. Ensure Python and necessary libraries (`pandas`, `scikit-learn`, `matplotlib`, `notebook`) are installed.
3. Navigate to the repository folder in your terminal.
4. Launch Jupyter Notebook (`jupyter notebook`).
5. Open and run the `House_Price_Regression.ipynb` notebook.

## Important Questions and Answers

### 1. What assumptions does linear regression make?
Linear regression relies on several key assumptions for the model estimates and inferences to be valid:
*   **Linearity:** The relationship between the independent variables (features) and the dependent variable (target) is linear.
*   **Independence:** The observations (and therefore, the errors or residuals) are independent of each other. This is often relevant in time-series data where autocorrelation might occur.
*   **Normality:** The errors (residuals) are normally distributed. This is particularly important for constructing valid confidence intervals and hypothesis tests for the coefficients.
*   **Equal Variance (Homoscedasticity):** The errors (residuals) have constant variance across all levels of the independent variables. The spread of errors should be consistent.
*   **(Often Implicit) No Perfect Multicollinearity:** The independent variables should not be perfectly correlated with each other. High (but not perfect) multicollinearity can also be problematic.

### 2. How do you interpret the coefficients?
*   **Intercept (β₀):** Represents the estimated average value of the dependent variable (target) when all independent variables (features) included in the model are equal to zero. Its practical interpretation depends on whether zero is a meaningful value for all features.
*   **Slope Coefficients (β₁, β₂, ... βₚ):** The coefficient for a specific independent variable represents the estimated average change in the dependent variable for a *one-unit increase* in that independent variable, **holding all other independent variables in the model constant**. The sign (+/-) indicates the direction of the relationship.

### 3. What is R² score and its significance?
*   **R-squared (R²)**, also known as the coefficient of determination, is a statistical measure representing the proportion of the variance in the dependent variable (target) that is explained by the independent variables (features) included in the model.
*   **Significance:**
    *   It ranges from 0 to 1 (or 0% to 100%).
    *   A higher R² indicates that a larger proportion of the target variable's variability is accounted for by the model, suggesting a better fit to the data.
    *   An R² of 0 means the model explains none of the variability. An R² of 1 means the model explains all the variability.
    *   It's a measure of "goodness of fit," but a high R² doesn't automatically mean the model is good or unbiased. It tends to increase just by adding more variables, so *Adjusted R²* is often preferred as it penalizes the inclusion of non-informative predictors.

### 4. When would you prefer MSE over MAE?
*   **MSE (Mean Squared Error):** Calculates the average of the *squared* differences between actual and predicted values. `MSE = mean((actual - predicted)^2)`
*   **MAE (Mean Absolute Error):** Calculates the average of the *absolute* differences between actual and predicted values. `MAE = mean(|actual - predicted|)`
*   **Preference:**
    *   Prefer **MSE** (or its square root, RMSE) when **large errors are particularly undesirable**. Squaring the error term heavily penalizes larger deviations. If outliers are significant and represent phenomena you need the model to account for accurately, MSE might be more sensitive to them.
    *   Prefer **MAE** when you want a metric that is **more robust to outliers** (as it doesn't square the errors) and is **more easily interpretable** in the original units of the target variable (representing the average magnitude of error).

### 5. How do you detect multicollinearity?
Multicollinearity occurs when independent variables in a regression model are highly correlated with each other. You can detect it using:
*   **Correlation Matrix/Heatmap:** Calculate the pairwise correlation coefficients between all independent variables. Look for high absolute values (e.g., > 0.7 or 0.8) between pairs of predictors.
*   **Variance Inflation Factor (VIF):** Calculate the VIF for each independent variable. VIF measures how much the variance of an estimated regression coefficient increases due to collinearity.
    *   A VIF > 1 indicates some correlation.
    *   A common rule of thumb is that VIF > 5 or VIF > 10 suggests problematic multicollinearity that may need addressing (e.g., by removing one of the correlated features or combining them).

### 6. What is the difference between simple and multiple regression?
*   **Simple Linear Regression:** Models the linear relationship between **one** independent variable (feature) and one dependent variable (target). Equation: `y = β₀ + β₁x + ε`.
*   **Multiple Linear Regression:** Models the linear relationship between **two or more** independent variables (features) and one dependent variable (target). Equation: `y = β₀ + β₁x₁ + β₂x₂ + ... + βₚxₚ + ε`.

### 7. Can linear regression be used for classification?
**No, not directly or appropriately.** Linear regression is designed for predicting continuous outcomes. Classification deals with predicting discrete categories or labels.
*   **Why not?**
    *   Linear regression outputs continuous values (e.g., from -∞ to +∞), which don't naturally map to probabilities (which should be between 0 and 1) or discrete class labels.
    *   The linear relationship assumed by the model often doesn't fit the decision boundary needed for classification.
    *   The error assumptions (especially normality) are usually violated when the dependent variable is categorical.
*   **Alternative:** Models like *Logistic Regression* are designed specifically for binary (and multinomial) classification tasks, outputting probabilities that can be thresholded into class predictions.

### 8. What happens if you violate regression assumptions?
Violating the assumptions can lead to unreliable or misleading results:
*   **Linearity Violation:** Model coefficients may be biased, and predictions will be inaccurate as the model doesn't capture the true underlying relationship.
*   **Independence Violation (Autocorrelation):** Standard errors of the coefficients tend to be underestimated, making predictors seem statistically significant when they might not be. Confidence intervals will be too narrow.
*   **Normality Violation:** Affects the validity of hypothesis tests (p-values) and confidence intervals for the coefficients, especially with small sample sizes. However, coefficient estimates remain unbiased.
*   **Homoscedasticity Violation (Heteroscedasticity):** Standard errors become unreliable (often underestimated), impacting hypothesis tests and confidence intervals. Coefficient estimates are still unbiased but are not the most efficient (BLUE - Best Linear Unbiased Estimator property is lost).
*   **Multicollinearity:** Coefficient estimates become unstable (sensitive to small data changes) and difficult to interpret reliably. Standard errors inflate, potentially making truly important predictors appear insignificant.
