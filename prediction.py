#Prediction model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# Load the dataset (assuming it's in a CSV-like format from your document)
data = pd.read_csv('a.csv')  # Replace with actual file path or use StringIO for your text

# Data Exploration
print("Dataset Info:")
print(data.info())
print("\nFirst 5 rows:")
print(data.head())

# Data Preprocessing
# Drop unnecessary columns (e.g., email might not be useful for prediction)
data = data.drop(['customer name', 'customer e-mail'], axis=1)

# Encode categorical variables
le = LabelEncoder()
data['country'] = le.fit_transform(data['country'])
data['gender'] = le.fit_transform(data['gender'])

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Feature Engineering
# Create interaction term: annual salary × net worth
data['salary_networth_interaction'] = data['annual Salary'] * data['net worth']

# Define features and target
X = data.drop('car purchase amount', axis=1)
y = data['car purchase amount']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Correlation Analysis
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Customer Segmentation with K-means
kmeans = KMeans(n_clusters=3, random_state=42)
data['customer_segment'] = kmeans.fit_predict(X_scaled)
print("\nCustomer Segments Distribution:")
print(data['customer_segment'].value_counts())

# Visualize Clusters (example with two features)
plt.figure(figsize=(8, 6))
sns.scatterplot(x='annual Salary', y='net worth', hue='customer_segment', data=data, palette='viridis')
plt.title('Customer Segmentation (K-means)')
plt.show()

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Baseline Model: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Evaluate Linear Regression
lr_mse = mean_squared_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
print("\nLinear Regression Performance:")
print(f"Mean Squared Error: {lr_mse:.2f}")
print(f"R² Score: {lr_r2:.2f}")

# Advanced Model: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Evaluate Random Forest
rf_mse = mean_squared_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
print("\nRandom Forest Performance:")
print(f"Mean Squared Error: {rf_mse:.2f}")
print(f"R² Score: {rf_r2:.2f}")

# Feature Importance from Random Forest
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': rf_model.feature_importances_})
feature_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Plot Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance (Random Forest)')
plt.show()

# Cross-Validation for Random Forest
cv_scores = cross_val_score(rf_model, X_scaled, y, cv=5, scoring='r2')
print("\nCross-Validation R² Scores:")
print(cv_scores)
print(f"Average CV R²: {cv_scores.mean():.2f} ± {cv_scores.std():.2f}")

# Visualize Predictions vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Car Purchase Amount')
plt.ylabel('Predicted Car Purchase Amount')
plt.title('Actual vs Predicted (Random Forest)')
plt.show()