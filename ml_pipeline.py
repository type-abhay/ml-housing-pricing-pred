import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report

# Ensure models directory exists
os.makedirs('models', exist_ok=True)

print("--- PHASE 1: Data Preprocessing & EDA ---")
# 1. Load Dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target # MedHouseVal (Continuous)

# 2. Fixed Data Split (70% Train, 15% Val, 15% Test) with z=42
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42)

# 3. Feature Scaling (Strictly on Training Set)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')

# (EDA Visualization Code would go here - e.g., sns.heatmap, plt.hist)

print("--- PHASE 2: Regression Analysis ---")
# Multiple Linear Regression
mlr = LinearRegression()
mlr.fit(X_train_scaled, y_train)
y_pred_test_mlr = mlr.predict(X_test_scaled)
print(f"MLR Test MSE: {mean_squared_error(y_test, y_pred_test_mlr):.3f}")
print(f"MLR Test R2: {r2_score(y_test, y_pred_test_mlr):.3f}")
joblib.dump(mlr, 'models/mlr_model.pkl')

print("--- PHASE 3: Classification Models ---")
# Derive Target (Bottom 33%, Middle 33%, Top 33%)
quantiles = np.quantile(y_train, [0.33, 0.67])
def categorize(val):
    if val <= quantiles[0]: return 0      # Low
    elif val <= quantiles[1]: return 1    # Medium
    else: return 2                        # High

y_train_class = np.array([categorize(v) for v in y_train])
y_val_class = np.array([categorize(v) for v in y_val])
y_test_class = np.array([categorize(v) for v in y_test])

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train_scaled, y_train_class)
y_pred_rf = rf.predict(X_test_scaled)
print("Random Forest Accuracy:", accuracy_score(y_test_class, y_pred_rf))

print("--- PHASE 4: Support Vector Machine ---")
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train_scaled, y_train_class)
y_pred_svm = svm_model.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test_class, y_pred_svm))

print("--- PHASE 5: Neural Network ---")
# MLP using early stopping on validation set implicitly via hyperparams or explicitly
nn = MLPRegressor(hidden_layer_sizes=(64, 32), early_stopping=True, validation_fraction=0.15, random_state=42)
nn.fit(X_train_scaled, y_train)
y_pred_nn = nn.predict(X_test_scaled)
print(f"Neural Network Test MSE: {mean_squared_error(y_test, y_pred_nn):.3f}")

print("Pipeline Complete. Models saved for Web Deployment.")