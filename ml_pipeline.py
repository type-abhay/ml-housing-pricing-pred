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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix


os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

print("==================================================")
print("PHASE 1: Data Preprocessing & Split")
print("==================================================")
print("Fetching data..")
try:
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target
    print("Fetching Successfully, beginning training..")
except:
    print("Failed..")

X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=(0.15/0.85), random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, 'models/scaler.pkl')
print("Data successfully split and scaled.")
print("\n--> Generating Exploratory Data Analysis (EDA) Plots...")

train_df = pd.DataFrame(X_train, columns=data.feature_names)
train_df['MedHouseVal'] = y_train

# 1. Feature Distribution (Histogram of Median Income)
plt.figure(figsize=(8, 6))
sns.histplot(train_df['MedInc'], bins=50, kde=True, color='purple')
plt.title('Distribution of Median Income (Training Set)')
plt.xlabel('Median Income')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)
plt.savefig('plots/4_EDA_Feature_Distribution.png')
plt.close()

# 2. Scatter Plot (Latitude vs Longitude overlaid with House Value)
plt.figure(figsize=(10, 7))
sns.scatterplot(data=train_df, x='Longitude', y='Latitude', hue='MedHouseVal', palette='viridis', alpha=0.6)
plt.title('Geographical Scatter Plot of House Values (Training Set)')
plt.savefig('plots/4_EDA_Scatter_Plot.png')
plt.close()

# 3. Correlation Heatmap
plt.figure(figsize=(10, 8))
correlation_matrix = train_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap (Training Set)')
plt.savefig('plots/4_EDA_Correlation_Heatmap.png')
plt.close()

print("EDA Plots successfully generated and saved to 'plots/' folder.")


print("\n==================================================")
print("PHASE 2: Regression Analysis")
print("==================================================")

# 5.1 Simple Linear Regression (Using MedInc - feature index 0)
print("\n-> Training Simple Linear Regression (Feature: MedInc)...")
slr = LinearRegression()
X_train_slr = X_train_scaled[:, 0:1] # Only Median Income
X_test_slr = X_test_scaled[:, 0:1]

slr.fit(X_train_slr, y_train)
y_pred_slr = slr.predict(X_test_slr)

slr_mse = mean_squared_error(y_test, y_pred_slr)
slr_r2 = r2_score(y_test, y_pred_slr)
print(f"SLR - Mean Squared Error (MSE): {slr_mse:.3f}")
print(f"SLR - $R^2$ Score: {slr_r2:.3f}")

# Plot Actual vs Predicted for SLR
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_slr, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Simple Linear Regression: Actual vs Predicted')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.grid(True)
plt.savefig('plots/5_1_SLR_Actual_vs_Predicted.png')
plt.close()

# 5.2 Multiple Linear Regression (All features)
print("\n-> Training Multiple Linear Regression (All Features)...")
mlr = LinearRegression()
mlr.fit(X_train_scaled, y_train)
y_pred_mlr = mlr.predict(X_test_scaled)

mlr_mse = mean_squared_error(y_test, y_pred_mlr)
mlr_r2 = r2_score(y_test, y_pred_mlr)
print(f"MLR - Mean Squared Error (MSE): {mlr_mse:.3f}")
print(f"MLR - $R^2$ Score: {mlr_r2:.3f}")
joblib.dump(mlr, 'models/mlr_model.pkl')

# Plot Actual vs Predicted for MLR
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_mlr, alpha=0.3, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Multiple Linear Regression: Actual vs Predicted')
plt.xlabel('Actual Median House Value')
plt.ylabel('Predicted Median House Value')
plt.grid(True)
plt.savefig('plots/5_2_MLR_Actual_vs_Predicted.png')
plt.close()

print("\n--- Regression Comparison ---")
print(f"Adding all features improved the $R^2$ score from {slr_r2:.3f} to {mlr_r2:.3f}.")


print("\n==================================================")
print("PHASE 3: Classification Models")
print("==================================================")

# Derive Target Classes: Low (0), Medium (1), High (2)
quantiles = np.quantile(y_train, [0.33, 0.67])
def categorize(val):
    if val <= quantiles[0]: return 0
    elif val <= quantiles[1]: return 1
    else: return 2

y_train_class = np.array([categorize(v) for v in y_train])
y_val_class = np.array([categorize(v) for v in y_val])
y_test_class = np.array([categorize(v) for v in y_test])
class_names = ['Low', 'Medium', 'High']

def evaluate_classifier(model, name, filename_prefix):
    model.fit(X_train_scaled, y_train_class)
    y_pred = model.predict(X_test_scaled)
    
    print(f"\n--- {name} ---")
    print(f"Accuracy: {accuracy_score(y_test_class, y_pred):.3f}")
    print("\n-> Classification Report:")
    print(classification_report(y_test_class, y_pred, target_names=class_names))
    
    # Generate and Save Confusion Matrix
    cm = confusion_matrix(y_test_class, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix: {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'plots/6_{filename_prefix}_Confusion_Matrix.png')
    plt.close()
    return accuracy_score(y_test_class, y_pred)

# Train and evaluate classifiers
acc_lr = evaluate_classifier(LogisticRegression(max_iter=1000, random_state=42), "1. Logistic Regression", "LR")
acc_dt = evaluate_classifier(DecisionTreeClassifier(random_state=42), "2. Decision Tree", "DT")
acc_rf = evaluate_classifier(RandomForestClassifier(random_state=42), "3. Random Forest", "RF")


print("\n==================================================")
print("PHASE 4: Support Vector Machine")
print("==================================================")
# Selected RBF Kernel for non-linear boundary mapping
print("\n-> Training SVM (Kernel used: RBF)")
svm_model = SVC(kernel='rbf', random_state=42)
acc_svm = evaluate_classifier(svm_model, "Support Vector Machine (RBF)", "SVM")

print("\n==================================================")
print("COMPARATIVE DISCUSSION")
print("==================================================")
print(f"1. Logistic Regression Accuracy: {acc_lr:.3f}")
print(f"2. Decision Tree Accuracy:       {acc_dt:.3f}")
print(f"3. Random Forest Accuracy:       {acc_rf:.3f}")
print(f"4. SVM (RBF Kernel) Accuracy:    {acc_svm:.3f}")
print("\nInsight: Random Forest typically outperforms standard Decision Trees due to ensemble bagging,")
print("while SVM with an RBF kernel performs exceptionally well at capturing non-linear geographical boundaries.")

print("\n==================================================")
print("PHASE 5: Neural Network")
print("==================================================")
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss

# Initialize the Neural Network for classification
nn = MLPClassifier(hidden_layer_sizes=(64, 32), solver='adam', 
                   max_iter=1, warm_start=True, random_state=42)

epochs = 100
train_loss, val_loss = [], []
train_acc, val_acc = [], []

best_val_loss = float('inf')
patience = 10
patience_counter = 0
classes = np.unique(y_train_class)

print("-> Training Neural Network with manual Early Stopping monitoring..")

for epoch in range(epochs):
    # Train strictly on the Training Set
    nn.partial_fit(X_train_scaled, y_train_class, classes=classes)
    # Predict probabilities to calculate Cross-Entropy Loss
    y_train_prob = nn.predict_proba(X_train_scaled)
    y_val_prob = nn.predict_proba(X_val_scaled)
    
    # Calculate Loss
    t_loss = log_loss(y_train_class, y_train_prob)
    v_loss = log_loss(y_val_class, y_val_prob)
    train_loss.append(t_loss)
    val_loss.append(v_loss)
    
    # Calculate Accuracy
    t_acc = accuracy_score(y_train_class, nn.predict(X_train_scaled))
    v_acc = accuracy_score(y_val_class, nn.predict(X_val_scaled))
    train_acc.append(t_acc)
    val_acc.append(v_acc)
    
    # Early Stopping Logic based on Validation Loss
    if v_loss < best_val_loss:
        best_val_loss = v_loss
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

# Final accuracy reporting
test_acc_nn = accuracy_score(y_test_class, nn.predict(X_test_scaled))
print(f"\nFinal Neural Network Test Accuracy: {test_acc_nn:.3f}")
joblib.dump(nn, 'models/nn_model.pkl')

print("\n-> Generating Required Neural Network Plots..")

# 1. Plot Training vs Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='red')
plt.title('Neural Network: Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Log Loss')
plt.legend()
plt.grid(True)
plt.savefig('plots/7_NN_Training_vs_Validation_Loss.png')
plt.close()

# 2. Plot Training vs Validation Accuracy
plt.figure(figsize=(8, 6))
plt.plot(train_acc, label='Training Accuracy', color='green')
plt.plot(val_acc, label='Validation Accuracy', color='orange')
plt.title('Neural Network: Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('plots/8_NN_Training_vs_Validation_Accuracy.png')
plt.close()

print("\nPipeline execution complete! Check your terminal for the reports and the 'plots/' folder for your graphs.")
print("\nAll phases complete! Pipeline Execution Successful!")
print("==================================================")