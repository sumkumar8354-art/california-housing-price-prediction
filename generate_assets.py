import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import root_mean_squared_error, r2_score
import joblib

# Style setup
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Load dataset
data = fetch_california_housing(as_frame=True)
df = pd.concat([data.data, data.target.rename("HousePrice")], axis=1)

# Features and target
X = df.drop("HousePrice", axis=1)
y = df["HousePrice"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42)
}

results = {}
preds_dict = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    preds_dict[name] = preds
    rmse = root_mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    results[name] = {"RMSE": rmse, "R2 Score": r2}

results_df = pd.DataFrame(results).T

# Save models
joblib.dump(models["Decision Tree"], "best_model.joblib")
joblib.dump(scaler, "scaler.joblib")

# Directory for figures
os.makedirs("figures", exist_ok=True)

# 1. Model Metrics Comparison Bar Chart
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), dpi=300)
colors_rmse = ['#3498db', '#2980b9', '#27ae60']
colors_r2 = ['#e67e22', '#d35400', '#2ecc71']

model_names = list(results.keys())
rmse_vals = [results[m]["RMSE"] for m in model_names]
r2_vals = [results[m]["R2 Score"] for m in model_names]

bars1 = ax[0].bar(model_names, rmse_vals, color=colors_rmse, width=0.55, edgecolor='black', alpha=0.85)
ax[0].set_title("RMSE Comparison (Lower is Better)", fontsize=12, fontweight='bold', pad=10)
ax[0].set_ylabel("Root Mean Squared Error ($100k)", fontsize=10)
ax[0].set_ylim(0.65, 0.80)
for bar in bars1:
    yval = bar.get_height()
    ax[0].text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

bars2 = ax[1].bar(model_names, r2_vals, color=colors_r2, width=0.55, edgecolor='black', alpha=0.85)
ax[1].set_title("R² Score Comparison (Higher is Better)", fontsize=12, fontweight='bold', pad=10)
ax[1].set_ylabel("R² Score (Variance Explained)", fontsize=10)
ax[1].set_ylim(0.50, 0.65)
for bar in bars2:
    yval = bar.get_height()
    ax[1].text(bar.get_x() + bar.get_width()/2.0, yval + 0.003, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig("figures/model_comparison.png", bbox_inches='tight')
plt.close()

# 2. Actual vs Predicted Plot (Decision Tree vs Baseline)
fig, ax = plt.subplots(1, 2, figsize=(11, 5), dpi=300)

# Linear Regression
ax[0].scatter(y_test, preds_dict["Linear Regression"], alpha=0.25, color="#2980b9", s=15, edgecolors='none')
min_val = min(y_test.min(), preds_dict["Linear Regression"].min())
max_val = max(y_test.max(), preds_dict["Linear Regression"].max())
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2, label="Ideal Fit (y=x)")
ax[0].set_title("Baseline: Linear Regression\nRMSE: 0.7456 | R²: 0.5758", fontsize=11, fontweight='bold')
ax[0].set_xlabel("Actual House Prices ($100k)", fontsize=10)
ax[0].set_ylabel("Predicted House Prices ($100k)", fontsize=10)
ax[0].legend(loc="upper left")
ax[0].set_xlim(0, 5.5)
ax[0].set_ylim(-0.5, 5.5)

# Decision Tree
ax[1].scatter(y_test, preds_dict["Decision Tree"], alpha=0.25, color="#27ae60", s=15, edgecolors='none')
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", linewidth=2, label="Ideal Fit (y=x)")
ax[1].set_title("Best Model: Decision Tree (max_depth=5)\nRMSE: 0.7242 | R²: 0.5997", fontsize=11, fontweight='bold')
ax[1].set_xlabel("Actual House Prices ($100k)", fontsize=10)
ax[1].set_ylabel("Predicted House Prices ($100k)", fontsize=10)
ax[1].legend(loc="upper left")
ax[1].set_xlim(0, 5.5)
ax[1].set_ylim(-0.5, 5.5)

plt.tight_layout()
plt.savefig("figures/actual_vs_predicted.png", bbox_inches='tight')
plt.close()

# 3. Feature Importance Plot (Decision Tree)
dt_model = models["Decision Tree"]
feature_importances = pd.Series(dt_model.feature_importances_, index=X.columns).sort_values(ascending=True)

plt.figure(figsize=(7, 4), dpi=300)
colors_fi = ['#bdc3c7' if x < 0.05 else '#27ae60' for x in feature_importances]
bars = plt.barh(feature_importances.index, feature_importances.values, color=colors_fi, edgecolor='black', alpha=0.85)
plt.title("Decision Tree Feature Importances", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Relative Importance Score", fontsize=10)
for bar in bars:
    xval = bar.get_width()
    plt.text(xval + 0.01, bar.get_y() + bar.get_height()/2.0, f"{xval:.3f}", ha='left', va='center', fontsize=9, fontweight='bold')
plt.xlim(0, 0.75)
plt.tight_layout()
plt.savefig("figures/feature_importance.png", bbox_inches='tight')
plt.close()

# 4. Residuals Distribution
residuals = y_test - preds_dict["Decision Tree"]
plt.figure(figsize=(7, 4), dpi=300)
sns.histplot(residuals, kde=True, color='#2c3e50', bins=40, edgecolor='white', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Error Mean')
plt.title("Residual Distribution (Decision Tree)", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Residual Error (Actual - Predicted)", fontsize=10)
plt.ylabel("Density Frequency", fontsize=10)
plt.legend()
plt.tight_layout()
plt.savefig("figures/residual_distribution.png", bbox_inches='tight')
plt.close()

print("All figures successfully created in figures/ directory.")
