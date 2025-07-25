# Visualize_RF.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For loading model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load the trained model
model = joblib.load("random_forest_model.pkl")

# Step 2: Load test data
df_test = pd.read_csv("UNSW_NB15_testing-set.csv")
df_test = df_test.drop(columns=['id', 'attack_cat', 'proto', 'service', 'state'])
df_test = df_test.dropna()

X_test = df_test.drop(columns=['label'])
y_test = df_test['label']

# ---- (a) FEATURE IMPORTANCE ----
importances = model.feature_importances_
features = X_test.columns

# Sort features by importance
sorted_idx = importances.argsort()
plt.figure(figsize=(10, 6))
plt.barh(features[sorted_idx], importances[sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()

# ---- (b) CONFUSION MATRIX ----
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Attack"], yticklabels=["Normal", "Attack"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

