import pandas as pd
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import IsolationForest
import numpy as np


# Step 1: Load the dataset
df = pd.read_csv("UNSW_NB15_training-set.csv")

# Step 2: Print basic info
print("Shape of original data:", df.shape)
print("Column names:", df.columns.tolist())
print("\nFirst 5 rows of data:\n", df.head())

# Step 3: Drop unnecessary columns
columns_to_drop = ['id', 'label', 'attack_cat', 'proto', 'service', 'state']
df_cleaned = df.drop(columns=columns_to_drop)

# Step 4: Drop missing values (if any)
df_cleaned = df_cleaned.dropna()

# Step 5: Normalize features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_cleaned)

# Step 6: Confirm
print("\nCleaned and scaled data shape:", X_scaled.shape)

# Step 7: Train Isolation Forest model
print("\nTraining Isolation Forest...")
model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
model.fit(X_scaled)

# Step 8: Predict anomalies (-1 = anomaly, 1 = normal)
predictions = model.predict(X_scaled)

# Convert to binary: 1 = anomaly, 0 = normal
anomaly_labels = np.where(predictions == -1, 1, 0)

# Step 9: Count and display results
anomalies = np.sum(anomaly_labels)
normals = len(anomaly_labels) - anomalies

print(f"\nDetected anomalies: {anomalies}")
print(f"Detected normal entries: {normals}")

# Step 10: Add anomaly labels back to the full DataFrame
df["anomaly"] = anomaly_labels

# Step 11: Extract and save anomalies to a CSV file
anomalies_df = df[df["anomaly"] == 1]

# Show a few rows of anomalies
print("\nSample detected anomalies:")
print(anomalies_df.head())

# Save to file
anomalies_df.to_csv("detected_anomalies.csv", index=False)
print("\nAnomalies saved to 'detected_anomalies.csv'")

from sklearn.metrics import classification_report, confusion_matrix

# Step 12: Evaluate the model using true labels
print("\n🔍 Evaluating model performance...")

# Reload true labels from original CSV (they are not in df_cleaned)
true_labels = pd.read_csv("UNSW_NB15_training-set.csv")["label"].values  # 0 = normal, 1 = attack

# Get predicted anomaly labels (already mapped: 1 = anomaly, 0 = normal)
predicted_labels = anomaly_labels  # from earlier

# Convert predicted anomalies to match true label meanings
# Anomaly = 1 = attack → keep it
# Normal = 0 → label 0
# So prediction is aligned with true labels

# Print classification report
print("\n📊 Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=["Normal", "Attack"]))

# Print confusion matrix
print("\n🧮 Confusion Matrix:")
print(confusion_matrix(true_labels, predicted_labels))

