import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load training dataset
df = pd.read_csv("UNSW_NB15_training-set.csv")

# Step 2: Drop non-numeric or irrelevant columns
columns_to_drop = ['id', 'attack_cat', 'proto', 'service', 'state']
df = df.drop(columns=columns_to_drop)

# Step 3: Drop missing values if any
df = df.dropna()

# Step 4: Split features and label
X_train = df.drop(columns=['label'])   # Features
y_train = df['label']                  # Target

# Step 5: Load actual testing data and preprocess it the same way
df_test = pd.read_csv("UNSW_NB15_testing-set.csv")
df_test = df_test.drop(columns=columns_to_drop)
df_test = df_test.dropna()

X_test = df_test.drop(columns=['label'])
y_test = df_test['label']

# Step 6: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions on real testing set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
print("\n✅ Classification Report (Test Set):\n")
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 9: Save the trained model to a file
import joblib
joblib.dump(model, "random_forest_model.pkl")
print("✅ Model saved as random_forest_model.pkl")

