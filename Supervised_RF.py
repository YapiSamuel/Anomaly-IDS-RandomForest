import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Load dataset
df = pd.read_csv("UNSW_NB15_training-set.csv")

# Step 2: Drop non-numeric or irrelevant columns
columns_to_drop = ['id', 'attack_cat', 'proto', 'service', 'state']
df = df.drop(columns=columns_to_drop)

# Step 3: Drop missing values if any
df = df.dropna()

# Step 4: Split features and label
X = df.drop(columns=['label'])   # Features
y = df['label']                  # Target: 0 = normal, 1 = attack

# Step 5: Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Step 6: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate
print("\n✅ Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

print("🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

