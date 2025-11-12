import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle
import os

# Load and clean dataset
df = pd.read_csv("datasets/CholeraSymptoms.csv")

# Drop columns not needed for prediction
df = df.drop(columns=["id", "name", "gender"])

# Rename target column to match model expectation
df = df.rename(columns={"target": "Cholera"})

# Split features and labels
X = df.drop("Cholera", axis=1)
y = df["Cholera"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train SVM model
model = SVC()
model.fit(X_train_scaled, y_train)

# Save model and scaler
os.makedirs("models", exist_ok=True)
with open("models/cholera_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Cholera model trained and saved with real dataset.")
