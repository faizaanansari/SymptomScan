import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load the actual typhoid dataset
df = pd.read_csv("datasets/ThypoidDataSet.csv")

# Drop unnecessary columns
df = df.drop(columns=["id", "name", "gender", "age"])

# Rename target column for clarity
df = df.rename(columns={"target": "Typhoid"})

# Separate features and target
X = df.drop("Typhoid", axis=1)
y = df["Typhoid"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# KNN classifier
model = KNeighborsClassifier()
model.fit(X_train_scaled, y_train)

# Save model and scaler
os.makedirs("models", exist_ok=True)
with open("models/typhoid_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Typhoid model trained and saved using real dataset.")
