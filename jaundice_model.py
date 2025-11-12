import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load real jaundice dataset
df = pd.read_csv("datasets/JaundiceDataSet.csv")

# Drop non-feature columns
df = df.drop(columns=["id", "name", "gender", "age"])

# Rename target column
df = df.rename(columns={"target": "Jaundice"})

# Feature matrix and label
X = df.drop("Jaundice", axis=1)
y = df["Jaundice"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Save the model and scaler
os.makedirs("models", exist_ok=True)
with open("models/jaundice_model.pkl", "wb") as f:
    pickle.dump((model, scaler), f)

print("✅ Jaundice model trained and saved using real dataset.")
