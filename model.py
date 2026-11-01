
import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("Social_Network_Ads.csv")

# Remove unnecessary column
data = data.drop("User ID", axis=1)

# Separate input and output
X = data.drop("Purchased", axis=1)
y = data["Purchased"]

# Convert Gender into numerical columns
X = pd.get_dummies(X, columns=["Gender"])

# Store feature names and order
feature_columns = X.columns.tolist()

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Evaluate model
y_pred = model.predict(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(feature_columns, open("feature_columns.pkl", "wb"))

print("Model and preprocessing files saved successfully!")
