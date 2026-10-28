import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load the sample dataset
data = pd.read_csv("flood_data.csv")

# Inputs for the model
X = data[["rainfall", "temperature", "humidity"]]

# Correct answers the model will learn
y = data["risk"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# Create the machine learning model
model = DecisionTreeClassifier(max_depth=3, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("Model training completed!")
print("Test accuracy:", round(accuracy * 100, 2), "%")

# Save the trained model
joblib.dump(model, "flood_model.pkl")
print("Model saved successfully!")