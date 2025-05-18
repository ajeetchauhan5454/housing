import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# Load the dataset
data = pd.read_csv("housing.csv")

# Features and target
X = data[["Area", "Bedrooms", "Bathrooms", "Stories", "Parking"]]
y = data["Price"]

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "house_price_model.pkl")

# Predict for sample input
sample = pd.DataFrame([[3500, 4, 2, 2, 1]], columns=["Area", "Bedrooms", "Bathrooms", "Stories", "Parking"])
predicted_price = model.predict(sample)

print(f"Predicted House Price: ₹ {predicted_price[0]:,.2f}")
