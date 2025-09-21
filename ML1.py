import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Expanded dataset (replace with real housing dataset if available)
data = {
    "square_feet": [1500, 1800, 2400, 3000, 3500, 1200, 2200, 2800, 3200, 4000],
    "bedrooms":    [3,    4,    3,    5,    4,    2,    3,    4,    5,    5],
    "bathrooms":   [2,    2,    3,    4,    3,    1,    2,    3,    4,    4],
    "price":       [300000, 400000, 500000, 600000, 650000,
                    200000, 450000, 550000, 620000, 700000]
}

df = pd.DataFrame(data)

# Features and target
X = df[["square_feet", "bedrooms", "bathrooms"]]
y = df["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Example prediction (wrapped in DataFrame to avoid warning)
example_house = pd.DataFrame([[2000, 3, 2]], columns=["square_feet", "bedrooms", "bathrooms"])
predicted_price = model.predict(example_house)
print("Predicted Price for Example House:", predicted_price[0])

# Visualization: Actual vs Predicted
plt.scatter(y_test, y_pred, color="blue")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Ideal line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.show()