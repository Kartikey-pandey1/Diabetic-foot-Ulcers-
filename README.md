# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler

# 1. Load the dataset (Assume you have a CSV with sensor data and labels)
# The dataset should contain sensor values and foot location data, with a label column for the output
# Example: columns might include 'sensor_value1', 'sensor_value2', 'foot_location', 'alert'
data = pd.read_csv("sensor_data.csv")

# 2. Data Preprocessing
# Split features (X) and labels (y)
X = data.drop(columns=['alert'])  # features (sensor data and foot location)
y = data['alert']  # target variable (alert/no alert)

# 3. Normalize the features (optional, but helps with many ML algorithms)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Initialize the Machine Learning Model
# Using Random Forest for classification, which can handle non-linear relationships well
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 6. Train the model
model.fit(X_train, y_train)

# 7. Make predictions on the test set
y_pred = model.predict(X_test)

# 8. Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Print classification report for detailed evaluation
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 9. Use the model to make predictions on new data (example)
# Let's assume you have new sensor data for which you need to predict if an alert should be triggered
new_data = pd.DataFrame({
    'sensor_value1': [45.6],
    'sensor_value2': [30.1],
    'foot_location': [2]  # example location
})
new_data_scaled = scaler.transform(new_data)
alert_prediction = model.predict(new_data_scaled)

if alert_prediction[0] == 1:
    print("Alert: Safety alarm triggered!")
else:
    print("No alert: All systems are safe.")
