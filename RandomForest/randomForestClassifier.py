# Import necessary libraries
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Load and prepare the data
data = load_iris()
X, y = data.data, data.target

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Train the Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Step 3: Make predictions and evaluate the model
y_pred = rf.predict(X_test)

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Print Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Step 4: Visualize Feature Importance
feature_importance = pd.Series(rf.feature_importances_, index=data.feature_names)
feature_importance.sort_values(ascending=False).plot(kind='bar', title="Feature Importance")
plt.show()