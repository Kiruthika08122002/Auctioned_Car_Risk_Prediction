from data import load_data

data = load_data()

from preprocessing import preprocess_data
from model import train_model

# Step 1: Load data
data = load_data()

# Step 2: Preprocess data
processed_data, label_encoders, scaler = preprocess_data(data)

# Step 3: Separate features and target
X = processed_data.drop('Risk', axis=1)
y = label_encoders['Risk'].transform(data['Risk'])

# Step 4: Train the model
model = train_model(X,y)

# Print the first few rows
print(data.head())

# Check if Risk column was created
print("\nRisk column value counts:")
print(data['Risk'].value_counts())

# Step 2: Explore Target Variable
print("Target Variable Distribution:\n", data['Risk'].value_counts())

# Visualize the target variable
import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='Risk', data=data)
plt.title('Distribution of Risk')
plt.xlabel('Risk (Low, Medium, High)')
plt.ylabel('Count')
plt.show()

# Step 3: Exploratory Data Analysis (EDA)

# (a) Check for missing values
print("Missing Values:\n", data.isnull().sum())

# (b) Summary statistics
print("Summary Statistics:\n", data.describe())

# (c) Visualize Selling Price by Risk category
sns.boxplot(x='Risk', y='Selling_Price', data=data)
plt.title('Selling Price by Risk Category')
plt.show()

# Step 4: Train/Test Split
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train a Random Forest Classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Make Predictions
y_pred = clf.predict(X_test)

# Step 7: Evaluate the Model
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Step 8: Save model and preprocessors
import joblib

joblib.dump(model, 'risk_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and preprocessors saved successfully.")