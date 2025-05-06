import matplotlib.pyplot as plt
import seaborn as sns
from data import load_data

# Load the dataset
data = load_data()

# Basic information
print(data.info())
print(data.describe())
print(data.isnull().sum())

# Selling Price Distribution
plt.figure(figsize=(6,4))
sns.histplot(data['Selling_Price'], bins=30, kde=True)
plt.title('Selling Price Distribution')
plt.show()

# Count of cars by Fuel Type
sns.countplot(data=data, x='Fuel_Type')
plt.title('Fuel Type Count')
plt.show()

# Risk distribution
sns.countplot(data=data, x='Risk')
plt.title('Car Risk Levels')
plt.show()

# Year vs. Selling Price
plt.figure(figsize=(6,4))
sns.boxplot(x='Year', y='Selling_Price', data=data)
plt.xticks(rotation=45)
plt.title('Year vs Selling Price')
plt.show()