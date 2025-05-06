import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv("car_data.csv")  # Change filename if different
print(df.columns)

# Step 2: Add the Risk column based on year
df['Risk'] = df['Year'].apply(lambda x: 'High' if 2025 - x >= 10 else 'low')

# Separate features and label
X = df.drop(columns=["Risk"])  # 'Risk' is the target column
y = df["Risk"]

# Encode categorical features
label_encoders = {}
categorical_cols = X.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Also encode target variable
target_le = LabelEncoder()
y = target_le.fit_transform(y)
label_encoders['Risk'] = target_le  # Add target encoder too

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("\nModel Evaluation Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Save model, encoders, and scaler using pickle
with open("risk_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

    