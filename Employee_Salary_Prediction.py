# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 2: Load the dataset
df = pd.read_csv("adult 3.csv")

# Step 3: Check the data
print("First 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# Step 4: Drop rows with missing values
df = df.dropna()

# Step 5: Encode categorical features
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Step 6: Define features (X) and target (y)
target_column = "income" if "income" in df.columns else "salary"
X = df.drop(columns=[target_column])
y = df[target_column]

# Step 7: Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Model evaluation
y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Plot 1 – Feature Importance (Bar Plot)
importances = model.feature_importances_
feature_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
bars = plt.bar(feature_df['Feature'], feature_df['Importance'], color='skyblue', edgecolor='black')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 0.005, f'{height:.2f}', ha='center', fontsize=8)
plt.title("Feature Importance - Random Forest")
plt.ylabel("Importance Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 11: Plot 2 – Income Distribution (Pie Chart)
income_counts = df[target_column].value_counts()
labels = [label_encoders[target_column].inverse_transform([i])[0] for i in income_counts.index]
colors = ['lightgreen', 'lightcoral']
plt.figure(figsize=(6, 6))
plt.pie(income_counts, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
plt.title("Income Class Distribution")
plt.axis('equal')
plt.show()

# Step 12: Plot 3 – Heatmap (Correlation Matrix)
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True)
plt.title("Correlation Matrix Heatmap")
plt.tight_layout()
plt.show()
