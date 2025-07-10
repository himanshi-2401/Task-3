# Step 1: Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load Dataset
df = pd.read_csv("breast cancer.csv")
print("✅ Dataset Loaded")
print(df.head())

# Step 3: Preprocessing
df.dropna(inplace=True)  # Remove nulls
if 'id' in df.columns:
    df.drop(columns=['id'], inplace=True)  # Drop ID column
if 'Unnamed: 32' in df.columns:
    df.drop(columns=['Unnamed: 32'], inplace=True)  # Remove any extra columns

# Step 4: Label Encoding (B = 0, M = 1)
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

# Step 5: Features & Target
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Step 6: Normalize Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("✅ Data Split Complete")

# Step 8: Train KNN Classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
print("✅ KNN Model Trained (k=5)")

# Step 9: Evaluate Model
y_pred = knn.predict(X_test)

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("🔍 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
