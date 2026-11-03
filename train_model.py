import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load Data
data = pd.read_csv('gesture_data.csv', header=None)

X = data.iloc[:, :-1].values  # 63 features
y = data.iloc[:, -1].values   # Class labels

# 2. Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 4. Evaluate Accuracy
y_pred = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
print(classification_report(y_test, y_pred))

# 5. Save Model
joblib.dump(clf, 'gesture_model.pkl')
print("Model saved to gesture_model.pkl")