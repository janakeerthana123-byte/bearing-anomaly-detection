import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# ----------------------------------------
# Load Dataset
# ----------------------------------------
df = pd.read_csv("bearing_binary_anomaly_dataset.csv")

X = df[[
    "RMS", "Peak", "Crest_Factor",
    "Kurtosis", "Skewness",
    "Spectral_Energy", "Max_FFT", "Dominant_Freq"
]]

y = df["Label"]

# ----------------------------------------
# Train-Test Split
# ----------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------------------
# Train Model
# ----------------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# ----------------------------------------
# Predictions
# ----------------------------------------
y_pred = model.predict(X_test)

# ----------------------------------------
# Evaluation
# ----------------------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
plt.imshow(cm)
plt.colorbar()
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

import joblib

# ----------------------------------------
# Save Trained Model
# ----------------------------------------
joblib.dump(model, "bearing_anomaly_model.pkl")

print("Model saved as 'bearing_anomaly_model.pkl'")