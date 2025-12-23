import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# -----------------------------
# 1. Create Training Data
# -----------------------------

# Feature format:
# [area, area_change, avg_magnitude, direction_variance]

X = np.array([
    # MOVING AWAY (0)
    [5000, -400, 0.5, 0.01],
    [4800, -300, 0.4, 0.02],
    [5200, -500, 0.6, 0.01],

    # CROSSING (1)
    [3000, 10, 1.2, 1.5],
    [3200, -20, 1.1, 1.8],
    [3100, 15, 1.3, 1.6],

    # APPROACHING (2)
    [2000, 300, 1.8, 0.02],
    [2300, 350, 2.0, 0.01],
    [2600, 400, 2.2, 0.01]
])

y = np.array([
    0, 0, 0,   # moving away
    1, 1, 1,   # crossing
    2, 2, 2    # approaching
])

# -----------------------------
# 2. Train / Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -----------------------------
# 3. Train Model
# -----------------------------

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate
# -----------------------------

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# -----------------------------
# 5. Save Model
# -----------------------------

joblib.dump(model, "model.pkl")
print("Model saved as model.pkl")
