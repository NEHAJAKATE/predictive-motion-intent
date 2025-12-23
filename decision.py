import joblib
import numpy as np
from collections import deque

class IntentDecision:
    def __init__(self, model_path="D:\\predictive_CV\\model.pkl", window_size=5):
        self.model = joblib.load(model_path)
        self.intent_window = deque(maxlen=window_size)

        self.intent_map = {
            0: "MOVING_AWAY",
            1: "CROSSING",
            2: "APPROACHING"
        }

    def predict(self, features):
        features = np.array(features).reshape(1, -1)

        # ML prediction
        pred_class = self.model.predict(features)[0]
        confidence = np.max(self.model.predict_proba(features))

        # Store for temporal smoothing
        self.intent_window.append(pred_class)

        # Majority voting for stability
        stable_class = max(
            set(self.intent_window),
            key=self.intent_window.count
        )

        intent_label = self.intent_map[stable_class]

        return intent_label, confidence
