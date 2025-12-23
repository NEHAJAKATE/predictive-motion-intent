# Predictive Motion Intent Estimation (Real-Time)

A real-time computer vision system that predicts object motion intent —
MOVING_AWAY, CROSSING, APPROACHING, and COLLISION_RISK
using optical flow, temporal feature extraction, and lightweight machine learning.

🚀 Why this project?
Most vision systems detect objects.
This system goes one step further by predicting motion intent, enabling
early collision-risk detection — a key requirement in autonomous drones and robotics.

🧠 System Overview
- Real-time webcam input
- Motion detection via frame differencing
- Optical flow for direction and speed
- Interpretable feature extraction
- Logistic Regression for intent classification
- Scene-level decision aggregation for stable output

📊 Output
- Object-level bounding boxes with intent labels
- Single scene-level conclusion displayed on screen
- Collision risk highlighted clearly

🛠️ Tech Stack
- Python
- OpenCV
- NumPy
- scikit-learn

▶️ How to Run
bash
pip install -r requirements.txt
python train_model.py
python main.py
