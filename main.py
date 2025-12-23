import cv2
import numpy as np

from motion import MotionDetector
from flow import OpticalFlowEstimator
from features import FeatureExtractor
from decision import IntentDecision
from visualize import draw_intent

MIN_CONTOUR_AREA = 800
COLLISION_AREA_THRESHOLD = 9000  # heuristic


def summarize_scene(intents):
    """
    Priority-based scene-level decision.
    Higher risk always overrides lower risk.
    """
    if "COLLISION_RISK" in intents:
        return " COLLISION RISK", (0, 0, 255)
    elif "APPROACHING" in intents:
        return "APPROACHING OBJECT", (0, 0, 255)
    elif "CROSSING" in intents:
        return "OBJECT CROSSING", (0, 255, 255)
    else:
        return "MOVING AWAY / SAFE", (0, 255, 0)


def main():
    cap = cv2.VideoCapture(0)

    motion_detector = MotionDetector()
    flow_estimator = OpticalFlowEstimator()
    feature_extractor = FeatureExtractor()
    intent_decider = IntentDecision()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        motion_mask = motion_detector.detect(frame)
        mag, ang = flow_estimator.compute(frame)

        # Collect intents for scene-level summary
        scene_intents = []

        if motion_mask is not None and mag is not None:
            contours, _ = cv2.findContours(
                motion_mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < MIN_CONTOUR_AREA:
                    continue

                x, y, w, h = cv2.boundingRect(cnt)

                # Feature extraction
                features = feature_extractor.extract(
                    (x, y, w, h),
                    mag,
                    ang
                )

                # ML intent prediction
                intent, confidence = intent_decider.predict(features)

                # Collision risk logic
                if intent == "APPROACHING" and (w * h) > COLLISION_AREA_THRESHOLD:
                    intent = "COLLISION_RISK"

                # Store intent for scene-level decision
                scene_intents.append(intent)

                # Object-level visualization (boxes remain unchanged)
                draw_intent(
                    frame,
                    (x, y, w, h),
                    intent,
                    confidence
                )

        # Scene-level conclusion (ONE clear message)
        if scene_intents:
            summary_text, summary_color = summarize_scene(scene_intents)

            cv2.putText(
                frame,
                summary_text,
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                summary_color,
                3
            )

        cv2.imshow("Predictive Motion Intent System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
