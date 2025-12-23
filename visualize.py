import cv2

INTENT_COLORS = {
    "MOVING_AWAY": (255, 255, 0),     # Cyan
    "CROSSING": (0, 255, 255),        # Yellow
    "APPROACHING": (0, 0, 255),       # Red
    "COLLISION_RISK": (0, 0, 255)     # Red (thick)
}

def draw_intent(frame, bbox, intent, confidence):
    x, y, w, h = bbox

    color = INTENT_COLORS.get(intent, (255, 255, 255))
    thickness = 3 if intent == "COLLISION_RISK" else 2

    # Bounding box
    cv2.rectangle(
        frame,
        (x, y),
        (x + w, y + h),
        color,
        thickness
    )

    # Label text
    label = f"{intent} ({confidence:.2f})"
    cv2.putText(
        frame,
        label,
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2
    )
