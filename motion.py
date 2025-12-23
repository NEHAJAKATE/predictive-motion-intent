import cv2
import numpy as np

class MotionDetector:
    def __init__(self, threshold=25):
        self.previous_frame = None
        self.threshold = threshold

    def preprocess(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        return blur

    def detect(self, frame):
        processed = self.preprocess(frame)

        if self.previous_frame is None:
            self.previous_frame = processed
            return None

        frame_diff = cv2.absdiff(self.previous_frame, processed)
        _, motion_mask = cv2.threshold(
            frame_diff, self.threshold, 255, cv2.THRESH_BINARY
        )

        self.previous_frame = processed
        return motion_mask

cv2.waitKey(0)