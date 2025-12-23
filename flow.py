import cv2
import numpy as np

class OpticalFlowEstimator:
    def __init__(self):
        self.prev_gray = None

    def compute(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return None, None

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )

        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        self.prev_gray = gray
        return magnitude, angle
