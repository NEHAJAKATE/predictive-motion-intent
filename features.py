import numpy as np
from collections import deque

class FeatureExtractor:
    def __init__(self, window_size=5):
        self.prev_areas = deque(maxlen=window_size)
        self.prev_directions = deque(maxlen=window_size)

    def extract(self, bbox, magnitude, angle):
        x, y, w, h = bbox

        # Feature 1: Bounding box area
        area = w * h
        self.prev_areas.append(area)

        # Feature 2: Area change rate
        if len(self.prev_areas) > 1:
            area_change = self.prev_areas[-1] - self.prev_areas[-2]
        else:
            area_change = 0

        # Feature 3: Average motion magnitude inside bounding box
        mag_roi = magnitude[y:y+h, x:x+w]
        avg_magnitude = np.mean(mag_roi) if mag_roi.size > 0 else 0

        # Feature 4: Direction consistency
        ang_roi = angle[y:y+h, x:x+w]
        mean_direction = np.mean(ang_roi) if ang_roi.size > 0 else 0
        self.prev_directions.append(mean_direction)

        if len(self.prev_directions) > 1:
            direction_variance = np.var(self.prev_directions)
        else:
            direction_variance = 0

        return [
            area,
            area_change,
            avg_magnitude,
            direction_variance
        ]
