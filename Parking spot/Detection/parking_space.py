# parking_space.py

import json

class ParkingSpaceDetector:
    def __init__(self, layout_file='data/parking_lot_layout.json'):
        self.parking_spaces = self.load_layout(layout_file)

    @staticmethod
    def load_layout(layout_file):
        with open(layout_file) as file:
            layout = json.load(file)
        return layout['spaces']

    def check_occupancy(self, detections):
        occupancy = [False] * len(self.parking_spaces)
        for i, space in enumerate(self.parking_spaces):
            for det in detections:
                if self.is_overlap(space, det['bbox']):
                    occupancy[i] = True
                    break
        return occupancy

    @staticmethod
    def is_overlap(space, bbox):
        # Extract coordinates
        sx1, sy1, sx2, sy2 = space
        bx1, by1, bx2, by2 = bbox
        
        # Check for overlap
        return not (bx2 < sx1 or bx1 > sx2 or by2 < sy1 or by1 > sy2)

