# utilities.py

import cv2
import numpy as np

def draw_bounding_boxes(frame, detections, color=(0, 255, 0), thickness=2):
    for det in detections:
        bbox = det['bbox']
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, thickness)

def count_available_spaces(occupancy):
    return occupancy.count(False)

def apply_gaussian_blur(frame, kernel_size=(5, 5)):
    return cv2.GaussianBlur(frame, kernel_size, 0)

def convert_to_grayscale(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
