# yolo_detector.py

import torch
from ultralytics import YOLO  # This might need adjustment based on the exact package structure

class YoloDetector:
    def __init__(self, model_name='yolov8n', device='cuda'):
        self.device = device
        self.model = YOLO(model_name).to(self.device)

    def detect(self, frame):
        # Ensure the frame is in the right format and dimension for YOLO
        frame = [frame]
        
        # Perform detection
        results = self.model(frame)
        
        # Parse results
        detections = self.parse_results(results)
        
        return detections

    @staticmethod
    def parse_results(results):
        # Extract detection results
        detections = []
        for det in results.xyxy[0]:  # results.xyxy[0] contains bbox, confidence, and class
            x1, y1, x2, y2, conf, cls = det
            detections.append({
                'bbox': (x1.item(), y1.item(), x2.item(), y2.item()),
                'confidence': conf.item(),
                'class': cls.item()  # Assuming vehicle class is what we're interested in
            })
        return detections
