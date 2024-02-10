# video_stream.py

import cv2
from yolo_detector import YoloDetector

class VideoStream:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.detector = YoloDetector(model_name='yolov8n', device='cuda' if torch.cuda.is_available() else 'cpu')

    def start(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break  # Break if the video has ended or cannot be accessed

            # Perform detection on the frame
            detections = self.detector.detect(frame)

            # Optionally, visualize the detections on the frame
            self.visualize_detections(frame, detections)

            # Display the frame
            cv2.imshow('Parking Space Detection', frame)

            # Break the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

    @staticmethod
    def visualize_detections(frame, detections):
        for det in detections:
            bbox = det['bbox']
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            # Optionally, display the confidence score
            cv2.putText(frame, f"{det['confidence']:.2f}", (int(bbox[0]), int(bbox[1]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

