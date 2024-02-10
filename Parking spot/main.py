# main.py

from video_stream import VideoStream
from parking_space import ParkingSpaceDetector
import cv2
import numpy as np

def main(video_source=0, layout_file='data/parking_lot_layout.json'):
    # Initialize the parking space detector with the layout file
    parking_detector = ParkingSpaceDetector(layout_file)

    # Initialize the video stream
    video_stream = VideoStream(source=video_source)

    # Start the video stream and process frames
    cap = video_stream.cap
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit if the video has ended or cannot be accessed

        # Perform vehicle detection on the current frame
        detections = video_stream.detector.detect(frame)

        # Check the occupancy of each parking space based on detections
        occupancy = parking_detector.check_occupancy(detections)

        # Visualize the results
        visualize(frame, parking_detector.parking_spaces, occupancy)

        # Display the frame
        cv2.imshow('Parking Space Occupancy', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

def visualize(frame, parking_spaces, occupancy):
    for space, is_occupied in zip(parking_spaces, occupancy):
        color = (0, 255, 0) if not is_occupied else (0, 0, 255)
        cv2.rectangle(frame, (int(space[0]), int(space[1])), (int(space[2]), int(space[3])), color, 2)
        cv2.putText(frame, 'Occupied' if is_occupied else 'Free', 
                    (int(space[0]), int(space[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

if __name__ == '__main__':
    main()
