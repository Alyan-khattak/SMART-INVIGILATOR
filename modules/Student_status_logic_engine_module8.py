from flask import Flask, request, jsonify
import cv2
import numpy as np
import sys
import os
from typing import List, Dict

# Add modules path dynamically
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))

from activity_classifier_module1 import ActivityClassifier, iou
from phone_detector_module4 import PhoneDetector
from student_detector_module6 import StudentDetector
from hand_detector_module5 import HandDetector
from screen_glow_module7 import ScreenGlowDetector
from preprocessing_module9 import VideoProcessor

app = Flask(__name__)

class StudentStatusLogicEngine:
    def __init__(self, screen_glow_threshold: float = 50, iou_threshold: float = 0.2):
        # Initialize the thresholds
        self.screen_glow_threshold = screen_glow_threshold
        self.iou_threshold = iou_threshold

        # Initialize other modules
        self.activity_classifier = ActivityClassifier(screen_glow_threshold=self.screen_glow_threshold)
        self.phone_detector = PhoneDetector(model_path="yolov8s.pt", confidence=0.45)
        self.student_detector = StudentDetector(model_path="yolov8n.pt", conf_threshold=0.6)
        self.hand_detector = HandDetector()
        self.screen_glow_detector = ScreenGlowDetector()
        self.video_processor = VideoProcessor(threaded=True, output_format="rgb_normalized")

    def classify_student_status(self, student_data: List[Dict], phone_detections: List[Dict], 
                                hands_data: List[Dict], head_poses: List[Dict]) -> Dict[int, str]:
        student_status = {}

        # Capture and preprocess frame (dummy frame for testing)
        ret, frame = self.video_processor.read_frame()
        if not ret or frame is None:
            print("[ERROR] Failed to capture frame! Using dummy frame for testing.")
            frame = self.generate_dummy_frame()  # Use the dummy frame if capture failed

        # Preprocess the frame
        processed_frame = self.video_processor.preprocess(frame)

        for student in student_data:
            student_id = student["id"]
            student_box = (student["x1"], student["y1"], student["x2"], student["y2"])
            status = "No Phone"  # Default status

            # Check if a phone is close to the student (using IoU threshold)
            active_phone = None
            for phone in phone_detections:
                phone_box = phone["bbox"]
                overlap = iou(student_box, phone_box)
                if overlap > self.iou_threshold:
                    active_phone = phone
                    break

            if active_phone:
                hand_near_phone = False
                for hand in hands_data:
                    hand_box = (hand["x1"], hand["y1"], hand["x2"], hand["y2"])
                    if iou(hand_box, active_phone["bbox"]) > 0.1:
                        hand_near_phone = True
                        break

                screen_glow = False
                if "brightness" in active_phone and active_phone["brightness"] > self.screen_glow_threshold:
                    screen_glow = True

                head_towards_phone = False
                for hp in head_poses:
                    if hp["label"] in ["Looking Down", "Looking Right", "Looking Left"]:
                        head_towards_phone = True

                if screen_glow and head_towards_phone and hand_near_phone:
                    status = "Actively Using Phone"
                elif hand_near_phone:
                    status = "Just Holding Phone"

            student_status[student_id] = status

        return student_status

    def visualize(self, frame, student_data: List[Dict], status_results: Dict[int, str], phone_detections: List[Dict] = []) -> np.ndarray:
        for student in student_data:
            x1, y1, x2, y2 = student["x1"], student["y1"], student["x2"], student["y2"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            status = status_results.get(student["id"], "No Phone")
            cv2.putText(frame, status, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        for phone in phone_detections:
            x1, y1, x2, y2 = phone["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return frame

    def generate_dummy_frame(self):
        """
        Generates a dummy frame for testing purposes. 
        The frame contains white rectangles to simulate students.
        """
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Adding dummy students as white rectangles
        cv2.rectangle(frame, (100, 100), (200, 300), (255, 255, 255), 2)  # Student 1
        cv2.rectangle(frame, (300, 100), (400, 300), (255, 255, 255), 2)  # Student 2
        return frame


@app.route('/api/status/all', methods=['GET'])
def get_all_status():
    # Sample dummy input data (Replace with actual detection module data)
    student_data = [
        {"id": 1, "x1": 100, "y1": 100, "x2": 200, "y2": 300},
        {"id": 2, "x1": 300, "y1": 100, "x2": 400, "y2": 300}
    ]
    
    phone_detections = [
        {"bbox": (110, 200, 150, 250), "confidence": 0.95, "brightness": 120}
    ]
    
    hands_data = [
        {"id": 1, "x1": 105, "y1": 180, "x2": 155, "y2": 260}
    ]
    
    head_poses = [
        {"pitch": 20, "yaw": 0, "roll": 0, "label": "Looking Down"}
    ]
    
    # Initialize the status logic engine
    engine = StudentStatusLogicEngine(screen_glow_threshold=50)
    
    # Generate a dummy frame for testing
    frame = engine.generate_dummy_frame()
    
    # Classify the student status
    results = engine.classify_student_status(student_data, phone_detections, hands_data, head_poses)
    
    # Visualize the results on the frame
    frame = engine.visualize(frame, student_data, results, phone_detections)

    # Save or show the frame (Optional: for debugging)
    cv2.imwrite("output_frame.jpg", frame)  # Save the frame as an image
    cv2.imshow("Test Frame", frame)         # Optionally display the frame for debugging
    
    return jsonify(results)


if __name__ == "__main__":
    app.run(debug=True)
