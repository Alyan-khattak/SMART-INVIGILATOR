# ==================================================
# main.py
# Entry point of AI Project – Mobile Detector
# ==================================================


### THIS MAIN HAS 7 MODULE MODULES INTEGRATED TOGETHER IN A SINGLE PIPELINE ### EXCLUDING MODIULE 8
### ==================================================          

import sys
import os
import cv2

# --------------------------------------------------
# Add PROJECT ROOT to Python Path
# --------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --------------------------------------------------
# Import Project Modules
# --------------------------------------------------
from modules.preprocessing_module9 import VideoProcessor
from modules.phone_detector_module4 import PhoneDetector
from modules.student_detector_module6 import StudentDetector
from modules.head_pose_module3 import HeadPoseDetector
from modules.hand_detector_module5 import HandDetector
from modules.activity_classifier_module1 import ActivityClassifier
from modules.screen_glow_module7 import detect_phones  # function-based

# --------------------------------------------------
# Main Function
# --------------------------------------------------
def main():
    print("\n===================================")
    print(" AI PROJECT – MOBILE DETECTOR START ")
    print("===================================\n")

    # -------------------------------
    # Initialize Video Source
    # -------------------------------
    video_processor = VideoProcessor(threaded=True, output_format="bgr")  # BGR for OpenCV & YOLO
    video_processor.capture_video(0)  # 0 = webcam

    # -------------------------------
    # Initialize All Modules
    # -------------------------------
    phone_detector = PhoneDetector()
    student_detector = StudentDetector()
    head_pose_detector = HeadPoseDetector()
    hand_detector = HandDetector()
    activity_classifier = ActivityClassifier()

    print("[INFO] All modules initialized successfully.\n")

    # -------------------------------
    # Main Processing Loop
    # -------------------------------
    while True:
        ret, frame = video_processor.read_frame()
        if not ret or frame is None:
            continue

        # Preprocess frame
        processed_frame = video_processor.preprocess(frame)

        # ---------------------------
        # Run Detection Modules
        # ---------------------------
        processed_frame, students = student_detector.process_frame(processed_frame)
        phones = phone_detector.detect(processed_frame)
        processed_frame = phone_detector.draw(processed_frame.copy(), phones)
        processed_frame, hands = hand_detector.process_frame(processed_frame)
        processed_frame, head_poses = head_pose_detector.process_frame(processed_frame)

        # ---------------------------
        # Screen Glow / Phone Detection
        # ---------------------------
        processed_frame, screen_glow_detected = detect_phones(processed_frame)

        # ---------------------------
        # Activity Classification
        # ---------------------------
        activity_results = activity_classifier.classify(
            students,    # student data
            phones,      # phone detections
            hands,       # hands
            head_poses   # head poses
        )

        # ---------------------------
        # Visualize Activities & Phones
        # ---------------------------
        processed_frame = activity_classifier.visualize(
            processed_frame,
            students,
            activity_results,
            phones
        )

        # ---------------------------
        # Display Output using OpenCV
        # ---------------------------
        cv2.imshow("AI Project - Mobile Detector", processed_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # -------------------------------
    # Cleanup
    # -------------------------------
    video_processor.stop()
    cv2.destroyAllWindows()
    print("\n[INFO] System closed successfully.")


# --------------------------------------------------
# Run Program
# --------------------------------------------------
if __name__ == "__main__":
    main()






# ==================================================
# main.py
# Entry point of AI Project – Mobile Detector with Flask API
# ==================================================

# import sys
# import os
# import cv2
# import numpy as np
# from flask import Flask, request, jsonify
# from typing import List, Dict

# # --------------------------------------------------
# # Add PROJECT ROOT and modules path dynamically
# # --------------------------------------------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# MODULES_PATH = os.path.join(PROJECT_ROOT, 'modules')
# if MODULES_PATH not in sys.path:
#     sys.path.insert(0, MODULES_PATH)

# # --------------------------------------------------
# # Import Project Modules
# # --------------------------------------------------
# from preprocessing_module9 import VideoProcessor
# from phone_detector_module4 import PhoneDetector
# from student_detector_module6 import StudentDetector
# from head_pose_module3 import HeadPoseDetector
# from hand_detector_module5 import HandDetector
# from activity_classifier_module1 import ActivityClassifier, iou
# from screen_glow_module7 import detect_phones  # function-based

# # --------------------------------------------------
# # Initialize Flask App
# # --------------------------------------------------
# app = Flask(__name__)

# # --------------------------------------------------
# # Student Status Logic Engine (for Flask)
# # --------------------------------------------------
# class StudentStatusLogicEngine:
#     def __init__(self, screen_glow_threshold: float = 50, iou_threshold: float = 0.2):
#         self.screen_glow_threshold = screen_glow_threshold
#         self.iou_threshold = iou_threshold

#         # Initialize modules
#         self.activity_classifier = ActivityClassifier(screen_glow_threshold=self.screen_glow_threshold)
#         self.phone_detector = PhoneDetector(model_path="yolov8s.pt", confidence=0.45)
#         self.student_detector = StudentDetector(model_path="yolov8n.pt", conf_threshold=0.6)
#         self.hand_detector = HandDetector()
#         self.video_processor = VideoProcessor(threaded=True, output_format="rgb_normalized")

#     def classify_student_status(self, student_data: List[Dict], phone_detections: List[Dict], 
#                                 hands_data: List[Dict], head_poses: List[Dict]) -> Dict[int, str]:
#         student_status = {}

#         # Capture dummy frame (for testing)
#         ret, frame = self.video_processor.read_frame()
#         if not ret or frame is None:
#             frame = self.generate_dummy_frame()

#         processed_frame = self.video_processor.preprocess(frame)

#         for student in student_data:
#             student_id = student["id"]
#             student_box = (student["x1"], student["y1"], student["x2"], student["y2"])
#             status = "No Phone"

#             # Check if a phone overlaps student (IoU)
#             active_phone = None
#             for phone in phone_detections:
#                 phone_box = phone["bbox"]
#                 overlap = iou(student_box, phone_box)
#                 if overlap > self.iou_threshold:
#                     active_phone = phone
#                     break

#             if active_phone:
#                 hand_near_phone = any(iou(
#                     (hand["x1"], hand["y1"], hand["x2"], hand["y2"]),
#                     active_phone["bbox"]) > 0.1 for hand in hands_data)

#                 screen_glow = active_phone.get("brightness", 0) > self.screen_glow_threshold

#                 head_towards_phone = any(hp["label"] in ["Looking Down", "Looking Right", "Looking Left"]
#                                          for hp in head_poses)

#                 if screen_glow and head_towards_phone and hand_near_phone:
#                     status = "Actively Using Phone"
#                 elif hand_near_phone:
#                     status = "Just Holding Phone"

#             student_status[student_id] = status

#         return student_status

#     def visualize(self, frame, student_data: List[Dict], status_results: Dict[int, str], phone_detections: List[Dict] = []):
#         for student in student_data:
#             x1, y1, x2, y2 = student["x1"], student["y1"], student["x2"], student["y2"]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#             status = status_results.get(student["id"], "No Phone")
#             cv2.putText(frame, status, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         for phone in phone_detections:
#             x1, y1, x2, y2 = phone["bbox"]
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

#         return frame

#     def generate_dummy_frame(self):
#         frame = np.zeros((480, 640, 3), dtype=np.uint8)
#         cv2.rectangle(frame, (100, 100), (200, 300), (255, 255, 255), 2)  # Student 1
#         cv2.rectangle(frame, (300, 100), (400, 300), (255, 255, 255), 2)  # Student 2
#         return frame

# # --------------------------------------------------
# # Flask Route for Testing
# # --------------------------------------------------
# @app.route('/api/status/all', methods=['GET'])
# def get_all_status():
#     student_data = [
#         {"id": 1, "x1": 100, "y1": 100, "x2": 200, "y2": 300},
#         {"id": 2, "x1": 300, "y1": 100, "x2": 400, "y2": 300}
#     ]
#     phone_detections = [
#         {"bbox": (110, 200, 150, 250), "confidence": 0.95, "brightness": 120}
#     ]
#     hands_data = [
#         {"id": 1, "x1": 105, "y1": 180, "x2": 155, "y2": 260}
#     ]
#     head_poses = [
#         {"pitch": 20, "yaw": 0, "roll": 0, "label": "Looking Down"}
#     ]

#     engine = StudentStatusLogicEngine(screen_glow_threshold=50)
#     frame = engine.generate_dummy_frame()
#     results = engine.classify_student_status(student_data, phone_detections, hands_data, head_poses)
#     frame = engine.visualize(frame, student_data, results, phone_detections)

#     cv2.imwrite("output_frame.jpg", frame)
#     cv2.imshow("Test Frame", frame)

#     return jsonify(results)

# # --------------------------------------------------
# # Run Flask App
# # --------------------------------------------------
# if __name__ == "__main__":
#     app.run(debug=True)






###======================================================================================
#### AROOSA'S MODULE INTIGRATION
###======================================================================================
#### THIS MAIN HAS 8 MODULES MODULES INTEGRATED TOGETHER IN A SINGLE PIPELINE ####
## BUT IT WILL NOT SHOW OUT HERE AS OUT WILL BE SHOWN ON WEB ######



# ==================================================
# main.py
# Flask + Live Webcam Integration for AI Project
# ==================================================

# import sys
# import os
# import cv2
# import threading
# from flask import Flask, jsonify

# # --------------------------------------------------
# # Add PROJECT ROOT to Python Path
# # --------------------------------------------------
# PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# # --------------------------------------------------
# # Import Project Modules
# # --------------------------------------------------
# from modules.preprocessing_module9 import VideoProcessor
# from modules.phone_detector_module4 import PhoneDetector
# from modules.student_detector_module6 import StudentDetector
# from modules.head_pose_module3 import HeadPoseDetector
# from modules.hand_detector_module5 import HandDetector
# from modules.activity_classifier_module1 import ActivityClassifier

# # Flask App
# app = Flask(__name__)

# # Global objects for thread-safe access
# latest_frame = None
# latest_status = {}
# lock = threading.Lock()

# # --------------------------------------------------
# # Initialize Modules
# # --------------------------------------------------
# video_processor = VideoProcessor(threaded=True, output_format="bgr")
# video_processor.capture_video(0)

# phone_detector = PhoneDetector()
# student_detector = StudentDetector()
# head_pose_detector = HeadPoseDetector()
# hand_detector = HandDetector()
# activity_classifier = ActivityClassifier()

# # --------------------------------------------------
# # Function to continuously process frames
# # --------------------------------------------------
# def process_frames():
#     global latest_frame, latest_status

#     while True:
#         ret, frame = video_processor.read_frame()
#         if not ret or frame is None:
#             continue

#         # Preprocess frame
#         processed_frame = video_processor.preprocess(frame)

#         # Run detection modules
#         processed_frame, students = student_detector.process_frame(processed_frame)
#         phones = phone_detector.detect(processed_frame)
#         processed_frame = phone_detector.draw(processed_frame.copy(), phones)
#         processed_frame, hands = hand_detector.process_frame(processed_frame)
#         processed_frame, head_poses = head_pose_detector.process_frame(processed_frame)

#         # Activity classification
#         activity_results = activity_classifier.classify(
#             students,
#             phones,
#             hands,
#             head_poses
#         )

#         # Visualize results on frame
#         processed_frame = activity_classifier.visualize(
#             processed_frame,
#             students,
#             activity_results,
#             phones
#         )

#         # Update global variables thread-safely
#         with lock:
#             latest_frame = processed_frame.copy()
#             latest_status = activity_results

# # Start frame processing in background thread
# threading.Thread(target=process_frames, daemon=True).start()

# # --------------------------------------------------
# # Flask API Route
# # --------------------------------------------------
# @app.route('/api/status/all', methods=['GET'])
# def get_all_status():
#     global latest_status
#     with lock:
#         if latest_status:
#             return jsonify(latest_status)
#         else:
#             return jsonify({"message": "No data yet. Wait a moment..."}), 202

# # --------------------------------------------------
# # Run Flask App
# # --------------------------------------------------
# if __name__ == "__main__":
#     print("[INFO] Flask API running with live webcam processing...")
#     app.run(debug=True)
