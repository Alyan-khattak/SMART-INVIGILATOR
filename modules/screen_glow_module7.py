# # Ubaid Version
#
# from ultralytics import YOLO
# import cv2
# import datetime
# import os
# import numpy as np
# import winsound  # for sound alert on Windows
#
# # Folder to save cropped phone images
# save_folder = "Phone_Crops"
# os.makedirs(save_folder, exist_ok=True)
#
# # Load YOLOv8 pretrained model
# model = YOLO("yolov8n.pt")  # Nano model, fast for live detection
#
#
# # Sound alert function
# def sound_alert():
#     duration = 500  # milliseconds
#     freq = 1000  # Hz
#     winsound.Beep(freq, duration)
#
#
# # Phone detection function (frame input)
# def detect_phones(frame):
#     results = model(frame)
#     detected = False
#
#     for r in results:
#         if hasattr(r, 'boxes'):
#             for box in r.boxes:
#                 cls = int(box.cls[0])
#                 conf = float(box.conf[0])
#
#                 # COCO class index for 'cell phone' is 67
#                 if cls == 67 and conf > 0.3:
#                     x1, y1, x2, y2 = map(int, box.xyxy[0])
#                     phone_crop = frame[y1:y2, x1:x2]
#
#                     # Draw rectangle
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
#
#                     # Brightness
#                     brightness = np.mean(cv2.cvtColor(phone_crop, cv2.COLOR_BGR2GRAY))
#                     cv2.putText(frame, f"Brightness: {brightness:.2f}", (x1, y1 - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#                     # Save crop
#                     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#                     filename = os.path.join(save_folder, f"phone_{timestamp}.jpg")
#                     cv2.imwrite(filename, phone_crop)
#                     print(f"Phone detected! Saved: {filename}, Brightness: {brightness:.2f}")
#
#                     # Sound alert
#                     sound_alert()
#                     detected = True
#     return frame, detected
#
#
# # Main menu
# def main_menu():
#     while True:
#         print("\n--- Module 7 Phone Detection ---")
#         print("1. Live webcam detection")
#         print("2. Analyze from photo")
#         print("3. Exit")
#         choice = input("Enter your choice (1/2/3): ")
#
#         if choice == '1':
#             cap = cv2.VideoCapture(0)
#             print("Press 'q' to quit live detection.")
#             while True:
#                 ret, frame = cap.read()
#                 if not ret:
#                     print("Failed to grab frame")
#                     break
#                 frame, _ = detect_phones(frame)
#                 cv2.imshow("Live Detection", frame)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break
#             cap.release()
#             cv2.destroyAllWindows()
#
#         elif choice == '2':
#             img_path = input("Enter image path: ")
#             if not os.path.isfile(img_path):
#                 print("File not found!")
#                 continue
#             frame = cv2.imread(img_path)
#             frame, detected = detect_phones(frame)
#             cv2.imshow("Photo Analysis", frame)
#             print("Press any key to close image.")
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
#
#         elif choice == '3':
#             print("Exiting program.")
#             break
#         else:
#             print("Invalid choice, try again.")
#
#
# # Run the program
# if __name__ == "__main__":
#     main_menu()

# XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX

# Integrated Version

import cv2
import datetime
import os
import numpy as np
import winsound  # for audio alerts on Windows
from ultralytics import YOLO
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.preprocessing_module9 import VideoProcessor
  # Module 9

# Folder to save cropped phone images
save_folder = "Phone_Crops"
os.makedirs(save_folder, exist_ok=True)

# Load YOLOv8 pretrained model (for phone detection)
model = YOLO("yolov8n.pt")  # Nano model for speed


# -----------------------------
# Audio alert function
# -----------------------------
def sound_alert():
    duration = 500  # milliseconds
    freq = 1000     # Hz
    winsound.Beep(freq, duration)


# -----------------------------
# Phone detection / screen glow detection
# -----------------------------
def detect_phones(frame):
    results = model(frame)
    detected = False

    for r in results:
        if hasattr(r, "boxes") and r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])

                if cls == 67 and conf > 0.3:  # COCO 'cell phone'
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    phone_crop = frame[y1:y2, x1:x2]

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Screen glow / brightness calculation
                    brightness = np.mean(cv2.cvtColor(phone_crop, cv2.COLOR_BGR2GRAY))
                    cv2.putText(frame, f"Brightness: {brightness:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Save crop with timestamp
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(save_folder, f"phone_{timestamp}.jpg")
                    cv2.imwrite(filename, phone_crop)
                    print(f"Phone detected! Saved: {filename}, Brightness: {brightness:.2f}")

                    # Play alert sound (non-blocking)
                    sound_alert()
                    detected = True

    return frame, detected


# -----------------------------
# Integrated runner with Module 9
# -----------------------------
def run_phone_detection_with_preprocessing(use_webcam=True, video_path="video.mp4"):
    # Initialize Module 9 video processor
    processor = VideoProcessor(threaded=True, output_format="bgr")  # YOLO expects BGR
    if use_webcam:
        processor.capture_video(0)
    else:
        processor.capture_video(video_path)

    while True:
        ret, frame = processor.read_frame()
        if not ret or frame is None:
            continue

        # Preprocess frame (resize + optional enhancement)
        processed_frame = processor.preprocess(frame)

        # Detect phones and screen glow
        output_frame, _ = detect_phones(processed_frame)

        # Display FPS
        cv2.putText(output_frame, f"FPS: {processor.fps:.1f}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("Phone Detection with Audio Alert", output_frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    processor.stop()
    cv2.destroyAllWindows()


# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    run_phone_detection_with_preprocessing(use_webcam=True)