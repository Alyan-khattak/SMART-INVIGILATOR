# Integrated Version

import cv2
import mediapipe as mp
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modules.preprocessing_module9 import VideoProcessor
  # Module 9

# ============================================================
# Module 5 – Hand Detection
# ============================================================

class HandDetector:
    """
    Detects hands in real-time video frames using MediaPipe.
    Assigns simple IDs to hands for basic tracking across frames.
    """

    def __init__(self,
                 max_hands=8,
                 detection_confidence=0.5, #minimum probability for detecting hand.
                 tracking_confidence=0.5): #Minimum confidence to continue tracking a hand across frames
        """
        Parameters:
        -----------
        max_hands : int
            Maximum number of hands to detect in a frame.

        detection_confidence : float
            Minimum confidence for detection.

        tracking_confidence : float
            Minimum confidence for tracking landmarks.
        """
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils  #Utility functions to draw landmarks and lines
        self.hands = self.mp_hands.Hands(   # intialize mediapipe model
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )

        self.hand_id_counter = 0
        self.hand_ids = {}  # Simple mapping: landmark center -> ID

    # ------------------------------------------------------------
    # Process frame and detect hands
    # ------------------------------------------------------------
    def process_frame(self, frame):
        """
        Detects hands and draws bounding boxes.
        Returns:
            - processed_frame : Frame with hand boxes drawn
            - hands_data      : List of hand coordinates + ID
        """
        hands_data = []

        if frame is None:
            return frame, hands_data

        # Convert BGR → RGB for MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:  #results.multi_hand_landmarks is a list each item repsnt 1 detected hand 
            for hand_landmarks in results.multi_hand_landmarks:  #hand_landmarks contains 21 landmarks of one hand.
                # Get bounding box from landmarks
                x_min = y_min = 1e9
                x_max = y_max = 0
                h, w, _ = frame.shape
                for lm in hand_landmarks.landmark: # lm store each landmark point  # cnvrt landmark to pixel
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x_min, x), min(y_min, y)
                    x_max, y_max = max(x_max, x), max(y_max, y)

                # Assign a simple ID based on position (can be improved)
                hand_center = ((x_min + x_max)//2, (y_min + y_max)//2)
                hand_id = self.hand_id_counter
                self.hand_id_counter += 1

                # Store data
                hand_info = {
                    "id": hand_id,
                    "x1": x_min,
                    "y1": y_min,
                    "x2": x_max,
                    "y2": y_max,
                    "center": hand_center
                }
                hands_data.append(hand_info)

                # Draw bounding box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                              (255, 0, 0), 2)
                cv2.putText(frame, f"Hand {hand_id}", (x_min, y_min - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                # Draw landmarks
                self.mp_draw.draw_landmarks(frame, hand_landmarks,
                                            self.mp_hands.HAND_CONNECTIONS)

        return frame, hands_data


# ============================================================
# Main runner for live webcam detection
# ============================================================
def run_hand_detection():
    """
    Uses Module 9 VideoProcessor to capture webcam frames
    and runs hand detection in real-time.
    """
    processor = VideoProcessor(threaded=True, output_format="bgr")
    processor.capture_video(0)  # 0 = default webcam

    detector = HandDetector(max_hands=6)

    while True:
        ret, frame = processor.read_frame()
        if not ret or frame is None:
            continue

        # Preprocess frame
        processed_frame = processor.preprocess(frame)

        # Detect hands
        output_frame, hands_data = detector.process_frame(processed_frame)

        # Display FPS
        cv2.putText(output_frame, f"FPS: {processor.fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Show output
        cv2.imshow("Module 8 - Hand Detection", output_frame)

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    processor.stop()
    cv2.destroyAllWindows()


# ============================================================
# Entry point
# ============================================================
if __name__ == "__main__":
    run_hand_detection()