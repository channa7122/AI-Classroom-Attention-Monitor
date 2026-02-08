import cv2
from deepface import DeepFace
import time
import os
from collections import deque
import statistics

class AttentionDetector:
    def __init__(self):
        # 1. Standard Detectors 
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # 2. Tracking Variables
        self.eyes_closed_start_time = None
        self.CLOSED_TIME_THRESH = 3
        
        # 3. Identity & Performance Variables
        self.frame_count = 0
        self.current_name = "Unknown"
        self.db_path = "assets/known_faces" 
        
        # --- NEW: STABILIZATION (The Fix) ---
        # We remember the last 10 emotions to smooth out the jitter
        self.emotion_history = deque(maxlen=10) 
        self.current_emotion = "Scanning..."
        
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)

    def process_frame(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        data = {
            "name": self.current_name,
            "emotion": self.current_emotion, # Use the STABLE emotion
            "eyes_closed": False,
            "alert": False,
            "box": (0, 0, 0, 0)
        }

        if len(faces) == 0:
            return data

        # Get the largest face
        (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
        data["box"] = (x, y, w, h)
        
        # --- 1. NAME RECOGNITION (Every 30 frames) ---
        if self.frame_count % 30 == 0:
            try:
                face_img = frame[y:y+h, x:x+w]
                results = DeepFace.find(img_path=face_img, db_path=self.db_path, 
                                      enforce_detection=False, silent=True)
                
                if len(results) > 0 and not results[0].empty:
                    full_path = results[0].iloc[0]['identity']
                    filename = os.path.basename(full_path)
                    self.current_name = os.path.splitext(filename)[0]
                else:
                    self.current_name = "Unknown"
            except:
                pass

        # Update name
        data["name"] = self.current_name

        # --- 2. EYES (Drowsiness) ---
        roi_gray = gray[y:y+h, x:x+w]
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(20, 20))
        
        if len(eyes) == 0:
            if self.eyes_closed_start_time is None:
                self.eyes_closed_start_time = time.time()
            if (time.time() - self.eyes_closed_start_time) > self.CLOSED_TIME_THRESH:
                data["alert"] = True
            data["eyes_closed"] = True
        else:
            self.eyes_closed_start_time = None
            data["eyes_closed"] = False

        # --- 3. EMOTION (Stabilized) ---
        # We check emotion every 5 frames to keep it fast
        if self.frame_count % 5 == 0:
            try:
                face_img = frame[y:y+h, x:x+w]
                # 'opencv' backend is faster/smoother than default
                analysis = DeepFace.analyze(face_img, actions=['emotion'], 
                                          enforce_detection=False, detector_backend='opencv', silent=True)
                
                raw_emotion = analysis[0]['dominant_emotion']
                self.emotion_history.append(raw_emotion)
                
                # --- THE MAGIC FIX: Pick the most common emotion in history ---
                try:
                    self.current_emotion = statistics.mode(self.emotion_history)
                except:
                    # If there's a tie (e.g., 5 Happy, 5 Neutral), just keep the last one
                    self.current_emotion = raw_emotion
                    
            except:
                pass
        
        # Update the data with the stable emotion
        data["emotion"] = self.current_emotion
            
        return data
