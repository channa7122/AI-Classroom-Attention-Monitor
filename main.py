import cv2
import os
import time
import csv
from datetime import datetime
from collections import deque
from src.detector import AttentionDetector

# --- CONFIGURATION ---
GRAPH_HEIGHT = 150
GRAPH_WIDTH = 640
MAX_HISTORY = 100 
LOG_INTERVAL = 1.0 # Save data every 1 second (to keep file size small)

def setup_logger():
    """Creates a CSV file for the session"""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    # Filename: logs/session_2023-10-27_15-30-00.csv
    filename = f"logs/session_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
    file = open(filename, "w", newline="")
    writer = csv.writer(file)
    # Write Header
    writer.writerow(["Timestamp", "Name", "Emotion", "Focus_Score", "Eyes_Closed", "Alert"])
    return file, writer

def draw_graph(frame, history, current_score):
    h, w, _ = frame.shape
    # Draw Dashboard Background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - GRAPH_HEIGHT), (w, h), (20, 20, 20), -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    # Grid & Score
    cv2.line(frame, (0, h - GRAPH_HEIGHT//2), (w, h - GRAPH_HEIGHT//2), (100, 100, 100), 1)
    
    # Plot Line
    if len(history) > 1:
        points = []
        for i, score in enumerate(history):
            x_pos = int((i / MAX_HISTORY) * w)
            y_pos = int(h - (score / 100) * GRAPH_HEIGHT)
            points.append((x_pos, y_pos))
        for i in range(1, len(points)):
            cv2.line(frame, points[i-1], points[i], (0, 255, 255), 2)

    score_color = (0, 255, 0) if current_score > 50 else (0, 0, 255)
    cv2.putText(frame, f"FOCUS: {int(current_score)}%", (w - 180, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, score_color, 2)

def main():
    detector = AttentionDetector()
    cap = cv2.VideoCapture(0)
    
    # Setup Logging
    log_file, log_writer = setup_logger()
    last_log_time = time.time()
    
    score_history = deque([50] * MAX_HISTORY, maxlen=MAX_HISTORY)
    current_score = 50.0 
    
    print(f"Session Started. Logging to {log_file.name}")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # 1. AI Analysis
        info = detector.process_frame(frame)
        x, y, face_w, face_h = info["box"]
        name = info.get("name", "Unknown")
        emotion = info.get("emotion", "Scanning")
        
        # 2. Score Logic
        target_step = -0.5 # Default decay
        if face_w > 0: 
            if info["eyes_closed"]:
                target_step = -5
            else:
                target_step = 1
                if emotion in ["happy", "surprise", "neutral"]:
                    target_step += 0.5
        
        current_score = max(0, min(100, current_score + target_step))
        score_history.append(current_score)
        
        # 3. SAVE TO CSV (Once per second)
        if time.time() - last_log_time > LOG_INTERVAL:
            timestamp = datetime.now().strftime("%H:%M:%S")
            # Save: Time, Name, Emotion, Score, Eyes Closed?, Alert?
            log_writer.writerow([timestamp, name, emotion, int(current_score), info["eyes_closed"], info["alert"]])
            last_log_time = time.time()
        
        # 4. Visualization
        draw_graph(frame, score_history, current_score)
        
        if face_w > 0:
            color = (0, 255, 0)
            if info["alert"]: color = (0, 0, 255)
            elif info["eyes_closed"]: color = (0, 255, 255)
            
            cv2.rectangle(frame, (x, y), (x+face_w, y+face_h), color, 2)
            cv2.putText(frame, f"{name} ({emotion})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            if info["alert"]:
                 cv2.putText(frame, "WAKE UP!", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

        cv2.imshow('AI Attention Monitor', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    # Clean up
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()
    print("Session Saved. Goodbye!")

if __name__ == "__main__":
    main()