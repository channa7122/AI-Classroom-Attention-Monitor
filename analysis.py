import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def analyze_latest_session():
    # 1. Find the latest CSV file in the 'logs' folder
    list_of_files = glob.glob('logs/*.csv') 
    if not list_of_files:
        print("No logs found!")
        return
    
    # Get the most recent file
    latest_file = max(list_of_files, key=os.path.getctime)
    print(f"Analyzing: {latest_file}")
    
    # 2. Load Data
    df = pd.read_csv(latest_file)
    
    # Clean data (remove spaces in column names if any)
    df.columns = df.columns.str.strip()
    
    if df.empty:
        print("Log file is empty.")
        return

    # 3. Create a Dashboard Image
    plt.figure(figsize=(14, 8))
    plt.suptitle(f"Session Analysis Report: {os.path.basename(latest_file)}", fontsize=16)

    # --- CHART 1: Focus Score Over Time ---
    plt.subplot(2, 2, 1) # Top Left
    plt.plot(df.index, df['Focus_Score'], color='green', linewidth=2)
    plt.axhline(y=50, color='r', linestyle='--', alpha=0.5, label="Distraction Threshold")
    plt.title("Attention Span Over Time")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Focus Score (0-100)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # --- CHART 2: Emotion Distribution ---
    plt.subplot(2, 2, 2) # Top Right
    emotion_counts = df['Emotion'].value_counts()
    plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, colors=plt.cm.Paired.colors)
    plt.title("Emotional State Breakdown")

    # --- CHART 3: Drowsiness / Eyes Closed ---
    plt.subplot(2, 2, 3) # Bottom Left
    eyes_counts = df['Eyes_Closed'].value_counts()
    labels = ['Open', 'Closed']
    # Handle case where eyes were never closed
    if len(eyes_counts) == 1:
        # If only False (Open) exists
        values = [eyes_counts.get(False, 0), 0]
    else:
        values = [eyes_counts.get(False, 0), eyes_counts.get(True, 0)]
        
    plt.bar(labels, values, color=['blue', 'orange'])
    plt.title("Eye Contact Analysis")
    plt.ylabel("Count (Seconds)")

    # --- SUMMARY TEXT ---
    plt.subplot(2, 2, 4) # Bottom Right
    plt.axis('off') # Hide axes
    
    avg_score = df['Focus_Score'].mean()
    most_common_emotion = df['Emotion'].mode()[0]
    total_time = len(df) # Since we log every 1 second
    
    summary_text = f"""
    SESSION SUMMARY
    ----------------------------
    Total Duration:   {total_time} seconds
    Average Focus:    {avg_score:.1f}%
    Dominant Mood:    {most_common_emotion}
    
    Performance Rating:
    {'⭐⭐⭐⭐⭐ (Excellent)' if avg_score > 80 else '⭐⭐⭐ (Good)' if avg_score > 50 else '⭐ (Needs Improvement)'}
    """
    plt.text(0.1, 0.5, summary_text, fontsize=14, fontfamily='monospace')

    # 4. Show and Save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("latest_report.png")
    print("Report saved as 'latest_report.png'")
    plt.show()

if __name__ == "__main__":
    analyze_latest_session()