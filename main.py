from ultralytics import YOLO
import cv2
import tkinter as tk
import os
import numpy as np

model = YOLO('yolo11n.pt')  # load a pretrained YOLOv8n model
video = 'video.mp4'  # path to the input video file

# Open the video file
cap = cv2.VideoCapture(video)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Cannot open video file {video}")
    exit()

# Get FPS of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Get screen resolution
root = tk.Tk()
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
root.destroy()

# Get video dimensions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate scaling factor (80% of screen height)
scale_factor = (screen_height * 0.8) / frame_height
new_width = int(frame_width * scale_factor)
new_height = int(frame_height * scale_factor)

print(f"Video: {video}")
print(f"FPS: {fps}")
print(f"Original resolution: {frame_width}x{frame_height}")
print(f"Display resolution: {new_width}x{new_height}")

# Define line coordinates on original resolution
line_x1, line_y1 = 855, 438
line_x2, line_y2 = 1270, 440

# Scale line coordinates
scaled_line_x1 = int(line_x1)
scaled_line_y1 = int(line_y1)
scaled_line_x2 = int(line_x2)
scaled_line_y2 = int(line_y2)

print(f"Line coordinates (original): ({line_x1},{line_y1}) to ({line_x2},{line_y2})")
print(f"Line coordinates (scaled): ({scaled_line_x1},{scaled_line_y1}) to ({scaled_line_x2},{scaled_line_y2})")

# COCO class ID for cars (2=car, 3=motorcycle, 5=bus, 7=truck, 10)
vehicle_classes = [2, 3, 5, 7]

# Create directory for saved frames
save_dir = "detected_violations"
os.makedirs(save_dir, exist_ok=True)
frame_counter = 0

# Traffic light ROI coordinates
traffic_light_roi = {
    'x1': 1345,
    'y1': 146,
    'x2': 1400,
    'y2': 194
}


def line_intersects_box(x1, y1, x2, y2, box_x1, box_y1, box_x2, box_y2):
    """Check if line intersects with bounding box"""
    # Check if line passes through the box horizontally
    if box_x1 <= x1 <= box_x2 or box_x1 <= x2 <= box_x2 or (x1 <= box_x1 and x2 >= box_x2):
        # Check if line y-coordinates overlap with box y-coordinates
        line_y_at_box_x1 = y1 + (y2 - y1) * (box_x1 - x1) / (x2 - x1) if x2 != x1 else y1
        line_y_at_box_x2 = y1 + (y2 - y1) * (box_x2 - x1) / (x2 - x1) if x2 != x1 else y1
        
        if (box_y1 <= line_y_at_box_x1 <= box_y2 or box_y1 <= line_y_at_box_x2 <= box_y2 or
            (min(line_y_at_box_x1, line_y_at_box_x2) <= box_y1 and max(line_y_at_box_x1, line_y_at_box_x2) >= box_y2)):
            return True
    return False

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("End of video or error reading frame")
        break
        
    line_intersects = False
    
    # Run YOLO detection on the frame
    results = model(frame, verbose=False)
    
    # Filter detections to only show vehicles (cars, motorcycles, buses, trucks)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Only show vehicles
            if cls in vehicle_classes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Check if line intersects with this bounding box
                if line_intersects_box(scaled_line_x1, scaled_line_y1, scaled_line_x2, scaled_line_y2, x1, y1, x2, y2):
                    line_intersects = True
                    label = f'ALERT! {label}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                else:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                # Get class name
                class_name = model.names[cls]
                label = f'{class_name} {conf:.2f}'
                
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Resize frame to fit screen
    frame_resized = cv2.resize(frame, (new_width, new_height))
    
    # Draw the line on resized frame
    cv2.line(frame_resized, (scaled_line_x1, scaled_line_y1), 
             (scaled_line_x2, scaled_line_y2), (0, 0, 255), 5)
    
    # Draw traffic light ROI on resized frame
    roi_x1 = traffic_light_roi['x1']
    roi_y1 = traffic_light_roi['y1']
    roi_x2 = traffic_light_roi['x2']
    roi_y2 = traffic_light_roi['y2']
    cv2.rectangle(frame_resized, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)
    
    # Save frame if line intersects with any vehicle
    if line_intersects:
        frame_counter += 1
        filename = os.path.join(save_dir, f"violation_{frame_counter:04d}.jpg")
        cv2.imwrite(filename, frame_resized)
        print(f"Saved frame: {filename}")
    
    # Show the frame
    cv2.imshow('Car Detection', frame_resized)
    
    # Press 'q' to quit, 'p' to pause
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Exiting...")
        break
    elif key == ord('p'):
        print("Paused. Press any key to continue...")
        cv2.waitKey(0)

# Release resources
cap.release()
cv2.destroyAllWindows()