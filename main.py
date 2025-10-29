from ultralytics import YOLO
import cv2
import os
import numpy as np

model = YOLO('yolo11n.pt')  # Load a pretrained YOLOv8n model
video = 'video.mp4'         # Path to the input video file
DEBUG = 0                   # Debug flag

# Open the video file
cap = cv2.VideoCapture(video)

# Create window and set to fullscreen
cv2.namedWindow('Car Detection', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Car Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Cannot open video file {video}")
    exit()

# Get FPS of the video
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define line coordinates on original resolution
line_x1, line_y1 = 1085, 563
line_x2, line_y2 = 1566, 575

# Scale line coordinates
scaled_line_x1 = int(line_x1)
scaled_line_y1 = int(line_y1)
scaled_line_x2 = int(line_x2)
scaled_line_y2 = int(line_y2)

# COCO class ID for cars (2=car, 3=motorcycle, 5=bus, 7=truck)
vehicle_classes = [2, 3, 5, 7]

# Create directory for saved frames
save_dir = "detected_violations"
os.makedirs(save_dir, exist_ok=True)
frame_counter = 0

# Traffic light ROI coordinates
traffic_light_roi = {
    'x1': 1699,
    'y1': 192,
    'x2': 1748,
    'y2': 241
}

def detect_traffic_light_color(frame, roi):
    """Detect if red light is on in the traffic light ROI"""
    # Extract ROI
    roi_frame = frame[roi['y1']:roi['y2'], roi['x1']:roi['x2']]
    
    if roi_frame.size == 0:
        return False, 0
    
    # Convert to HSV
    hsv = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
    
    # Define HSV range for red light (based on the image)
    # Hue: 15-35 (red range)
    # Saturation: 100-255 (highly saturated)
    # Value: 150-255 (bright)
    lower = np.array([15, 100, 150])
    upper = np.array([35, 255, 255])
    
    # Create mask
    mask = cv2.inRange(hsv, lower, upper)
    
    # Calculate percentage of matching pixels
    total_pixels = mask.size
    matching_pixels = np.count_nonzero(mask)
    percentage = (matching_pixels / total_pixels) * 100
    
    # Light is detected if more than 10% of pixels match
    light_detected = percentage > 10
    
    return light_detected, percentage

def line_intersects_box(x1, y1, x2, y2, box_x1, box_y1, box_x2, box_y2):
    """Check if line intersects with bounding box"""
    if x1 is None or x2 is None or y1 is None or y2 is None or box_x1 is None or box_x2 is None or box_y1 is None or box_y2 is None:
        return False
    
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

    # Detect traffic light color
    light_on, light_percentage = detect_traffic_light_color(frame, traffic_light_roi)
    
    # Run YOLO detection on the frame
    results =  model(frame, verbose=False)
    cv2.putText(frame, f'FPS: {fps}', (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 204, 255), 2)
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
                
                # Get class name
                class_name = model.names[cls]
                label = f'{class_name} {conf:.2f}'
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                # Draw label text
                cv2.putText(frame, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Draw the line on original frame
    cv2.line(frame, (scaled_line_x1, scaled_line_y1), 
             (scaled_line_x2, scaled_line_y2), (0, 0, 255), 5)
    
    
    # Draw traffic light ROI on original frame
    roi_x1 = traffic_light_roi['x1']
    roi_y1 = traffic_light_roi['y1']
    roi_x2 = traffic_light_roi['x2']
    roi_y2 = traffic_light_roi['y2']
    roi_color = (0, 255, 0) if light_on else (0, 0, 255)
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), roi_color, 2)

    # Display traffic light status
    light_status = f"Red Light: {'ON' if light_on else 'OFF'}"
    if DEBUG: print(f"Light status: {light_status} ({light_percentage:.1f}%)")

    cv2.putText(frame, light_status, (roi_x1, roi_y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, roi_color, 2)

    line_intersects = line_intersects_box(scaled_line_x1, scaled_line_y1, scaled_line_x2, scaled_line_y2, x1, y1, x2, y2)

    # Save frame if line intersects with any vehicle AND light is on (red light violation)
    if line_intersects and light_on:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        frame_counter += 1
        filename = os.path.join(save_dir, f"violation_{frame_counter:04d}.jpg")

        cv2.imwrite(filename, frame)
        print(f"RED LIGHT VIOLATION! Saved frame: {filename}")
    
    # Show the frame
    cv2.imshow('Car Detection', frame)

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