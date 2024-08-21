import os
import random
import cv2
from object_tracking import Object_tracking
import json

# Specify the name of the video file to be processed
video = "v8.mp4"

# Define the detection threshold for object detection (confidence score above which detections are considered valid)
detection_threshold = 0.5

# Set the paths for the input video and output video
video_path = os.path.join('.', 'Data', video)  # Path to the input video
video_out_path = os.path.join('.', 'out', video)  # Path to save the processed output video

# Open the input video file for reading
cap = cv2.VideoCapture(video_path)

# Read the first frame from the video
ret, frame = cap.read()

# Set up the video writer to save the output video, maintaining the original video's frame size and frame rate
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'mp4v'),cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

# Initialize object tracking with a YOLOv8 model for detection and a MARS model for tracking
object_tracking = Object_tracking('Models/yolov8n.pt', 'Models/mars-small128.pb')

# Define coordinates for two lines in the frame, which might be used for tracking objects crossing these lines
line1={
    "point_1":{
                "x":940,
                "y":370},
    "point_2":{
                "x":370,
                "y":370}
}
line2={
    "point_1":{
                "x":1244,
                "y":550},
    "point_2":{
                "x":30,
                "y":550}
}

# Define an offset used when cropping/snipping images of detected objects
snip_offset = 10

# Initialize a list of random colors for drawing bounding boxes for tracked objects
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

# Initialize data structures to store tracking information
list_objects_Tracking = list()  # List to hold tracking data
objects_Tracking = dict()  # Dictionary to hold tracking data by object ID

# Initialize counters for frames and snipped images
frame_counter = 0
snip_counter = 0

# Main loop to process the video frame by frame
while ret:
    # Make a copy of the current frame for potential use later
    original_frame = frame.copy()
    # Draw the defined lines on the frame (used to check if objects cross certain areas)
    cv2.line(frame,(line1["point_1"]["x"], line1["point_1"]["y"]),
                    (line1["point_2"]["x"], line1["point_2"]["y"]), (255, 0, 0), 2)
    cv2.line(frame,(line2["point_1"]["x"],line2["point_1"]["y"]),
                    (line2["point_2"]["x"], line2["point_2"]["y"]), (255, 0, 0), 2)

    # Perform object detection on the current frame, looking for specified classes (e.g., "cars", "truck")
    object_tracking.detect(frame, detect_class=["car", "truck"])
    # Perform object tracking based on the detections
    object_tracking.track(frame)
    # Loop through each tracked object
    for track in object_tracking.tracks:
        bbox = track.bbox  # Get the bounding box coordinates of the tracked object
        x1, y1, x2, y2 = bbox
        track_id = track.track_id  # Get the unique ID of the tracked object
        obj_center_x, obj_center_y = int((x1 + x2) / 2), int((y1 + y2) / 2)  # Calculate the center of the bounding box
        # Check if object is within the area of interest
        if line1['point_1']['y'] < obj_center_y < line2['point_1']['y']:
            # Prepare tracking data
            objects_Tracking[f"ID_{track_id}"] = {f"frame_{frame_counter}": {'bbox': [x1, y1, x2, y2]}}
            # If the object is near the second line, snip (crop) a portion of the frame around the object
            if abs(obj_center_y - line2['point_1']['y']) <= 200:
                snip_img = original_frame[int(y1) + snip_offset:int(y2) + snip_offset,int(x1) + snip_offset:int(x2) + snip_offset]
                snip_img = cv2.resize(snip_img, (340, 340))  # Resize the snip image
                objects_Tracking[f"ID_{track_id}"]['snip_img'] = snip_img.tolist()  # Store the snip image in the tracking data
            # Draw a circle at the object's center and a rectangle around the object on the frame
            cv2.circle(frame, (obj_center_x, obj_center_y), 4, (0, 0, 255), 2, -1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)),colors[track_id % len(colors)], 3)
            # Label the object with its ID
            cv2.putText(frame, f"ID {track_id}", (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,0.8, (0, 0, 255))
        # Uncomment these lines to show the frame and write it to the output video file
        cv2.imshow("RGB", frame)
        # cap_out.write(frame)
    if cv2.waitKey(0)&0xFF==27:
        break
    # Check if we've reached the 100th frame, and if so, break out of the loop
    # if frame_counter == 100:
    #     break

    # Read the next frame from the video
    ret, frame = cap.read()

    # Increment the frame counter
    frame_counter += 1

# Release the video capture and writer resources
cap.release()
cap_out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

# Save the tracking data to a JSON file
with open("out/objects_Tracking.json", 'a') as f_json:
    json.dump(objects_Tracking, f_json, indent=4)
