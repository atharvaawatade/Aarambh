import cv2
import numpy as np
from collections import defaultdict, deque
from pytz import utc
from ultralytics import YOLO
import time
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])

generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 40,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
  model_name="gemini-1.5-pro",
  generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

yolo_model = YOLO("yolo11n.pt")
video_path = "C:/Users/Asus/Desktop/Carla/test1.mp4"
cap = cv2.VideoCapture(video_path)

track_history = defaultdict(lambda: [])
vehicle_speeds = defaultdict(lambda: [0, 0])
vehicle_distances = defaultdict(lambda: 0)
recent_events = deque(maxlen=10)

fcw_warning_distance = 20
fcw_danger_distance = 15
ldw_threshold = 0.5
fps = 30
focal_length = 800
camera_height = 1.5
time_to_brake_threshold = 2.0
fcw_confidence_threshold = 0.7
false_positive_ratio = 0.5

def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])*2 + (point1[1] - point2[1])*2)

def estimate_real_distance(bbox1, bbox2, focal_length, camera_height):
    if len(bbox1) >= 4 and len(bbox2) >= 4:
        h1, h2 = bbox1[3], bbox2[3]
        if h1 != h2 and h1 > 0 and h2 > 0:
            distance = (camera_height * focal_length) / (abs(h2 - h1))
            return distance
    return 0

def calculate_confidence(distance, relative_speed):
    if relative_speed > 0:
        time_to_collision = distance / relative_speed
        confidence = max(0, min(1, (time_to_brake_threshold - time_to_collision) / time_to_brake_threshold))
        return confidence
    return 0

def predict_forward_collision(track, vehicle_speeds, vehicle_distances, track_id, fcw_warning_distance, fcw_danger_distance):
    if len(track) > 1:
        distance = vehicle_distances[track_id]
        speed_diff = vehicle_speeds[track_id][0] - vehicle_speeds[track_id][1]
        confidence = calculate_confidence(distance, speed_diff)
        if confidence > fcw_confidence_threshold:
            if distance < fcw_danger_distance:
                return "Danger", confidence
            elif distance < fcw_warning_distance:
                return "Warning", confidence
    return None, 0

def predict_lane_departure(track, lane_width):
    if len(track) > 1:
        x1, y1 = track[-1]
        x2, y2 = track[-2]
        if abs(x2 - x1) > lane_width * ldw_threshold:
            return "Warning", 1.0
    return None, 0

def adjust_fcw_threshold(events):
    true_positives = sum(1 for event in events if event == "TP")
    false_positives = sum(1 for event in events if event == "FP")
    
    if true_positives > 0 and false_positives / true_positives > false_positive_ratio:
        return min(1.0, fcw_confidence_threshold + 0.05)
    elif "FN" in events:
        return max(0.5, fcw_confidence_threshold - 0.05)
    
    return fcw_confidence_threshold

def display_adas_warnings(frame, track_ids, vehicle_speeds, vehicle_distances, fcw_warning_distance, fcw_danger_distance, lane_width):
    global fcw_confidence_threshold
    for track_id in track_ids:
        track = track_history[track_id]
        
        fcw_status, fcw_confidence = predict_forward_collision(track, vehicle_speeds, vehicle_distances, track_id, fcw_warning_distance, fcw_danger_distance)
        if fcw_status == "Warning":
            cv2.putText(frame, "Forward Collision Warning!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            recent_events.append("TP")
        elif fcw_status == "Danger":
            cv2.putText(frame, "Forward Collision Danger!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            recent_events.append("TP")
        else:
            recent_events.append("FP")
        
        ldw_status, ldw_confidence = predict_lane_departure(track, lane_width)
        if ldw_status == "Warning":
            cv2.putText(frame, "Lane Departure Warning!", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            recent_events.append("LDW")
        
        if len(track) > 1:
            start_point = (int(track[-1][0]), int(track[-1][1]))
            end_point = (int(track[-2][0]), int(track[-2][1]))
            distance = estimate_real_distance(track[-1], track[-2], focal_length, camera_height)
            vehicle_distances[track_id] = distance

            if distance > 0:
                color = (230, 230, 230)
                if distance < fcw_danger_distance:
                    color = (0, 0, 255)
                elif distance < fcw_warning_distance:
                    color = (255, 255, 0)
                cv2.line(frame, start_point, end_point, color, 2)
                cv2.putText(frame, f"{distance:.2f}m", (end_point[0], end_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                speed = vehicle_speeds[track_id][0]
                cv2.putText(frame, f"{speed:.2f} km/h", (end_point[0], end_point[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    fcw_confidence_threshold = adjust_fcw_threshold(recent_events)

def main():
    start_time = time.time()
    frame_count = 0

    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = yolo_model.track(frame, persist=True)
            if results[0].boxes:
                boxes = results[0].boxes.xywh.cpu()
                track_ids = results[0].boxes.id
                if track_ids is not None:
                    track_ids = track_ids.int().cpu().tolist()
                else:
                    track_ids = []
                annotated_frame = results[0].plot()
                
                for box, track_id in zip(boxes, track_ids):
                    x, y, w, h = box
                    track = track_history[track_id]
                    track.append((float(x), float(y)))
                    if len(track) > 1:
                        dx = track[-1][0] - track[-2][0]
                        dy = track[-1][1] - track[-2][1]
                        if dx != 0 or dy != 0:
                            speed = np.sqrt(dx*2 + dy*2) * fps / 1000
                            vehicle_speeds[track_id][1] = vehicle_speeds[track_id][0]
                            vehicle_speeds[track_id][0] = speed
                    if len(track) > 30:
                        track.pop(0)
                    points = np.array(track, np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=2)

                display_adas_warnings(annotated_frame, track_ids, vehicle_speeds, vehicle_distances, fcw_warning_distance, fcw_danger_distance, lane_width=3.5)
                cv2.imshow("ADAS Demonstration", annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            frame_count += 1
        else:
            break
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Processed {frame_count} frames in {total_time:.2f} seconds")

if __name__ == "_main_":
    main()
