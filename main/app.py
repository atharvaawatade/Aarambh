OPENAI_API_KEY = ""  # Add your OpenAI API key here

from concurrent.futures import ThreadPoolExecutor
import glob
import os
import sys
import random
import time
import numpy as np
import cv2
import torch
from collections import defaultdict, deque
import carla
from queue import Queue
from queue import Empty
import weakref
from openai import OpenAI
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path


# YOLO setup for model
YOLO_MODEL = torch.hub.load('ultralytics/ultralytics', 'yolov11n', pretrained=True)
YOLO_MODEL.eval()

# ADAS parameters 
FCW_WARNING_DISTANCE = 20.0    
FCW_DANGER_DISTANCE = 15.0     
LDW_THRESHOLD = 0.5           
FPS = 30                      
CAMERA_HEIGHT = 1.5           
TIME_TO_BRAKE_THRESHOLD = 2.0  
FCW_CONFIDENCE_THRESHOLD = 0.7  
FALSE_POSITIVE_RATIO = 0.5    

# Advanced parameters
ADAPTIVE_THRESHOLD_LEARNING_RATE = 0.1
MAX_TRACKING_HISTORY = 30
COLLISION_PREDICTION_HORIZON = 3.0 
WEATHER_IMPACT_FACTOR = 0.8 

@dataclass
class WeatherInfo:
    fog_density: float
    precipitation: float
    visibility: str

class SensorSyncManager:
    """Manages synchronization of multiple sensors"""
    def __init__(self, timeout: float = 0.1):
        self.timeout = timeout
        self.sensor_data = {}
        self.ready = False
    
    def update(self, sensor_id: str, data: any) -> bool:
        self.sensor_data[sensor_id] = data
        return len(self.sensor_data) == 3  # camera, lidar, imu
    
    def get_data(self) -> Optional[Dict]:
        if len(self.sensor_data) == 3:
            data = self.sensor_data.copy()
            self.sensor_data.clear()
            return data
        return None

class VehicleState:
    """Data class to store vehicle state information"""
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    heading: float
    timestamp: float

class ADASLogger:
    """Custom logger for ADAS events and metrics"""
    def __init__(self, log_file: str = "adas_logs.txt"):
        self.logger = logging.getLogger("ADAS")
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)

    def log_event(self, event_type: str, details: Dict):
        self.logger.info(f"{event_type}: {json.dumps(details)}")

class GPT4Assistant:
    """GPT-4 integration for adaptive behavior and decision making"""
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.context_window = deque(maxlen=5)

    def get_adaptive_parameters(self, scenario_data: Dict) -> Dict:
        """Get adaptive parameters based on current scenario"""
        scenario_description = self._format_scenario(scenario_data)
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{
                    "role": "system",
                    "content": "You are an ADAS parameter optimization assistant. Provide numerical parameters based on the scenario."
                }, {
                    "role": "user",
                    "content": scenario_description
                }],
                temperature=0.2
            )
            
            # Parse the response and return parameters here
            params = json.loads(response.choices[0].message.content)
            return self._validate_parameters(params)
        except Exception as e:
            logging.error(f"GPT-4 API error: {e}")
            return self._get_default_parameters()

    def _format_scenario(self, data: Dict) -> str:
        """Format scenario data for GPT-4 prompt"""
        return json.dumps({
            "weather_conditions": data.get("weather", "clear"),
            "traffic_density": data.get("traffic_density", "medium"),
            "road_type": data.get("road_type", "highway"),
            "average_speed": data.get("avg_speed", 60),
            "visibility": data.get("visibility", "good")
        })

    def _validate_parameters(self, params: Dict) -> Dict:
        """Validate and constrain parameters from GPT-4"""
        return {
            "fcw_distance": max(10.0, min(30.0, params.get("fcw_distance", FCW_WARNING_DISTANCE))),
            "confidence_threshold": max(0.5, min(0.9, params.get("confidence_threshold", FCW_CONFIDENCE_THRESHOLD)))
        }

    def _get_default_parameters(self) -> Dict:
        """Return default parameters if GPT-4 fails"""
        return {
            "fcw_distance": FCW_WARNING_DISTANCE,
            "confidence_threshold": FCW_CONFIDENCE_THRESHOLD
        }

class YOLODetector:
    """YOLO object detection wrapper"""
    def __init__(self, conf_threshold: float = 0.5):
        self.model = YOLO_MODEL
        self.conf_threshold = conf_threshold
        self.classes = self.model.names
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Perform object detection on frame"""
        results = self.model(frame)
        detections = []
        
        for *box, conf, cls in results.xyxy[0].cpu().numpy():
            if conf > self.conf_threshold:
                detections.append({
                    "bbox": box,
                    "confidence": float(conf),
                    "class": self.classes[int(cls)]
                })
        
        return detections

class CollisionPredictor:
    """Improved collision prediction with better trajectory modeling"""
    def __init__(self, horizon: float = 3.0):
        self.horizon = horizon
        self.history: Dict[int, List[VehicleState]] = {}
        self.min_history_points = 3
        
    def update(self, vehicle_id: int, state: VehicleState) -> None:
        if vehicle_id not in self.history:
            self.history[vehicle_id] = []
        self.history[vehicle_id].append(state)
        
        # Keep only recent history
        if len(self.history[vehicle_id]) > 30:
            self.history[vehicle_id] = self.history[vehicle_id][-30:]
    
    def predict_collision(self, ego_state: VehicleState, 
                         other_state: VehicleState) -> Tuple[bool, float]:
        if len(self.history.get(id(other_state), [])) < self.min_history_points:
            return False, float('inf')
        
        # Calculated relative motion in this section
        relative_pos = other_state.position - ego_state.position
        relative_vel = other_state.velocity - ego_state.velocity
        relative_acc = other_state.acceleration - ego_state.acceleration
        
        # Used quadratic prediction for better accuracy
        distances = []
        for t in np.linspace(0, self.horizon, 20):
            future_pos = (relative_pos + 
                         relative_vel * t + 
                         0.5 * relative_acc * t * t)
            distances.append(np.linalg.norm(future_pos))
        
        min_distance = min(distances)
        min_distance_time = np.linspace(0, self.horizon, 20)[np.argmin(distances)]
        
        collision_prob = self._calculate_collision_probability(
            min_distance, np.linalg.norm(relative_vel), min_distance_time)
        
        return collision_prob > 0.7, min_distance_time

    def _calculate_collision_probability(self, distance: float, 
                                      speed: float, ttc: float) -> float:
        """Enhanced collision probability calculation"""
        if speed < 0.1 or ttc <= 0:
            return 0.0
        
        safety_margin = 2.0  # meters
        effective_distance = max(0.0, distance - safety_margin)
        
        base_prob = 1.0 / (1.0 + np.exp(effective_distance - speed * ttc * 0.5))
        
        uncertainty = 1.0 - np.exp(-ttc / self.horizon)
        
        acc_factor = min(1.0, speed / 20.0)  
        
        return base_prob * (1.0 - uncertainty) * acc_factor

class EnhancedCarlaADAS:
    """Enhanced CARLA ADAS system with advanced features"""
    def __init__(self, gpt4_key: str = ""):
        self.track_history = defaultdict(lambda: deque(maxlen=MAX_TRACKING_HISTORY))
        self.collision_predictor = CollisionPredictor()
        self.yolo_detector = YOLODetector()
        self.gpt4_assistant = GPT4Assistant(gpt4_key)
        self.logger = ADASLogger()
        
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.camera = None
        self.vehicle = None
        self.setup_vehicle()
        self.setup_camera()
        self.setup_sensors()
        
        self.metrics = {
            "false_positives": 0,
            "false_negatives": 0,
            "true_positives": 0,
            "detection_latency": []
        }

class CarlaSensor:
    """Improved sensor wrapper with proper cleanup and synchronization"""
    def __init__(self, world: carla.World, blueprint: carla.ActorBlueprint, 
                 transform: carla.Transform, attached: carla.Actor):
        self.sensor = None
        self.queue: Queue = Queue()
        self.blueprint = blueprint
        self.transform = transform
        self.callbacks = []
        
        try:
            self.sensor = world.spawn_actor(blueprint, transform, attach_to=attached)
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda data: CarlaSensor._on_data(weak_self, data))
        except Exception as e:
            logging.error(f"Failed to initialize sensor: {e}")
            raise
    
    @staticmethod
    def _on_data(weak_self, data):
        self = weak_self()
        if self is not None:
            self.queue.put(data)
            for callback in self.callbacks:
                try:
                    callback(data)
                except Exception as e:
                    logging.error(f"Sensor callback error: {e}")
    
    def get_data(self, timeout: float = 1.0) -> Optional[Any]:
        try:
            return self.queue.get(timeout=timeout)
        except Empty:
            return None
    
    def add_callback(self, callback):
        """Add custom callback for sensor data"""
        self.callbacks.append(callback)
    
    def destroy(self):
        """Enhanced cleanup"""
        try:
            if self.sensor is not None:
                self.sensor.stop()
                self.sensor.destroy()
            self.queue.queue.clear()
            self.callbacks.clear()
        except Exception as e:
            logging.error(f"Error during sensor cleanup: {e}")

PROMPT_TEMPLATES = {
    "weather_adaptation": """
Analyze the following driving scenario and provide optimal ADAS parameters:
Current conditions:
- Weather: {weather}
- Visibility: {visibility}
- Road type: {road_type}
- Traffic density: {traffic_density}
- Average speed: {avg_speed} km/h

Required parameters:
1. FCW warning distance (10-30 meters)
2. FCW confidence threshold (0.5-0.9)
3. Detection sensitivity (0.3-0.8)

Provide parameters as JSON with explanation.
""",

    "traffic_adaptation": """
Optimize ADAS parameters for current traffic conditions:
- Number of nearby vehicles: {vehicle_count}
- Average inter-vehicle distance: {avg_distance}m
- Speed differential: {speed_diff} km/h
- Road type: {road_type}

Required adjustments:
1. Warning timing (earlier/later)
2. False positive tolerance
3. Emergency braking threshold

Respond with JSON containing numerical parameters and brief justification.
""",

    "risk_assessment": """
Evaluate collision risk factors:
- Relative velocity: {rel_velocity} m/s
- Time to collision: {ttc} seconds
- Driver attention status: {attention_status}
- Weather impact: {weather_impact}
- Road condition: {road_condition}

Determine:
1. Risk level (0-1)
2. Warning threshold adjustment
3. Intervention necessity (0-1)

Provide analysis as JSON with confidence scores.
"""
}

def _format_prompt(self, template_name: str, data: Dict) -> str:
    """Format GPT-4 prompt template with current data"""
    if template_name not in PROMPT_TEMPLATES:
        return ""
    
    template = PROMPT_TEMPLATES[template_name]
    try:
        return template.format(**data)
    except KeyError as e:
        logging.error(f"Missing data for prompt template: {e}")
        return template

def predict_trajectories(self, nearby_vehicles: Dict[int, carla.Vehicle]) -> Dict[int, List[np.ndarray]]:
    """
    Predict future trajectories of nearby vehicles with improved accuracy
    
    Args:
        nearby_vehicles: Dictionary of vehicle IDs to vehicle objects
        
    Returns:
        Dictionary of vehicle IDs to predicted positions
    """
    predictions = {}
    
    try:
        for vehicle_id, vehicle in nearby_vehicles.items():
            if vehicle_id not in self.track_history:
                continue
                
            history = self.track_history[vehicle_id]
            if len(history) < 3:  
                continue
            
            positions = np.array([state.position for state in history])
            velocities = np.array([state.velocity for state in history])
            accelerations = np.array([state.acceleration for state in history])
            
            current_pos = positions[-1]
            current_vel = np.mean(velocities[-3:], axis=0)  
            current_acc = np.mean(accelerations[-3:], axis=0) 
            
            max_speed = self._get_speed_limit(vehicle) * 1.1  
            max_acc = 3.0  
            max_decel = -5.0  
            
            future_positions = []
            dt = COLLISION_PREDICTION_HORIZON / 10
            
            for t in np.linspace(0, COLLISION_PREDICTION_HORIZON, 10):
                current_acc = np.clip(current_acc, max_decel, max_acc)
                
                new_vel = current_vel + current_acc * dt
                
                speed = np.linalg.norm(new_vel)
                if speed > max_speed:
                    new_vel = new_vel * max_speed / speed
                
                new_vel = self._adjust_velocity_for_road(vehicle, new_vel)
                
                pred_pos = current_pos + current_vel * dt + 0.5 * current_acc * dt * dt
                
                pred_pos = self._constrain_to_road(vehicle, pred_pos)
                
                future_positions.append(pred_pos)
                
                current_pos = pred_pos
                current_vel = new_vel
            
            predictions[vehicle_id] = future_positions
            
    except Exception as e:
        logging.error(f"Error in trajectory prediction: {e}")
        
    return predictions

def run(self):
    """Main enhanced ADAS loop with improved error handling and synchronization"""
    self.sync_manager = SensorSyncManager()
    self.executor = ThreadPoolExecutor(max_workers=4)
    
    try:
        while True:
            try:
                self.world.tick()
                
                sensor_data = self._get_synchronized_sensor_data()
                if not sensor_data:
                    continue
                
                camera_future = self.executor.submit(self._process_camera_data, 
                                                   sensor_data['camera'])
                lidar_future = self.executor.submit(self._process_lidar_data, 
                                                  sensor_data['lidar'])
                
                try:
                    frame = camera_future.result(timeout=0.1)
                    point_cloud = lidar_future.result(timeout=0.1)
                except TimeoutError:
                    logging.warning("Sensor processing timeout")
                    continue
                
                nearby_vehicles = self._get_nearby_vehicles(max_distance=100.0)
                weather_info = self._get_weather_info()
                
                with self.tracking_lock:
                    detections = self.yolo_detector.detect(frame)
                    self._update_tracking(detections, nearby_vehicles)
                
                trajectories = self.predict_trajectories(nearby_vehicles)
                collision_risks = self._analyze_collision_risks(
                    trajectories, 
                    self._get_adaptive_parameters(weather_info)
                )
                
                self._update_visualization(frame, detections, trajectories, collision_risks)
                self._update_warnings(collision_risks)
                
                cv2.imshow('Enhanced CARLA ADAS', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
            except Empty:
                logging.warning("Sensor data timeout")
                continue
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                continue
                
    finally:
        self._cleanup()

def _process_camera_data(self, camera_data: carla.Image) -> np.ndarray:
    """Process camera data with error handling"""
    try:
        array = np.frombuffer(camera_data.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (camera_data.height, camera_data.width, 4))
        return array[:, :, :3].copy()
    except Exception as e:
        logging.error(f"Error processing camera data: {e}")
        raise

def _get_weather_info(self) -> WeatherInfo:
    """Get current weather information with validation"""
    try:
        weather = self.world.get_weather()
        
        fog_density = max(0.0, min(100.0, weather.fog_density))
        precipitation = max(0.0, min(100.0, weather.precipitation))
        
        if fog_density > 50:
            visibility = "poor"
        elif precipitation > 50:
            visibility = "moderate"
        else:
            visibility = "good"
            
        return WeatherInfo(
            fog_density=fog_density,
            precipitation=precipitation,
            visibility=visibility
        )
        
    except Exception as e:
        logging.error(f"Error getting weather info: {e}")
        return WeatherInfo(0.0, 0.0, "good")  

def _get_road_type(self, location: Optional[carla.Location] = None) -> str:
    """Determine road type with error handling"""
    try:
        if location is None:
            location = self.vehicle.get_location()
            
        waypoint = self.world.get_map().get_waypoint(location)
        
        if waypoint.is_junction:
            return "intersection"
        elif len(waypoint.next(10.0)) > 1:
            return "highway"
        elif waypoint.lane_type == carla.LaneType.Shoulder:
            return "shoulder"
        return "urban"
        
    except Exception as e:
        logging.error(f"Error determining road type: {e}")
        return "urban"  

def _get_average_speed(self, nearby_vehicles: Dict[int, carla.Vehicle]) -> float:
    """Calculate average speed with validation"""
    if not nearby_vehicles:
        return 0.0
        
    try:
        speeds = []
        for vehicle in nearby_vehicles.values():
            if vehicle.is_alive:
                velocity = vehicle.get_velocity()
                speed = 3.6 * np.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
                if 0 <= speed <= 200:  
                    speeds.append(speed)
        
        return np.mean(speeds) if speeds else 0.0
        
    except Exception as e:
        logging.error(f"Error calculating average speed: {e}")
        return 0.0

def _analyze_collision_risks(self, trajectories: Dict[int, List[np.ndarray]], 
                           adaptive_params: Dict[str, float]) -> None:
    """Improved collision risk analysis with uncertainty handling"""
    ego_state = self._get_vehicle_state(self.vehicle)
    high_risk_vehicles = []
    
    for vehicle_id, future_positions in trajectories.items():
        vehicle = self.world.get_actor(vehicle_id)
        if vehicle is None:
            continue
            
        vehicle_state = self._get_vehicle_state(vehicle)
        collision_prob, ttc = self.collision_predictor.predict_collision(
            ego_state, vehicle_state)
        
        warning_distance = adaptive_params.get('fcw_distance', FCW_WARNING_DISTANCE)
        confidence_threshold = adaptive_params.get('confidence_threshold', 
                                                FCW_CONFIDENCE_THRESHOLD)
        
        distance = np.linalg.norm(vehicle_state.position - ego_state.position)
        relative_speed = np.linalg.norm(vehicle_state.velocity - ego_state.velocity)
        
        risk_score = self._calculate_risk_score(
            distance, relative_speed, ttc, collision_prob)
        
        if risk_score > confidence_threshold:
            high_risk_vehicles.append({
                'vehicle_id': vehicle_id,
                'ttc': ttc,
                'risk_score': risk_score,
                'distance': distance
            })
            
            self.logger.log_event("COLLISION_RISK", {
                'vehicle_id': vehicle_id,
                'ttc': ttc,
                'risk_score': risk_score,
                'distance': distance,
                'relative_speed': relative_speed,
                'collision_probability': collision_prob
            })
    
    self._update_warning_system(high_risk_vehicles)

    def _get_vehicle_state(self, vehicle) -> VehicleState:
        """Get current state of a vehicle"""
        location = vehicle.get_location()
        velocity = vehicle.get_velocity()
        acceleration = vehicle.get_acceleration()
        transform = vehicle.get_transform()
        
        return VehicleState(
            position=np.array([location.x, location.y, location.z]),
            velocity=np.array([velocity.x, velocity.y, velocity.z]),
            acceleration=np.array([acceleration.x, acceleration.y, acceleration.z]),
            heading=np.radians(transform.rotation.yaw),
            timestamp=time.time()
        )

    def _update_visualization(self, frame: np.ndarray, detections: List[Dict], trajectories: Dict):
        """Update visualization with detection and prediction overlays"""
        for detection in detections:
            bbox = detection["bbox"]
            conf = detection["confidence"]
            cls = detection["class"]
            
            x1, y1, x2, y2 = map(int, bbox)
            color = (0, 255, 0) if cls == "car" else (255, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls}: {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        for vehicle_id, future_positions in trajectories.items():
            points = [self.world_to_screen(pos) for pos in future_positions]
            points = [p for p in points if p is not None]  # Filter out invalid projections
            
            if len(points) > 1:
                points = np.array(points, dtype=np.int32)
                cv2.polylines(frame, [points], False, (255, 0, 0), 2)

        if self.active_warnings:
            self._draw_warning_overlay(frame)

        self._draw_metrics(frame)

    def _draw_warning_overlay(self, frame: np.ndarray):
        """Draw warning overlay for active warnings"""
        for warning in self.active_warnings:
            if warning.type == "FCW":
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 10)
                cv2.putText(frame, "COLLISION WARNING!", (frame.shape[1]//2 - 200, frame.shape[0]//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
                
                cv2.putText(frame, f"Distance: {warning.distance:.1f}m", (50, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                if warning.ttc < float('inf'):
                    cv2.putText(frame, f"TTC: {warning.ttc:.1f}s", (50, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def _draw_metrics(self, frame: np.ndarray):
        """Draw performance metrics overlay"""
        metrics_text = [
            f"FPS: {self._calculate_fps():.1f}",
            f"Detection Latency: {np.mean(self.metrics['detection_latency']):.1f}ms",
            f"True Positives: {self.metrics['true_positives']}",
            f"False Positives: {self.metrics['false_positives']}",
            f"False Negatives: {self.metrics['false_negatives']}"
        ]
        
        y_offset = 30
        for text in metrics_text:
            cv2.putText(frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25

    def _calculate_fps(self) -> float:
        """Calculate current FPS"""
        current_time = time.time()
        if not hasattr(self, 'last_frame_time'):
            self.last_frame_time = current_time
            return 0.0
        
        fps = 1.0 / (current_time - self.last_frame_time)
        self.last_frame_time = current_time
        return fps

    def _update_tracking(self, detections: List[Dict], nearby_vehicles: Dict):
        """Update vehicle tracking with new detections"""
        current_time = time.time()
        
        # Update existing tracks
        for vehicle_id, vehicle in nearby_vehicles.items():
            state = self._get_vehicle_state(vehicle)
            self.track_history[vehicle_id].append(state)
            self.collision_predictor.update(vehicle_id, state)
            
        for detection in detections:
            bbox = detection["bbox"]
            detection_center = np.array([(bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2])
            
            best_match = None
            min_distance = float('inf')
            
            for vehicle_id in nearby_vehicles:
                vehicle_pos = self.world_to_screen(self.track_history[vehicle_id][-1].position)
                if vehicle_pos is not None:
                    distance = np.linalg.norm(detection_center - vehicle_pos)
                    if distance < min_distance:
                        min_distance = distance
                        best_match = vehicle_id
            
            if best_match is not None:
                detection_time = time.time() - current_time
                self.metrics['detection_latency'].append(detection_time * 1000)  # Convert to ms

    def _log_metrics(self):
        """Log performance metrics"""
        self.logger.log_event("PERFORMANCE_METRICS", {
            "fps": self._calculate_fps(),
            "detection_latency": np.mean(self.metrics['detection_latency']),
            "true_positives": self.metrics['true_positives'],
            "false_positives": self.metrics['false_positives'],
            "false_negatives": self.metrics['false_negatives']
        })

    def _cleanup(self):
        """Cleanup resources"""
        cv2.destroyAllWindows()
        if self.camera:
            self.camera.destroy()
        if self.lidar:
            self.lidar.destroy()
        if self.imu:
            self.imu.destroy()
        if self.vehicle:
            self.vehicle.destroy()
        
        self.logger.log_event("FINAL_METRICS", {
            "total_frames": len(self.metrics['detection_latency']),
            "average_fps": np.mean([1000/lat for lat in self.metrics['detection_latency']]),
            "average_latency": np.mean(self.metrics['detection_latency']),
            "true_positive_rate": self.metrics['true_positives'] / 
                                (self.metrics['true_positives'] + self.metrics['false_negatives'])
                                if (self.metrics['true_positives'] + self.metrics['false_negatives']) > 0 else 0
        })

def main():
    """Initialize and run enhanced CARLA ADAS system"""
    try:
        adas = EnhancedCarlaADAS(gpt4_key="")  # Add your GPT-4 API key here from open ai
        adas.run()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
    except Exception as e:
        print(f'Error occurred: {e}')
        logging.exception("Fatal error in main loop")
    finally:
        print('Cleaning up...')
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
