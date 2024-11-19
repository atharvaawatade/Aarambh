# DriveWise Testing Implementation

## Overview
This repository contains a testing implementation of an Advanced Driver Assistance System (ADAS) using computer vision and AI. While this version runs independently of Carla simulator and may experience performance limitations, the main implementation is integrated with Carla for efficient real-time processing.

## Features
- Forward Collision Warning (FCW)
- Lane Departure Warning (LDW)
- Real-time vehicle tracking
- Distance estimation
- Speed calculation
- GPT-4 integration for intelligent decision making

## Performance Note
⚠️ **Important**: This is a testing version that runs without Carla simulator integration. As a result:
- Frame processing may be slower than the production version
- Resource usage might be higher
- Some features might experience latency

For optimal performance, please refer to the main Carla-integrated version.

## Requirements
- Python 3.8+
- OpenCV
- NumPy
- Ultralytics YOLO
- OpenAI GPT-4 API key
- YOLO11n.pt model file

## Setup
1. Install required packages:
```bash
pip install opencv-python numpy ultralytics openai pytz
```

2. Set up your OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

3. Place the YOLO11n.pt model file in your project directory

## Usage
Run the script with:
```bash
python adas_system.py
```

## Main Features Explanation
- **FCW (Forward Collision Warning)**: Alerts when distance to forward vehicle is critically low
- **LDW (Lane Departure Warning)**: Warns when vehicle is drifting from lane
- **Speed Tracking**: Calculates relative speeds between vehicles
- **AI Integration**: Uses GPT-4 for enhanced decision making and risk assessment



## Note
This is a testing implementation. For production use, please refer to the Carla-integrated version which offers:
- Real-time performance
- Lower latency
- Better resource optimization
- Full integration with Carla simulator
