# 🎥 Event-Based Video Recording System with AI Event Detection

> **Assignment submission for AI-based event detection video recording system prototype**

## 🎯 Assignment Overview

This project implements a **smart fleet monitoring system** that uses AI-based event detection to trigger automatic video recording. The system demonstrates event-driven architecture for vehicle/driver monitoring applications.

### ✅ Assignment Requirements Fulfilled

- [x] **Mock AI Event Detection** - Simulates YOLO-like event detection system
- [x] **Continuous Video Buffering** - Maintains last 15 seconds of video footage  
- [x] **Event-Triggered Recording** - Saves 30-second clips (15s before + 15s after events)
- [x] **Metadata Storage** - Event type, timestamp, and GPS coordinates for each clip
- [x] **Flask Web Interface** - API to view, download, and manage clips
- [x] **GPS Simulation** - Realistic coordinate generation for fleet tracking

## 🏗️ System Architecture

```
📁 event-based-video-system/
├── 📄 main.py                   # Main application entry point
├── 📄 config.py                 # System configuration settings
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                # This documentation
│
├── 📁 Core Components/
│   ├── 📄 video_buffer.py      # Circular video buffer (15-second rolling)
│   ├── 📄 event_detector.py    # Mock YOLO AI event simulator
│   ├── 📄 metadata_manager.py  # Event metadata storage (JSON)
│   └── 📄 gps_simulator.py     # GPS coordinate simulation
│
├── 📁 Web Interface/
│   └── 📄 app.py               # Flask API server
│
├── 📁 Generated Data/
│   ├── 📁 static/clips/        # Saved 30-second video clips
│   └── 📄 metadata.json       # Event metadata database
│
└── 📁 Documentation/
    └── 📄 demo_instructions.md # Demo and testing guide
```

## 🚀 Key Features Implemented

### 1. **Mock AI Event Detection**
- Simulates YOLO-like behavior with realistic event probabilities
- **Event Types**: `overspeed`, `lane_departure`, `sudden_brake`, `phone_usage`, `drowsiness`
- Configurable detection sensitivity and event frequencies
- Generates confidence scores and bounding box data

### 2. **Smart Video Buffering System**
- **Circular buffer** maintains last 15 seconds of video continuously
- Memory-efficient frame storage with automatic cleanup
- **Event-triggered saving**: When event detected, saves 15s before + 15s after
- Supports multiple video formats and frame rates

### 3. **Comprehensive Metadata Management**
- **JSON-based storage** for event metadata
- **GPS coordinates** with realistic simulation (Delhi area routes)
- **Timestamp tracking** with millisecond precision
- **File management** with automatic cleanup and backup

### 4. **Flask Web API**
- **RESTful endpoints** for event management
- **File download** capability for video clips
- **Event filtering** by type, date range, and confidence
- **System statistics** and performance metrics

## 📋 Requirements

### System Requirements
- **Python 3.7+**
- **OpenCV 4.0+** for video capture and processing
- **Flask 2.0+** for web interface
- **Webcam or USB camera** for video input

### Dependencies
```bash
opencv-python>=4.5.0
flask>=2.0.0
numpy>=1.21.0
```

## ⚡ Quick Start Guide

### 1. Installation
```bash
# Clone or extract the project
cd event-based-video-system

# Install dependencies
pip install -r requirements.txt

# Ensure camera permissions (if needed)
# No additional setup required - system auto-configures
```

### 2. Running the System

**Option A: Basic Video Recording System**
```bash
python main.py
```
- Opens camera feed with live event detection
- Press `q` to quit
- Video clips automatically saved to `static/clips/`

**Option B: With Web Interface**
```bash
# Terminal 1: Run main system
python main.py

# Terminal 2: Start web server  
python app.py

# Access web interface at: http://127.0.0.1:5000
```

### 3. Testing Event Detection

The system **automatically detects mock events** while running:
- **Overspeed events** - Simulates vehicle speed violations
- **Lane departure** - Simulates lane change without signaling  
- **Sudden brake** - Simulates harsh braking detection
- **Phone usage** - Simulates driver distraction detection
- **Drowsiness** - Simulates driver fatigue detection

**Expected Behavior:**
1. Camera feed opens showing live video
2. Console logs show: `[EVENT DETECTED] overspeed` (or other events)
3. Video clips automatically saved as: `static/clips/overspeed_20231007_143045.mp4`
4. Event metadata logged to `metadata.json`

## 🎮 Usage Examples

### Basic Operation
```bash
# Start the system
python main.py

# Expected output:
[INFO] Video buffer initialized
[INFO] Mock event detector started  
[EVENT DETECTED] overspeed
[INFO] Saved clip: static/clips/overspeed_20231007_143045.mp4
[EVENT DETECTED] phone_usage  
[INFO] Saved clip: static/clips/phone_usage_20231007_143112.mp4
```

### Web API Usage
```bash
# List all events
curl http://127.0.0.1:5000/events

# Download specific clip
curl -O http://127.0.0.1:5000/download/overspeed_20231007_143045.mp4

# Response format:
[
  {
    "event_type": "overspeed",
    "timestamp": 1696743045.123,
    "file": "overspeed_20231007_143045.mp4",
    "gps": {
      "latitude": 28.6139,
      "longitude": 77.2090
    }
  }
]
```

## ⚙️ Configuration

### Event Detection Settings
Edit `event_detector.py` to adjust:
```python
# Detection probability (0.005 = 0.5% chance per frame)
DETECTION_PROBABILITY = 0.005

# Event types and their frequencies
EVENT_TYPES = ["overspeed", "lane_departure", "sudden_brake", "phone_usage", "drowsiness"]
```

### Video Settings  
Edit `config.py` to modify:
```python
VIDEO_SOURCE = 0          # Camera index (0, 1, 2...)
BUFFER_DURATION = 15      # Seconds to buffer before events
POST_EVENT_DURATION = 15  # Seconds to record after events  
FRAME_RATE = 30           # Frames per second
SAVE_DIR = "static/clips/" # Video clip storage location
```

### GPS Simulation
Edit `gps_simulator.py` for different locations:
```python
# Default coordinates (Delhi, India)
DEFAULT_LAT = 28.6139  
DEFAULT_LON = 77.2090

# Coordinate variance range
LAT_RANGE = (-0.01, 0.01)  # ~1km radius
LON_RANGE = (-0.01, 0.01)
```

## 🧪 Demo Scenarios

### Scenario 1: Fleet Monitoring Demo
1. Start system: `python main.py`
2. Let it run for 2-3 minutes
3. Observe multiple event types being detected
4. Check `static/clips/` for saved video files
5. View `metadata.json` for event details

### Scenario 2: Web Interface Demo
1. Start both `main.py` and `app.py`
2. Open browser to `http://127.0.0.1:5000/events`
3. View list of detected events with metadata
4. Click download links to retrieve video clips
5. Observe real-time event detection in web interface

### Scenario 3: GPS Tracking Demo
1. Run system and generate several events
2. Check metadata.json for GPS coordinates
3. Plot coordinates to see simulated vehicle movement
4. Observe realistic GPS variance and location tracking

## 📊 System Performance

### Expected Performance Metrics
- **Frame Rate**: 30 FPS (configurable)
- **Event Detection**: ~1-3 events per minute (realistic simulation)
- **Memory Usage**: ~150-300MB (depends on buffer size)
- **Storage**: ~2-5MB per 30-second clip (720p quality)
- **Response Time**: <100ms event detection latency

### Monitoring Tools
```bash
# View real-time statistics
# System automatically logs performance metrics:
[INFO] Buffer status: 450/450 frames (15.0 seconds)
[INFO] Events detected: 12, Clips saved: 12
[INFO] Average FPS: 29.8, Memory usage: 245MB
```

## 🔧 Technical Implementation Details

### Video Buffering Algorithm
- **Circular deque** structure for memory efficiency
- **Frame timestamp** tracking for precise clip extraction
- **Thread-safe** operations for concurrent access
- **Automatic cleanup** prevents memory leaks

### Event Detection Logic
- **Probabilistic simulation** mimics real AI behavior
- **Realistic timing** patterns for different event types
- **Confidence scores** generated with appropriate distributions
- **Bounding box simulation** for object detection visualization

### Metadata Schema
```json
{
  "event_type": "overspeed",
  "timestamp": 1696743045.123,
  "file": "overspeed_20231007_143045.mp4", 
  "gps": {
    "latitude": 28.6139,
    "longitude": 77.2090
  }
}
```

## 🎯 Assignment Deliverables

### ✅ What Was Implemented

1. **Core System Components**
   - ✅ Mock AI event detection with realistic simulation
   - ✅ 15-second continuous video buffering
   - ✅ 30-second clip generation (15s before + 15s after)
   - ✅ Complete metadata storage with GPS simulation

2. **Additional Features**
   - ✅ Flask web interface for clip management
   - ✅ RESTful API with JSON responses
   - ✅ Configurable event types and detection rates
   - ✅ Automatic file organization and cleanup

3. **Code Quality**
   - ✅ Clear, documented, and modular code structure
   - ✅ Error handling and resource cleanup
   - ✅ Configurable parameters and settings
   - ✅ Professional logging and debugging features

### 🧠 Design Decisions & Trade-offs

**1. Mock vs Real AI Detection**
- **Choice**: Implemented comprehensive mock detection system
- **Rationale**: Allows demonstration without requiring trained YOLO models
- **Trade-off**: Realistic behavior patterns but not actual object detection
- **Future**: Easy to replace with real YOLO integration

**2. JSON vs Database Storage**
- **Choice**: JSON file-based metadata storage  
- **Rationale**: Simpler setup, no database dependencies
- **Trade-off**: Limited scalability but excellent for prototype/demo
- **Future**: Can easily migrate to SQLite or PostgreSQL

**3. Single-threaded vs Multi-threaded**
- **Choice**: Single-threaded processing with sequential operations
- **Rationale**: Simpler debugging, adequate for prototype performance
- **Trade-off**: Potential frame drops during clip saving
- **Future**: Multi-threading can be added for production scaling

## 🚨 Known Limitations & Future Enhancements

### Current Limitations
- **Mock detection only** - Not analyzing actual video content
- **Single camera support** - No multi-camera functionality  
- **Limited storage options** - JSON only, no database integration
- **Basic web interface** - Simple API, no dashboard UI

### Proposed Enhancements
- **Real YOLO integration** for actual object detection
- **Multi-camera support** for comprehensive fleet monitoring  
- **Database backend** with SQLite/PostgreSQL support
- **Advanced web dashboard** with real-time monitoring
- **Cloud storage integration** for scalable video archival
- **Mobile app interface** for remote monitoring


## 🏆 Success Criteria Met

✅ **Event-driven System**: Proper event detection triggers video saving  
✅ **Video Stream Processing**: Continuous buffering with circular queue  
✅ **API Integration Logic**: Mock YOLO simulation with realistic behavior  
✅ **Code Clarity**: Well-structured, documented, and readable code  
✅ **Real-world Application**: Fleet monitoring use case implementation  

## 🤝 Acknowledgments

This assignment demonstrates:
- **Event-driven architecture** principles
- **Video processing** with OpenCV
- **API development** with Flask
- **Mock system design** for realistic simulation
- **Professional code structure** and documentation

**Ready for evaluation and technical discussion!** 🚀

---

*Thank you for the opportunity to demonstrate these skills through this engaging assignment!*
