"""
Production-Ready Event-Based Video Recording System
Main application with comprehensive error handling, monitoring, and fleet management features
"""

import cv2
import time
import signal
import sys
import threading
import logging
import traceback
from typing import Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import psutil
import queue

# Import system components
from config import config
from video_buffer import AdaptiveVideoBuffer
from event_detector import EventDetectorManager, DetectionResult
from gps_simulator import get_gps_simulator, GPSSimulator
from metadata_manager import AdvancedMetadataManager

class SystemHealthMonitor:
    """Monitor system health and performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {
            'fps': 0.0,
            'cpu_percent': 0.0,
            'memory_percent': 0.0,
            'disk_usage_percent': 0.0,
            'events_per_minute': 0.0,
            'buffer_health': 'unknown',
            'last_update': time.time()
        }
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start system health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        self.logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        """Stop system health monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        self.logger.info("System health monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                # Update system metrics
                self.metrics.update({
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'memory_percent': psutil.virtual_memory().percent,
                    'disk_usage_percent': psutil.disk_usage(str(config.storage.base_dir)).percent,
                    'last_update': time.time()
                })
                
                # Check for alerts
                self._check_alert_conditions()
                
                # Sleep before next check
                time.sleep(config.monitoring.metrics_interval)
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
                time.sleep(5)
    
    def _check_alert_conditions(self):
        """Check if any metrics exceed alert thresholds"""
        thresholds = config.monitoring.alert_thresholds
        
        # Check CPU usage
        if self.metrics['cpu_percent'] > thresholds.get('cpu_usage', 80):
            self._add_alert('high_cpu', f"CPU usage: {self.metrics['cpu_percent']:.1f}%")
        
        # Check memory usage
        if self.metrics['memory_percent'] > thresholds.get('memory_usage', 85):
            self._add_alert('high_memory', f"Memory usage: {self.metrics['memory_percent']:.1f}%")
        
        # Check disk usage
        if self.metrics['disk_usage_percent'] > thresholds.get('disk_usage', 90):
            self._add_alert('high_disk', f"Disk usage: {self.metrics['disk_usage_percent']:.1f}%")
    
    def _add_alert(self, alert_type: str, message: str):
        """Add new alert with timestamp"""
        alert = {
            'type': alert_type,
            'message': message,
            'timestamp': time.time(),
            'formatted_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.alerts.append(alert)
        
        # Limit alert history
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]  # Keep last 50 alerts
        
        self.logger.warning(f"System alert: {alert_type} - {message}")
    
    def get_health_report(self) -> Dict:
        """Get comprehensive health report"""
        # Calculate system status
        status = "healthy"
        if self.metrics['cpu_percent'] > 80 or self.metrics['memory_percent'] > 85:
            status = "warning"
        if self.metrics['disk_usage_percent'] > 90:
            status = "critical"
        
        recent_alerts = [a for a in self.alerts if time.time() - a['timestamp'] < 3600]  # Last hour
        
        return {
            'status': status,
            'metrics': self.metrics.copy(),
            'recent_alerts': recent_alerts,
            'total_alerts': len(self.alerts),
            'uptime_seconds': time.time() - config.monitoring.start_time if hasattr(config.monitoring, 'start_time') else 0,
            'recommendations': self._get_health_recommendations()
        }
    
    def _get_health_recommendations(self) -> list:
        """Generate health-based recommendations"""
        recommendations = []
        
        if self.metrics['memory_percent'] > 80:
            recommendations.append("High memory usage detected - consider reducing buffer size")
        
        if self.metrics['disk_usage_percent'] > 85:
            recommendations.append("Disk space running low - enable automatic cleanup")
        
        if self.metrics['cpu_percent'] > 85:
            recommendations.append("High CPU usage - consider reducing video resolution or FPS")
        
        return recommendations

class EventBasedVideoRecorder:
    """
    Main event-based video recording system with enterprise features
    """
    
    def __init__(self):
        # Setup logging
        self.logger = config.setup_logging()
        self.logger.info("Initializing Event-Based Video Recording System")
        
        # System components
        self.video_buffer = AdaptiveVideoBuffer()
        self.event_detector = EventDetectorManager()
        self.gps_simulator = get_gps_simulator()
        self.metadata_manager = AdvancedMetadataManager()
        
        # System state
        self.is_running = False
        self.camera = None
        self.recording_thread = None
        self.detection_thread = None
        
        # Performance tracking
        self.stats = {
            'start_time': None,
            'frames_processed': 0,
            'events_detected': 0,
            'clips_saved': 0,
            'current_fps': 0.0,
            'last_fps_update': time.time()
        }
        
        # Health monitoring
        self.health_monitor = SystemHealthMonitor()
        
        # Event queue for processing
        self.event_queue = queue.Queue(maxsize=100)
        
        # Graceful shutdown handling
        self.shutdown_event = threading.Event()
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Set monitoring start time
        config.monitoring.start_time = time.time()
        
        self.logger.info("System initialization complete")
    
    def start_recording(self) -> bool:
        """Start the complete video recording system"""
        if self.is_running:
            self.logger.warning("System is already running")
            return False
        
        self.logger.info("Starting Event-Based Video Recording System")
        
        try:
            # Initialize camera
            if not self._initialize_camera():
                return False
            
            # Start system components
            self.is_running = True
            self.stats['start_time'] = time.time()
            
            # Start health monitoring
            self.health_monitor.start_monitoring()
            
            # Start GPS simulation with movement
            self.gps_simulator.set_movement_pattern('random_walk')
            
            # Start recording thread
            self.recording_thread = threading.Thread(target=self._recording_loop)
            self.recording_thread.daemon = True
            self.recording_thread.start()
            
            # Start event detection thread
            self.detection_thread = threading.Thread(target=self._detection_loop)
            self.detection_thread.daemon = True
            self.detection_thread.start()
            
            # Start event processing thread
            self.event_processing_thread = threading.Thread(target=self._event_processing_loop)
            self.event_processing_thread.daemon = True
            self.event_processing_thread.start()
            
            self.logger.info("All system threads started successfully")
            
            # Show initial status
            self._display_status()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            traceback.print_exc()
            self.stop_recording()
            return False
    
    def _initialize_camera(self) -> bool:
        """Initialize camera with error handling and retries"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Attempting to initialize camera (attempt {attempt + 1}/{max_retries})")
                
                self.camera = cv2.VideoCapture(config.video.source)
                
                if not self.camera.isOpened():
                    raise RuntimeError("Camera failed to open")
                
                # Set camera properties
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.video.width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.video.height)
                self.camera.set(cv2.CAP_PROP_FPS, config.video.fps)
                
                # Verify camera settings
                actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
                
                self.logger.info(f"Camera initialized: {actual_width}x{actual_height} @ {actual_fps}fps")
                
                # Test frame capture
                ret, test_frame = self.camera.read()
                if not ret or test_frame is None:
                    raise RuntimeError("Failed to capture test frame")
                
                return True
                
            except Exception as e:
                self.logger.error(f"Camera initialization attempt {attempt + 1} failed: {e}")
                
                if self.camera:
                    self.camera.release()
                    self.camera = None
                
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    self.logger.error("All camera initialization attempts failed")
                    return False
        
        return False
    
    def _recording_loop(self):
        """Main video recording loop"""
        self.logger.info("Video recording loop started")
        frame_count = 0
        last_fps_time = time.time()
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                ret, frame = self.camera.read()
                
                if not ret or frame is None:
                    self.logger.warning("Failed to capture frame")
                    time.sleep(0.1)
                    continue
                
                # Add frame to buffer
                if self.video_buffer.add_frame(frame):
                    frame_count += 1
                    self.stats['frames_processed'] += 1
                
                # Calculate FPS periodically
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.stats['current_fps'] = frame_count / (current_time - last_fps_time)
                    self.health_monitor.metrics['fps'] = self.stats['current_fps']
                    
                    frame_count = 0
                    last_fps_time = current_time
                
                # Small delay to prevent overwhelming the system
                time.sleep(1.0 / config.video.fps)
                
        except Exception as e:
            self.logger.error(f"Recording loop error: {e}")
            traceback.print_exc()
        
        self.logger.info("Video recording loop ended")
    
    def _detection_loop(self):
        """Event detection loop"""
        self.logger.info("Event detection loop started")
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                # Get current frame from buffer
                current_frame = self.video_buffer.get_current_frame()
                
                if current_frame is not None:
                    # Detect events
                    detected_events = self.event_detector.detect_events(
                        current_frame, 
                        frame_id=self.stats['frames_processed']
                    )
                    
                    # Queue events for processing
                    for event in detected_events:
                        try:
                            self.event_queue.put(event, timeout=1)
                            self.stats['events_detected'] += 1
                        except queue.Full:
                            self.logger.warning("Event queue full, dropping event")
                
                # Sleep between detection cycles
                time.sleep(config.ai.detection_interval)
                
        except Exception as e:
            self.logger.error(f"Detection loop error: {e}")
            traceback.print_exc()
        
        self.logger.info("Event detection loop ended")
    
    def _event_processing_loop(self):
        """Process detected events and save clips"""
        self.logger.info("Event processing loop started")
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                try:
                    # Get event from queue (with timeout)
                    event = self.event_queue.get(timeout=1)
                    
                    # Process the event
                    self._process_event(event)
                    
                    # Mark task as done
                    self.event_queue.task_done()
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Event processing error: {e}")
                    traceback.print_exc()
                    
        except Exception as e:
            self.logger.error(f"Event processing loop error: {e}")
            traceback.print_exc()
        
        self.logger.info("Event processing loop ended")
    
    def _process_event(self, event: DetectionResult):
        """Process a detected event and save video clip"""
        try:
            self.logger.info(f"Processing event: {event.event_type} (confidence: {event.confidence:.3f})")
            
            # Get GPS data
            gps_data = self.gps_simulator.get_current_location().to_dict()
            
            # Extract frames for the event
            event_frames, frame_metadata = self.video_buffer.get_event_clip_data(event.timestamp)
            
            if not event_frames:
                self.logger.warning("No frames available for event clip")
                return
            
            # Save video clip
            filename, clip_metadata = self.video_buffer.save_clip(
                event_frames, 
                event.event_type, 
                gps_data,
                event.to_dict()
            )
            
            # Save event metadata
            event_id = self.metadata_manager.save_event_metadata(
                event_type=event.event_type,
                timestamp=event.timestamp,
                filename=filename,
                gps_data=gps_data,
                detection_metadata=event.to_dict(),
                system_metadata=clip_metadata
            )
            
            self.stats['clips_saved'] += 1
            
            self.logger.info(f"Event processed successfully: {event_id} -> {filename}")
            
        except Exception as e:
            self.logger.error(f"Failed to process event: {e}")
            traceback.print_exc()
    
    def _display_status(self):
        """Display current system status"""
        print("\n" + "="*60)
        print("🎥 EVENT-BASED VIDEO RECORDING SYSTEM")
        print("="*60)
        print(f"📹 Camera: {config.video.width}x{config.video.height} @ {config.video.fps}fps")
        print(f"🧠 AI Model: {self.event_detector.active_detector}")
        print(f"📍 GPS: Simulation {'ON' if config.gps.enable_simulation else 'OFF'}")
        print(f"💾 Storage: {config.storage.clips_dir}")
        print(f"🔧 Buffer: {config.buffer.pre_event_duration}s + {config.buffer.post_event_duration}s")
        print("="*60)
        print("System Status: 🟢 RUNNING")
        print("Press Ctrl+C to stop")
        print("="*60)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
        self.stop_recording()
        sys.exit(0)
    
    def stop_recording(self):
        """Stop the video recording system gracefully"""
        if not self.is_running:
            return
        
        self.logger.info("Stopping Event-Based Video Recording System...")
        
        # Set shutdown flag
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop health monitoring
        self.health_monitor.stop_monitoring()
        
        # Wait for threads to complete
        threads = [self.recording_thread, self.detection_thread, 
                  getattr(self, 'event_processing_thread', None)]
        
        for thread in threads:
            if thread and thread.is_alive():
                thread.join(timeout=5)
                if thread.is_alive():
                    self.logger.warning(f"Thread {thread.name} did not stop gracefully")
        
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
        
        # Clean up OpenCV windows
        cv2.destroyAllWindows()
        
        # Final statistics
        self._display_final_stats()
        
        self.logger.info("System stopped successfully")
    
    def _display_final_stats(self):
        """Display final system statistics"""
        runtime = time.time() - (self.stats['start_time'] or time.time())
        
        print("\n" + "="*60)
        print("📊 FINAL SYSTEM STATISTICS")
        print("="*60)
        print(f"⏱️  Runtime: {runtime/3600:.1f} hours")
        print(f"🎞️  Frames Processed: {self.stats['frames_processed']:,}")
        print(f"🎯 Events Detected: {self.stats['events_detected']}")
        print(f"💾 Clips Saved: {self.stats['clips_saved']}")
        print(f"⚡ Avg FPS: {self.stats['frames_processed']/runtime:.1f}")
        print(f"📈 Events/Hour: {self.stats['events_detected']/(runtime/3600):.1f}")
        
        # Buffer statistics
        buffer_stats = self.video_buffer.get_buffer_stats()
        print(f"🗃️  Buffer Utilization: {buffer_stats['utilization_percent']:.1f}%")
        print(f"💿 Memory Used: {buffer_stats['memory_usage_mb']:.1f}MB")
        
        # Health report
        health = self.health_monitor.get_health_report()
        print(f"🏥 System Health: {health['status'].upper()}")
        print("="*60)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'is_running': self.is_running,
            'runtime_seconds': time.time() - (self.stats['start_time'] or time.time()),
            'performance_stats': self.stats.copy(),
            'buffer_stats': self.video_buffer.get_buffer_stats(),
            'detector_stats': self.event_detector.get_detector_stats(),
            'gps_stats': self.gps_simulator.get_location_stats(),
            'health_report': self.health_monitor.get_health_report(),
            'config_summary': {
                'video_resolution': f"{config.video.width}x{config.video.height}",
                'fps': config.video.fps,
                'buffer_duration': config.buffer.pre_event_duration + config.buffer.post_event_duration,
                'ai_model': self.event_detector.active_detector,
                'storage_path': str(config.storage.clips_dir)
            }
        }

def main():
    """Main entry point"""
    print("""
    🎥 Event-Based Video Recording System v2.0
    ==========================================
    Production-ready AI-powered video surveillance
    with automatic event detection and clip saving.
    """)
    
    try:
        # Create and start the system
        recorder = EventBasedVideoRecorder()
        
        if recorder.start_recording():
            # Keep running until interrupted
            try:
                while recorder.is_running:
                    time.sleep(10)
                    
                    # Display periodic status updates
                    if time.time() % 60 < 10:  # Every minute
                        status = recorder.get_system_status()
                        print(f"\n📊 Status: {status['performance_stats']['events_detected']} events, "
                              f"{status['performance_stats']['clips_saved']} clips saved, "
                              f"{status['performance_stats']['current_fps']:.1f} fps")
                        
            except KeyboardInterrupt:
                pass
        else:
            print("❌ Failed to start system")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ System error: {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        print("\n👋 Thank you for using Event-Based Video Recording System!")

if __name__ == "__main__":
    main()