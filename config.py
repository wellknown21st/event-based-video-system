"""
Advanced Configuration Management System
Production-grade settings with environment variable support and validation
"""

import os
import logging
from typing import Dict, List, Union
from dataclasses import dataclass
from pathlib import Path

@dataclass
class VideoConfig:
    """Video recording configuration parameters"""
    source: Union[int, str] = 0
    width: int = 1920
    height: int = 1080
    fps: int = 30
    codec: str = 'mp4v'
    quality: float = 0.8
    
@dataclass 
class BufferConfig:
    """Video buffer configuration parameters"""
    pre_event_duration: int = 15  # seconds
    post_event_duration: int = 15  # seconds 
    max_memory_usage_mb: int = 1024  # MB
    compression_ratio: float = 0.7
    
@dataclass
class AIConfig:
    """AI detection configuration parameters"""
    model_path: str = None
    confidence_threshold: float = 0.75
    detection_interval: float = 0.1  # seconds
    event_types: List[str] = None
    use_gpu: bool = False
    batch_size: int = 1
    
    def __post_init__(self):
        if self.event_types is None:
            self.event_types = [
                'person_detected',
                'vehicle_detected', 
                'overspeed',
                'lane_departure',
                'sudden_brake',
                'tailgating',
                'phone_usage',
                'drowsiness'
            ]

@dataclass
class StorageConfig:
    """Storage and file management configuration"""
    base_dir: Path = Path("./data")
    clips_dir: Path = Path("./data/clips")  
    metadata_file: Path = Path("./data/metadata/events.json")
    logs_dir: Path = Path("./data/logs")
    backup_dir: Path = Path("./data/backups")
    max_storage_gb: float = 10.0
    cleanup_days: int = 30
    
@dataclass
class GPSConfig:
    """GPS and location configuration"""
    enable_simulation: bool = True
    default_lat: float = 28.6139  # Delhi coordinates
    default_lon: float = 77.2090
    accuracy_meters: float = 5.0
    update_interval: float = 1.0
    
@dataclass
class APIConfig:
    """REST API configuration"""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    enable_cors: bool = True
    rate_limit: str = "100 per minute"
    auth_required: bool = False
    jwt_secret: str = None

@dataclass
class MonitoringConfig:
    """System monitoring and alerts configuration"""
    enable_performance_monitoring: bool = True
    metrics_interval: float = 10.0
    alert_thresholds: Dict[str, float] = None
    webhook_url: str = None
    
    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'cpu_usage': 80.0,
                'memory_usage': 85.0, 
                'disk_usage': 90.0,
                'fps_drop': 20.0
            }

class ConfigManager:
    """Central configuration management with environment variable support"""
    
    def __init__(self):
        self.video = VideoConfig()
        self.buffer = BufferConfig()
        self.ai = AIConfig()
        self.storage = StorageConfig()
        self.gps = GPSConfig()
        self.api = APIConfig()
        self.monitoring = MonitoringConfig()
        
        self._load_from_environment()
        self._validate_config()
        self._ensure_directories()
        
    def _load_from_environment(self):
        """Load configuration from environment variables"""
        # Video settings
        self.video.source = os.getenv('VIDEO_SOURCE', self.video.source)
        self.video.width = int(os.getenv('VIDEO_WIDTH', self.video.width))
        self.video.height = int(os.getenv('VIDEO_HEIGHT', self.video.height))
        self.video.fps = int(os.getenv('VIDEO_FPS', self.video.fps))
        
        # AI settings  
        self.ai.model_path = os.getenv('AI_MODEL_PATH', self.ai.model_path)
        self.ai.confidence_threshold = float(os.getenv('AI_CONFIDENCE', self.ai.confidence_threshold))
        self.ai.use_gpu = os.getenv('AI_USE_GPU', 'false').lower() == 'true'
        
        # Storage settings
        base_dir = os.getenv('STORAGE_BASE_DIR')
        if base_dir:
            self.storage.base_dir = Path(base_dir)
            self.storage.clips_dir = Path(base_dir) / "clips"
            self.storage.metadata_file = Path(base_dir) / "metadata" / "events.json"
            self.storage.logs_dir = Path(base_dir) / "logs"
            
        # API settings
        self.api.host = os.getenv('API_HOST', self.api.host)
        self.api.port = int(os.getenv('API_PORT', self.api.port))
        self.api.debug = os.getenv('API_DEBUG', 'false').lower() == 'true'
        
    def _validate_config(self):
        """Validate configuration parameters"""
        errors = []
        
        # Video validation
        if self.video.fps < 1 or self.video.fps > 120:
            errors.append("Video FPS must be between 1 and 120")
            
        if self.video.width < 320 or self.video.height < 240:
            errors.append("Video resolution too low (minimum 320x240)")
            
        # Buffer validation
        total_duration = self.buffer.pre_event_duration + self.buffer.post_event_duration
        if total_duration > 300:  # 5 minutes max
            errors.append("Total buffer duration exceeds maximum (300 seconds)")
            
        # AI validation
        if self.ai.confidence_threshold < 0.1 or self.ai.confidence_threshold > 1.0:
            errors.append("AI confidence threshold must be between 0.1 and 1.0")
            
        # Storage validation
        if self.storage.max_storage_gb < 1.0:
            errors.append("Maximum storage must be at least 1GB")
            
        if errors:
            raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")
            
    def _ensure_directories(self):
        """Create required directories"""
        directories = [
            self.storage.base_dir,
            self.storage.clips_dir, 
            self.storage.metadata_file.parent,
            self.storage.logs_dir,
            self.storage.backup_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    def get_buffer_frames(self) -> int:
        """Calculate required buffer size in frames"""
        return int(self.buffer.pre_event_duration * self.video.fps)
        
    def get_total_clip_frames(self) -> int:
        """Calculate total frames per clip"""
        total_duration = self.buffer.pre_event_duration + self.buffer.post_event_duration
        return int(total_duration * self.video.fps)
        
    def get_estimated_memory_usage(self) -> float:
        """Estimate memory usage in MB"""
        # Rough calculation: width * height * 3 (RGB) * buffer_frames * compression_ratio
        frame_size = self.video.width * self.video.height * 3
        buffer_frames = self.get_buffer_frames()
        total_bytes = frame_size * buffer_frames * self.buffer.compression_ratio
        return total_bytes / (1024 * 1024)  # Convert to MB
        
    def export_config(self) -> Dict:
        """Export current configuration as dictionary"""
        return {
            'video': self.video.__dict__,
            'buffer': self.buffer.__dict__, 
            'ai': self.ai.__dict__,
            'storage': {k: str(v) for k, v in self.storage.__dict__.items()},
            'gps': self.gps.__dict__,
            'api': self.api.__dict__,
            'monitoring': self.monitoring.__dict__
        }
        
    def setup_logging(self):
        """Configure advanced logging"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
        
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.storage.logs_dir / 'system.log'),
                logging.FileHandler(self.storage.logs_dir / 'events.log')
            ]
        )
        
        # Set specific log levels for different components
        logging.getLogger('opencv').setLevel(logging.WARNING)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        
        return logging.getLogger(__name__)

# Global configuration instance
config = ConfigManager()

# Backward compatibility aliases
VIDEO_SOURCE = config.video.source
BUFFER_DURATION = config.buffer.pre_event_duration
POST_EVENT_DURATION = config.buffer.post_event_duration
FRAME_RATE = config.video.fps
SAVE_DIR = str(config.storage.clips_dir) + "/"
METADATA_FILE = str(config.storage.metadata_file)

# Export main configuration
__all__ = [
    'config', 'VideoConfig', 'BufferConfig', 'AIConfig', 
    'StorageConfig', 'GPSConfig', 'APIConfig', 'MonitoringConfig',
    'ConfigManager'
]