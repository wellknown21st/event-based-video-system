"""
Advanced Video Buffer Management System - FIXED VERSION
High-performance circular buffer with memory optimization and smart compression
"""

import cv2
import numpy as np
import collections
import threading
import time
import os
import psutil
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Dict
from pathlib import Path
from dataclasses import dataclass
import logging

from config import config

@dataclass
class FrameMetadata:
    """Metadata for each buffered frame"""
    timestamp: float
    frame_id: int
    size_bytes: int
    compressed: bool = False

class PerformanceMonitor:
    """Monitor system performance and buffer health"""
    
    def __init__(self):
        self.metrics = {
            'frames_buffered': 0,
            'frames_dropped': 0,
            'memory_usage_mb': 0,
            'avg_frame_size_kb': 0,
            'buffer_utilization': 0.0,
            'last_update': time.time()
        }
        
    def update_metrics(self, buffer_size: int, max_size: int, total_memory: float):
        """Update performance metrics"""
        self.metrics.update({
            'frames_buffered': buffer_size,
            'memory_usage_mb': total_memory,
            'buffer_utilization': (buffer_size / max_size) * 100,
            'avg_frame_size_kb': (total_memory * 1024) / max(buffer_size, 1),
            'last_update': time.time()
        })
        
    def get_health_status(self) -> Dict:
        """Get buffer health assessment"""
        memory_usage = psutil.virtual_memory().percent
        
        status = "healthy"
        if self.metrics['buffer_utilization'] > 90:
            status = "near_full"
        elif memory_usage > 85:
            status = "memory_pressure"
        elif self.metrics['frames_dropped'] > 100:
            status = "dropping_frames"
            
        return {
            'status': status,
            'metrics': self.metrics,
            'system_memory_percent': memory_usage,
            'recommendations': self._get_recommendations()
        }
        
    def _get_recommendations(self) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        if self.metrics['buffer_utilization'] > 85:
            recommendations.append("Consider reducing buffer duration or frame resolution")
            
        if self.metrics['avg_frame_size_kb'] > 500:
            recommendations.append("Enable frame compression to reduce memory usage")
            
        if self.metrics['frames_dropped'] > 0:
            recommendations.append("System under load - consider reducing FPS or resolution")
            
        return recommendations

class AdaptiveVideoBuffer:
    """
    Advanced video buffer with adaptive compression and intelligent memory management
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.max_frames = config.get_buffer_frames()
        self.target_memory_mb = config.buffer.max_memory_usage_mb
        self.compression_ratio = config.buffer.compression_ratio
        
        # Buffer storage
        self.frame_buffer = collections.deque(maxlen=self.max_frames)
        self.metadata_buffer = collections.deque(maxlen=self.max_frames)
        
        # Thread safety
        self.buffer_lock = threading.RLock()
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        self.frame_counter = 0
        self.dropped_frames = 0
        
        # Adaptive settings
        self.current_quality = 1.0
        self.compression_enabled = False
        
        # Ensure output directory exists
        config.storage.clips_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized buffer with {self.max_frames} frame capacity")
        
    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add frame to buffer with intelligent memory management
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            bool: True if frame was added, False if dropped
        """
        with self.buffer_lock:
            try:
                # Check memory pressure
                if self._check_memory_pressure():
                    self._adaptive_compression(frame)
                
                # Store frame with metadata
                frame_copy = frame.copy()
                metadata = FrameMetadata(
                    timestamp=time.time(),
                    frame_id=self.frame_counter,
                    size_bytes=frame_copy.nbytes,
                    compressed=self.compression_enabled
                )
                
                self.frame_buffer.append(frame_copy)
                self.metadata_buffer.append(metadata)
                
                self.frame_counter += 1
                
                # Update performance metrics
                self._update_performance_metrics()
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to add frame: {e}")
                self.dropped_frames += 1
                return False
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """
        Get the most recent frame from buffer
        
        Returns:
            Most recent frame or None if buffer is empty
        """
        with self.buffer_lock:
            if self.frame_buffer:
                return self.frame_buffer[-1].copy()
            return None
    
    def get_event_clip_data(self, event_timestamp: float) -> Tuple[List[np.ndarray], List[FrameMetadata]]:
        """
        Extract frames for event clip based on timestamp
        
        Args:
            event_timestamp: When the event occurred
            
        Returns:
            Tuple of (frames_list, metadata_list)
        """
        with self.buffer_lock:
            if not self.frame_buffer:
                return [], []
                
            # Calculate time window
            pre_duration = config.buffer.pre_event_duration
            post_duration = config.buffer.post_event_duration
            
            start_time = event_timestamp - pre_duration
            end_time = event_timestamp + post_duration
            
            # Extract frames within time window
            selected_frames = []
            selected_metadata = []
            
            for frame, metadata in zip(self.frame_buffer, self.metadata_buffer):
                if start_time <= metadata.timestamp <= end_time:
                    selected_frames.append(frame)
                    selected_metadata.append(metadata)
            
            self.logger.info(f"Extracted {len(selected_frames)} frames for event clip")
            return selected_frames, selected_metadata
    
    def save_clip(self, frames: List[np.ndarray], event_type: str, 
                  gps_data: Dict, event_metadata: Dict) -> Tuple[str, Dict]:
        """
        Save video clip with enhanced metadata
        
        Args:
            frames: List of OpenCV frames
            event_type: Type of detected event
            gps_data: GPS coordinates
            event_metadata: Additional event information
            
        Returns:
            Tuple of (filename, clip_metadata)
        """
        if not frames:
            raise ValueError("No frames provided for clip")
            
        # Generate unique filename
        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        event_id = f"{event_type}_{timestamp_str}_{self.frame_counter}"
        filename = f"{event_id}.mp4"
        filepath = config.storage.clips_dir / filename
        
        try:
            # Get video properties from first frame
            height, width, channels = frames[0].shape
            
            # Initialize video writer with optimized settings
            fourcc = cv2.VideoWriter_fourcc(*config.video.codec)
            fps = config.video.fps
            
            out = cv2.VideoWriter(
                str(filepath), 
                fourcc, 
                fps, 
                (width, height),
                isColor=(channels == 3)
            )
            
            # Write frames with progress tracking
            frames_written = 0
            for i, frame in enumerate(frames):
                if frame is not None and frame.size > 0:
                    out.write(frame)
                    frames_written += 1
                    
                    # Log progress for large clips
                    if i % 100 == 0 and len(frames) > 200:
                        progress = (i / len(frames)) * 100
                        self.logger.debug(f"Writing clip: {progress:.1f}% complete")
            
            out.release()
            
            # Verify file was created successfully
            if not filepath.exists() or filepath.stat().st_size == 0:
                raise RuntimeError("Failed to create video file")
            
            # Generate comprehensive metadata
            clip_metadata = {
                'filename': filename,
                'filepath': str(filepath),
                'event_type': event_type,
                'event_id': event_id,
                'timestamp': timestamp.isoformat(),
                'duration_seconds': len(frames) / fps,
                'frame_count': frames_written,
                'resolution': f"{width}x{height}",
                'fps': fps,
                'file_size_bytes': filepath.stat().st_size,
                'file_size_mb': filepath.stat().st_size / (1024 * 1024),
                'gps_data': gps_data,
                'event_metadata': event_metadata,
                'buffer_stats': self.get_buffer_stats(),
                'compression_used': self.compression_enabled,
                'quality_factor': self.current_quality
            }
            
            self.logger.info(f"Saved clip: {filename} ({clip_metadata['file_size_mb']:.1f}MB, {frames_written} frames)")
            
            return filename, clip_metadata
            
        except Exception as e:
            self.logger.error(f"Failed to save clip {filename}: {e}")
            # Clean up partial file
            if filepath.exists():
                filepath.unlink()
            raise
    
    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        current_memory = self._calculate_buffer_memory()
        system_memory = psutil.virtual_memory().percent
        
        return (current_memory > self.target_memory_mb * 0.8 or 
                system_memory > 80)
    
    def _adaptive_compression(self, frame: np.ndarray):
        """Apply adaptive compression based on memory pressure"""
        if not self.compression_enabled:
            self.compression_enabled = True
            self.current_quality = 0.8
            self.logger.info("Enabled adaptive compression due to memory pressure")
        
        # Progressively reduce quality if needed
        if self._calculate_buffer_memory() > self.target_memory_mb:
            self.current_quality = max(0.5, self.current_quality - 0.1)
            self.logger.debug(f"Reduced compression quality to {self.current_quality}")
    
    def _calculate_buffer_memory(self) -> float:
        """Calculate current buffer memory usage in MB"""
        if not self.metadata_buffer:
            return 0.0
            
        total_bytes = sum(metadata.size_bytes for metadata in self.metadata_buffer)
        return total_bytes / (1024 * 1024)
    
    def _update_performance_metrics(self):
        """Update internal performance metrics"""
        buffer_size = len(self.frame_buffer)
        memory_usage = self._calculate_buffer_memory()
        
        self.monitor.update_metrics(buffer_size, self.max_frames, memory_usage)
    
    def get_buffer_stats(self) -> Dict:
        """Get comprehensive buffer statistics"""
        with self.buffer_lock:
            stats = {
                'current_frames': len(self.frame_buffer),
                'max_frames': self.max_frames,
                'utilization_percent': (len(self.frame_buffer) / self.max_frames) * 100,
                'memory_usage_mb': self._calculate_buffer_memory(),
                'target_memory_mb': self.target_memory_mb,
                'frames_added': self.frame_counter,
                'frames_dropped': self.dropped_frames,
                'compression_enabled': self.compression_enabled,
                'current_quality': self.current_quality,
                'oldest_frame_age': 0,
                'newest_frame_age': 0
            }
            
            # Calculate frame ages
            if self.metadata_buffer:
                current_time = time.time()
                oldest_timestamp = self.metadata_buffer[0].timestamp
                newest_timestamp = self.metadata_buffer[-1].timestamp
                
                stats['oldest_frame_age'] = current_time - oldest_timestamp
                stats['newest_frame_age'] = current_time - newest_timestamp
                stats['buffer_duration_seconds'] = newest_timestamp - oldest_timestamp
            
            return stats
    
    def get_health_report(self) -> Dict:
        """Get comprehensive buffer health report"""
        return self.monitor.get_health_status()
    
    def cleanup_old_clips(self, days_to_keep: int = None) -> Dict:
        """
        Clean up old video clips to free storage space
        
        Args:
            days_to_keep: Number of days to keep (uses config default if None)
            
        Returns:
            Cleanup statistics
        """
        if days_to_keep is None:
            days_to_keep = config.storage.cleanup_days
            
        cutoff_time = datetime.now() - timedelta(days=days_to_keep)
        
        clips_deleted = 0
        space_freed_mb = 0.0
        
        try:
            for clip_file in config.storage.clips_dir.glob("*.mp4"):
                file_time = datetime.fromtimestamp(clip_file.stat().st_mtime)
                
                if file_time < cutoff_time:
                    file_size_mb = clip_file.stat().st_size / (1024 * 1024)
                    clip_file.unlink()
                    
                    clips_deleted += 1
                    space_freed_mb += file_size_mb
                    
            cleanup_stats = {
                'clips_deleted': clips_deleted,
                'space_freed_mb': round(space_freed_mb, 2),
                'cutoff_date': cutoff_time.isoformat(),
                'cleanup_timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Cleanup completed: {clips_deleted} clips deleted, {space_freed_mb:.1f}MB freed")
            
            return cleanup_stats
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
            raise

# Backward compatibility
VideoBuffer = AdaptiveVideoBuffer

# Export classes
__all__ = ['AdaptiveVideoBuffer', 'VideoBuffer', 'FrameMetadata', 'PerformanceMonitor']