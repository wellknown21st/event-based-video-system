"""
Advanced Metadata Management System
Comprehensive event data storage with analytics, backup, and multi-format support
"""

import json
import sqlite3
import os
import time
import threading
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import hashlib

from config import config
from gps_simulator import GPSCoordinate

@dataclass
class EventMetadata:
    """Comprehensive event metadata structure"""
    event_id: str
    event_type: str
    timestamp: float
    confidence: float
    filename: str
    file_size_bytes: int
    duration_seconds: float
    gps_data: Dict[str, Any]
    detection_metadata: Dict[str, Any]
    system_metadata: Dict[str, Any]
    created_at: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'EventMetadata':
        """Create instance from dictionary"""
        return cls(**data)

class DatabaseManager:
    """Handles database operations with multiple backend support"""
    
    def __init__(self, db_type: str = "json"):
        self.db_type = db_type
        self.logger = logging.getLogger(__name__)
        
        if db_type == "sqlite":
            self._init_sqlite()
        else:
            self._init_json()
    
    def _init_sqlite(self):
        """Initialize SQLite database with comprehensive schema"""
        self.db_path = config.storage.metadata_file.with_suffix('.db')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create main events table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                event_type TEXT NOT NULL,
                timestamp REAL NOT NULL,
                confidence REAL,
                filename TEXT,
                file_size_bytes INTEGER,
                duration_seconds REAL,
                gps_latitude REAL,
                gps_longitude REAL,
                gps_accuracy REAL,
                gps_speed_kmh REAL,
                gps_heading REAL,
                detection_metadata TEXT,
                system_metadata TEXT,
                created_at TEXT,
                INDEX(timestamp),
                INDEX(event_type),
                INDEX(created_at)
            )
        ''')
        
        # Create analytics table for performance metrics
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_events INTEGER,
                event_type_distribution TEXT,
                avg_confidence REAL,
                total_storage_mb REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                INDEX(date)
            )
        ''')
        
        conn.commit()
        conn.close()
        
        self.logger.info(f"SQLite database initialized at {self.db_path}")
    
    def _init_json(self):
        """Initialize JSON file storage"""
        self.json_path = config.storage.metadata_file
        self.json_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.json_path.exists():
            with open(self.json_path, 'w') as f:
                json.dump({
                    'metadata_version': '2.0',
                    'created_at': datetime.now().isoformat(),
                    'events': []
                }, f, indent=2)
        
        self.logger.info(f"JSON storage initialized at {self.json_path}")
    
    def save_event(self, metadata: EventMetadata) -> bool:
        """Save event metadata to database"""
        try:
            if self.db_type == "sqlite":
                return self._save_to_sqlite(metadata)
            else:
                return self._save_to_json(metadata)
        except Exception as e:
            self.logger.error(f"Failed to save event metadata: {e}")
            return False
    
    def _save_to_sqlite(self, metadata: EventMetadata) -> bool:
        """Save to SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT OR REPLACE INTO events (
                    event_id, event_type, timestamp, confidence, filename,
                    file_size_bytes, duration_seconds, gps_latitude, gps_longitude,
                    gps_accuracy, gps_speed_kmh, gps_heading, detection_metadata,
                    system_metadata, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                metadata.event_id,
                metadata.event_type,
                metadata.timestamp,
                metadata.confidence,
                metadata.filename,
                metadata.file_size_bytes,
                metadata.duration_seconds,
                metadata.gps_data.get('latitude'),
                metadata.gps_data.get('longitude'),
                metadata.gps_data.get('accuracy'),
                metadata.gps_data.get('speed_kmh'),
                metadata.gps_data.get('heading'),
                json.dumps(metadata.detection_metadata),
                json.dumps(metadata.system_metadata),
                metadata.created_at
            ))
            
            conn.commit()
            return True
            
        finally:
            conn.close()
    
    def _save_to_json(self, metadata: EventMetadata) -> bool:
        """Save to JSON file with thread safety"""
        with threading.Lock():
            try:
                # Read current data
                with open(self.json_path, 'r') as f:
                    data = json.load(f)
                
                # Add new event
                data['events'].append(metadata.to_dict())
                data['last_updated'] = datetime.now().isoformat()
                
                # Write back
                with open(self.json_path, 'w') as f:
                    json.dump(data, f, indent=2)
                
                return True
                
            except Exception as e:
                self.logger.error(f"JSON save error: {e}")
                return False
    
    def get_events(self, limit: Optional[int] = None, 
                   event_type: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> List[EventMetadata]:
        """Retrieve events with filtering options"""
        try:
            if self.db_type == "sqlite":
                return self._get_from_sqlite(limit, event_type, start_time, end_time)
            else:
                return self._get_from_json(limit, event_type, start_time, end_time)
        except Exception as e:
            self.logger.error(f"Failed to retrieve events: {e}")
            return []
    
    def _get_from_sqlite(self, limit, event_type, start_time, end_time) -> List[EventMetadata]:
        """Retrieve from SQLite with advanced filtering"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            query = "SELECT * FROM events WHERE 1=1"
            params = []
            
            if event_type:
                query += " AND event_type = ?"
                params.append(event_type)
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
            
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            # Convert to EventMetadata objects
            events = []
            for row in rows:
                gps_data = {
                    'latitude': row[7],
                    'longitude': row[8], 
                    'accuracy': row[9],
                    'speed_kmh': row[10],
                    'heading': row[11]
                }
                
                metadata = EventMetadata(
                    event_id=row[0],
                    event_type=row[1],
                    timestamp=row[2],
                    confidence=row[3],
                    filename=row[4],
                    file_size_bytes=row[5],
                    duration_seconds=row[6],
                    gps_data=gps_data,
                    detection_metadata=json.loads(row[12] or '{}'),
                    system_metadata=json.loads(row[13] or '{}'),
                    created_at=row[14]
                )
                events.append(metadata)
            
            return events
            
        finally:
            conn.close()
    
    def _get_from_json(self, limit, event_type, start_time, end_time) -> List[EventMetadata]:
        """Retrieve from JSON with filtering"""
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        events = []
        for event_data in data.get('events', []):
            # Apply filters
            if event_type and event_data.get('event_type') != event_type:
                continue
                
            if start_time and event_data.get('timestamp', 0) < start_time:
                continue
                
            if end_time and event_data.get('timestamp', 0) > end_time:
                continue
            
            events.append(EventMetadata.from_dict(event_data))
        
        # Sort by timestamp descending
        events.sort(key=lambda x: x.timestamp, reverse=True)
        
        # Apply limit
        if limit:
            events = events[:limit]
        
        return events

class AdvancedMetadataManager:
    """
    Advanced metadata management with analytics, backup, and optimization
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.db_manager = DatabaseManager(config.storage.database_type if hasattr(config.storage, 'database_type') else 'json')
        
        # Analytics tracking
        self.analytics_cache = {
            'daily_stats': {},
            'event_type_counters': {},
            'last_analytics_update': 0
        }
        
        # Backup management
        self.backup_manager = BackupManager()
        
        # Performance optimization
        self.cache_ttl = 300  # 5 minutes
        self.cache = {}
        
        self.logger.info("Advanced metadata manager initialized")
    
    def save_event_metadata(self, event_type: str, timestamp: float, filename: str,
                           gps_data: Dict, detection_metadata: Dict = None,
                           system_metadata: Dict = None) -> str:
        """
        Save comprehensive event metadata with auto-generated ID
        
        Returns:
            str: Generated event ID
        """
        # Generate unique event ID
        event_id = self._generate_event_id(event_type, timestamp)
        
        # Get file information
        file_path = config.storage.clips_dir / filename
        file_size = file_path.stat().st_size if file_path.exists() else 0
        
        # Estimate duration (could be extracted from video metadata in production)
        duration = config.buffer.pre_event_duration + config.buffer.post_event_duration
        
        # Prepare metadata
        metadata = EventMetadata(
            event_id=event_id,
            event_type=event_type,
            timestamp=timestamp,
            confidence=detection_metadata.get('confidence', 0.0) if detection_metadata else 0.0,
            filename=filename,
            file_size_bytes=file_size,
            duration_seconds=duration,
            gps_data=gps_data or {},
            detection_metadata=detection_metadata or {},
            system_metadata=system_metadata or {},
            created_at=datetime.now().isoformat()
        )
        
        # Save to database
        success = self.db_manager.save_event(metadata)
        
        if success:
            # Update analytics
            self._update_analytics(metadata)
            
            # Schedule backup if needed
            self._check_backup_schedule()
            
            self.logger.info(f"Saved event metadata: {event_id}")
        else:
            self.logger.error(f"Failed to save event metadata: {event_id}")
        
        return event_id
    
    def _generate_event_id(self, event_type: str, timestamp: float) -> str:
        """Generate unique event ID"""
        # Create hash from event details for uniqueness
        hash_input = f"{event_type}_{timestamp}_{time.time()}"
        hash_value = hashlib.md5(hash_input.encode()).hexdigest()[:8]
        
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d_%H%M%S")
        return f"{event_type}_{timestamp_str}_{hash_value}"
    
    def get_events(self, **filters) -> List[EventMetadata]:
        """
        Get events with advanced filtering options
        
        Supported filters:
        - limit: Maximum number of events
        - event_type: Filter by event type
        - hours: Events from last N hours
        - start_time: Events after this timestamp
        - end_time: Events before this timestamp
        - min_confidence: Minimum confidence threshold
        """
        # Process time-based filters
        if 'hours' in filters:
            filters['start_time'] = time.time() - (filters['hours'] * 3600)
            del filters['hours']
        
        # Get events from database
        events = self.db_manager.get_events(
            limit=filters.get('limit'),
            event_type=filters.get('event_type'),
            start_time=filters.get('start_time'),
            end_time=filters.get('end_time')
        )
        
        # Apply additional filters
        if 'min_confidence' in filters:
            min_conf = filters['min_confidence']
            events = [e for e in events if e.confidence >= min_conf]
        
        return events
    
    def get_event_by_id(self, event_id: str) -> Optional[EventMetadata]:
        """Get specific event by ID"""
        events = self.db_manager.get_events()
        for event in events:
            if event.event_id == event_id:
                return event
        return None
    
    def delete_event(self, event_id: str) -> bool:
        """Delete event and associated video file"""
        event = self.get_event_by_id(event_id)
        if not event:
            return False
        
        try:
            # Delete video file
            file_path = config.storage.clips_dir / event.filename
            if file_path.exists():
                file_path.unlink()
            
            # Delete from database
            if self.db_manager.db_type == "sqlite":
                conn = sqlite3.connect(self.db_manager.db_path)
                cursor = conn.cursor()
                cursor.execute("DELETE FROM events WHERE event_id = ?", (event_id,))
                conn.commit()
                conn.close()
            else:
                # JSON deletion
                with open(self.db_manager.json_path, 'r') as f:
                    data = json.load(f)
                
                data['events'] = [e for e in data['events'] if e.get('event_id') != event_id]
                
                with open(self.db_manager.json_path, 'w') as f:
                    json.dump(data, f, indent=2)
            
            self.logger.info(f"Deleted event: {event_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete event {event_id}: {e}")
            return False
    
    def get_analytics(self, days: int = 7) -> Dict:
        """Get comprehensive analytics for the specified period"""
        start_time = time.time() - (days * 24 * 3600)
        events = self.get_events(start_time=start_time)
        
        if not events:
            return {
                'period_days': days,
                'total_events': 0,
                'message': 'No events found for the specified period'
            }
        
        # Calculate analytics
        total_events = len(events)
        
        # Event type distribution
        event_types = {}
        confidence_scores = []
        file_sizes = []
        daily_counts = {}
        
        for event in events:
            # Event type counting
            event_types[event.event_type] = event_types.get(event.event_type, 0) + 1
            
            # Confidence tracking
            if event.confidence > 0:
                confidence_scores.append(event.confidence)
            
            # File size tracking
            if event.file_size_bytes > 0:
                file_sizes.append(event.file_size_bytes)
            
            # Daily distribution
            event_date = datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d')
            daily_counts[event_date] = daily_counts.get(event_date, 0) + 1
        
        # Calculate statistics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        total_storage_mb = sum(file_sizes) / (1024 * 1024)
        avg_file_size_mb = total_storage_mb / len(file_sizes) if file_sizes else 0
        
        # Find peak activity times
        hourly_distribution = {}
        for event in events:
            hour = datetime.fromtimestamp(event.timestamp).hour
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
        
        peak_hour = max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else 0
        
        return {
            'period_days': days,
            'analysis_timestamp': datetime.now().isoformat(),
            'total_events': total_events,
            'event_type_distribution': event_types,
            'confidence_stats': {
                'average': round(avg_confidence, 3),
                'min': min(confidence_scores) if confidence_scores else 0,
                'max': max(confidence_scores) if confidence_scores else 0,
                'total_scored_events': len(confidence_scores)
            },
            'storage_stats': {
                'total_storage_mb': round(total_storage_mb, 2),
                'average_file_size_mb': round(avg_file_size_mb, 2),
                'largest_file_mb': max(file_sizes) / (1024 * 1024) if file_sizes else 0
            },
            'temporal_patterns': {
                'daily_distribution': daily_counts,
                'hourly_distribution': hourly_distribution,
                'peak_activity_hour': peak_hour,
                'most_active_day': max(daily_counts.items(), key=lambda x: x[1])[0] if daily_counts else None
            },
            'recommendations': self._generate_recommendations(events)
        }
    
    def _generate_recommendations(self, events: List[EventMetadata]) -> List[str]:
        """Generate optimization recommendations based on event data"""
        recommendations = []
        
        if not events:
            return recommendations
        
        # Analyze file sizes
        large_files = [e for e in events if e.file_size_bytes > 100 * 1024 * 1024]  # >100MB
        if len(large_files) > len(events) * 0.3:
            recommendations.append("Consider reducing video quality to save storage space")
        
        # Analyze confidence scores
        low_confidence = [e for e in events if e.confidence < 0.6]
        if len(low_confidence) > len(events) * 0.4:
            recommendations.append("High rate of low-confidence detections - consider adjusting AI model threshold")
        
        # Analyze event frequency
        total_duration_hours = (max(e.timestamp for e in events) - min(e.timestamp for e in events)) / 3600
        events_per_hour = len(events) / max(total_duration_hours, 1)
        
        if events_per_hour > 10:
            recommendations.append("High event detection rate - review sensitivity settings")
        elif events_per_hour < 0.1:
            recommendations.append("Low event detection rate - verify system is functioning correctly")
        
        return recommendations
    
    def _update_analytics(self, metadata: EventMetadata):
        """Update internal analytics cache"""
        event_date = datetime.fromtimestamp(metadata.timestamp).strftime('%Y-%m-%d')
        
        if event_date not in self.analytics_cache['daily_stats']:
            self.analytics_cache['daily_stats'][event_date] = {'count': 0, 'types': {}}
        
        self.analytics_cache['daily_stats'][event_date]['count'] += 1
        
        event_type = metadata.event_type
        if event_type not in self.analytics_cache['daily_stats'][event_date]['types']:
            self.analytics_cache['daily_stats'][event_date]['types'][event_type] = 0
        
        self.analytics_cache['daily_stats'][event_date]['types'][event_type] += 1
        self.analytics_cache['last_analytics_update'] = time.time()
    
    def _check_backup_schedule(self):
        """Check if backup is needed and schedule if necessary"""
        # Simple backup trigger - every 100 events or daily
        events_count = len(self.get_events(limit=1000))  # Quick count
        
        if events_count % 100 == 0:  # Every 100 events
            self.backup_manager.create_backup()

class BackupManager:
    """Manages automated backups of metadata and video clips"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.backup")
        self.backup_dir = config.storage.backup_dir
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def create_backup(self) -> str:
        """Create timestamped backup of metadata and clips"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # Backup metadata
            if config.storage.metadata_file.exists():
                shutil.copy2(config.storage.metadata_file, backup_path / "metadata.json")
            
            # Backup recent clips (last 24 hours)
            recent_time = time.time() - (24 * 3600)
            clips_backed_up = 0
            
            for clip_file in config.storage.clips_dir.glob("*.mp4"):
                if clip_file.stat().st_mtime > recent_time:
                    shutil.copy2(clip_file, backup_path)
                    clips_backed_up += 1
            
            # Create backup manifest
            manifest = {
                'backup_timestamp': datetime.now().isoformat(),
                'clips_backed_up': clips_backed_up,
                'metadata_included': True,
                'backup_size_mb': self._calculate_directory_size(backup_path)
            }
            
            with open(backup_path / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"Backup created: {backup_name} ({clips_backed_up} clips)")
            return str(backup_path)
            
        except Exception as e:
            self.logger.error(f"Backup failed: {e}")
            return ""
    
    def _calculate_directory_size(self, directory: Path) -> float:
        """Calculate total size of directory in MB"""
        total_size = sum(f.stat().st_size for f in directory.rglob('*') if f.is_file())
        return total_size / (1024 * 1024)
    
    def list_backups(self) -> List[Dict]:
        """List available backups with metadata"""
        backups = []
        
        for backup_dir in self.backup_dir.glob("backup_*"):
            if backup_dir.is_dir():
                manifest_file = backup_dir / "manifest.json"
                
                if manifest_file.exists():
                    with open(manifest_file) as f:
                        manifest = json.load(f)
                    
                    manifest['backup_name'] = backup_dir.name
                    manifest['backup_path'] = str(backup_dir)
                    backups.append(manifest)
        
        return sorted(backups, key=lambda x: x['backup_timestamp'], reverse=True)

# Backward compatibility function
def save_metadata(event_type: str, timestamp: float, filename: str, gps_data: Dict):
    """Save metadata (backward compatibility wrapper)"""
    manager = AdvancedMetadataManager()
    return manager.save_event_metadata(event_type, timestamp, filename, gps_data)

# Export classes and functions
__all__ = [
    'EventMetadata', 'DatabaseManager', 'AdvancedMetadataManager', 
    'BackupManager', 'save_metadata'
]