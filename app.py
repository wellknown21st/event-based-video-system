"""
Advanced Flask API for Event-Based Video Recording System
Production-grade REST API with authentication, rate limiting, and comprehensive endpoints
"""

import os
import sys
import time
import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.exceptions import RequestEntityTooLarge
import jwt

# Import system components
from config import config
from main import EventBasedVideoRecorder
from metadata_manager import AdvancedMetadataManager
from gps_simulator import get_gps_simulator

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Rate limiting
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[config.api.rate_limit]
)
limiter.init_app(app)


# Global instances
recorder_instance: Optional[EventBasedVideoRecorder] = None
metadata_manager = AdvancedMetadataManager()
gps_simulator = get_gps_simulator()

# API versioning
API_VERSION = "v2"
API_PREFIX = f"/api/{API_VERSION}"

class APIAuth:
    """Simple JWT-based authentication"""
    
    @staticmethod
    def generate_token(payload: Dict) -> str:
        """Generate JWT token"""
        if not config.api.jwt_secret:
            return "no-auth-required"
        
        return jwt.encode(
            {**payload, 'exp': time.time() + 86400},  # 24 hour expiry
            config.api.jwt_secret,
            algorithm='HS256'
        )
    
    @staticmethod
    def verify_token(token: str) -> Optional[Dict]:
        """Verify JWT token"""
        if not config.api.auth_required:
            return {'user': 'anonymous'}
        
        try:
            payload = jwt.decode(token, config.api.jwt_secret, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

def require_auth(f):
    """Decorator for endpoints requiring authentication"""
    def decorated_function(*args, **kwargs):
        if not config.api.auth_required:
            return f(*args, **kwargs)
        
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Authentication required'}), 401
        
        token = auth_header.split(' ')[1]
        payload = APIAuth.verify_token(token)
        
        if not payload:
            return jsonify({'error': 'Invalid or expired token'}), 401
        
        request.user = payload
        return f(*args, **kwargs)
    
    decorated_function.__name__ = f.__name__
    return decorated_function

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested API endpoint does not exist',
        'api_version': API_VERSION
    }), 404

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': 'Too many requests. Please try again later.',
        'retry_after': str(error.retry_after)
    }), 429

@app.errorhandler(RequestEntityTooLarge)
def file_too_large(error):
    return jsonify({
        'error': 'File too large',
        'message': 'The uploaded file exceeds the maximum allowed size'
    }), 413

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

# API Root and Info
@app.route('/')
def api_root():
    """API root with comprehensive information"""
    return jsonify({
        'name': 'Event-Based Video Recording System API',
        'version': API_VERSION,
        'description': 'Production-grade API for AI-powered video surveillance',
        'features': [
            'Real-time video recording with event detection',
            'GPS tracking and geolocation services',
            'Comprehensive metadata management',
            'System health monitoring and analytics',
            'RESTful API with authentication support'
        ],
        'endpoints': {
            'system': f'{API_PREFIX}/system/*',
            'events': f'{API_PREFIX}/events/*',
            'analytics': f'{API_PREFIX}/analytics/*',
            'gps': f'{API_PREFIX}/gps/*',
            'health': f'{API_PREFIX}/health/*'
        },
        'documentation': f'{API_PREFIX}/docs',
        'status': 'operational',
        'timestamp': datetime.now().isoformat()
    })

@app.route(f'{API_PREFIX}/docs')
def api_documentation():
    """Comprehensive API documentation"""
    return jsonify({
        'api_version': API_VERSION,
        'base_url': request.base_url.replace('/docs', ''),
        'authentication': {
            'required': config.api.auth_required,
            'type': 'JWT Bearer Token',
            'header': 'Authorization: Bearer <token>'
        },
        'endpoints': {
            'System Control': {
                f'GET {API_PREFIX}/system/status': 'Get comprehensive system status',
                f'POST {API_PREFIX}/system/start': 'Start video recording system',
                f'POST {API_PREFIX}/system/stop': 'Stop video recording system',
                f'POST {API_PREFIX}/system/restart': 'Restart system with new configuration',
                f'GET {API_PREFIX}/system/config': 'Get current system configuration',
                f'PUT {API_PREFIX}/system/config': 'Update system configuration'
            },
            'Event Management': {
                f'GET {API_PREFIX}/events': 'List events with filtering options',
                f'GET {API_PREFIX}/events/<id>': 'Get specific event details',
                f'GET {API_PREFIX}/events/<id>/download': 'Download event video clip',
                f'DELETE {API_PREFIX}/events/<id>': 'Delete event and associated data',
                f'POST {API_PREFIX}/events/<id>/metadata': 'Update event metadata'
            },
            'Analytics': {
                f'GET {API_PREFIX}/analytics/summary': 'Get analytics summary',
                f'GET {API_PREFIX}/analytics/events': 'Get event analytics',
                f'GET {API_PREFIX}/analytics/performance': 'Get system performance metrics',
                f'GET {API_PREFIX}/analytics/export': 'Export analytics data'
            },
            'GPS & Location': {
                f'GET {API_PREFIX}/gps/current': 'Get current GPS location',
                f'GET {API_PREFIX}/gps/history': 'Get location history',
                f'POST {API_PREFIX}/gps/simulate': 'Control GPS simulation',
                f'GET {API_PREFIX}/gps/export': 'Export location data'
            },
            'Health Monitoring': {
                f'GET {API_PREFIX}/health/status': 'Get system health status',
                f'GET {API_PREFIX}/health/metrics': 'Get detailed health metrics',
                f'GET {API_PREFIX}/health/alerts': 'Get system alerts',
                f'POST {API_PREFIX}/health/alerts/clear': 'Clear system alerts'
            }
        },
        'query_parameters': {
            'events': {
                'limit': 'Maximum number of results (default: 50)',
                'offset': 'Number of results to skip (pagination)',
                'type': 'Filter by event type',
                'start_time': 'ISO timestamp or Unix timestamp',
                'end_time': 'ISO timestamp or Unix timestamp',
                'min_confidence': 'Minimum confidence threshold (0.0-1.0)'
            }
        },
        'response_format': {
            'success': {'status': 'success', 'data': '...', 'timestamp': 'ISO string'},
            'error': {'status': 'error', 'error': 'error_code', 'message': 'description'}
        }
    })

# System Control Endpoints
@app.route(f'{API_PREFIX}/system/status')
@require_auth
def get_system_status():
    """Get comprehensive system status"""
    try:
        if recorder_instance and recorder_instance.is_running:
            status_data = recorder_instance.get_system_status()
            status_data['api_status'] = 'operational'
        else:
            status_data = {
                'is_running': False,
                'api_status': 'standby',
                'message': 'Recording system is not active'
            }
        
        return jsonify({
            'status': 'success',
            'data': status_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'status_retrieval_failed',
            'message': str(e)
        }), 500

@app.route(f'{API_PREFIX}/system/start', methods=['POST'])
@require_auth
def start_system():
    """Start the video recording system"""
    global recorder_instance
    
    try:
        if recorder_instance and recorder_instance.is_running:
            return jsonify({
                'status': 'error',
                'error': 'already_running',
                'message': 'System is already running'
            }), 400
        
        # Create new recorder instance
        recorder_instance = EventBasedVideoRecorder()
        
        # Start in separate thread to prevent blocking
        def start_recorder():
            recorder_instance.start_recording()
        
        start_thread = threading.Thread(target=start_recorder)
        start_thread.daemon = True
        start_thread.start()
        
        # Wait a moment for initialization
        time.sleep(2)
        
        if recorder_instance.is_running:
            return jsonify({
                'status': 'success',
                'data': {
                    'message': 'System started successfully',
                    'system_status': recorder_instance.get_system_status()
                },
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'status': 'error',
                'error': 'startup_failed',
                'message': 'System failed to start properly'
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'startup_error',
            'message': str(e)
        }), 500

@app.route(f'{API_PREFIX}/system/stop', methods=['POST'])
@require_auth
def stop_system():
    """Stop the video recording system"""
    global recorder_instance
    
    try:
        if not recorder_instance or not recorder_instance.is_running:
            return jsonify({
                'status': 'error',
                'error': 'not_running',
                'message': 'System is not currently running'
            }), 400
        
        # Get final statistics before stopping
        final_stats = recorder_instance.get_system_status()
        
        # Stop the system
        recorder_instance.stop_recording()
        recorder_instance = None
        
        return jsonify({
            'status': 'success',
            'data': {
                'message': 'System stopped successfully',
                'final_statistics': final_stats
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'shutdown_error',
            'message': str(e)
        }), 500

# Event Management Endpoints
@app.route(f'{API_PREFIX}/events')
@require_auth
def list_events():
    """List events with advanced filtering"""
    try:
        # Parse query parameters
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        event_type = request.args.get('type')
        hours = request.args.get('hours', type=int)
        min_confidence = request.args.get('min_confidence', type=float)
        
        # Build filters
        filters = {'limit': limit + offset}  # Get extra for offset handling
        
        if event_type:
            filters['event_type'] = event_type
        if hours:
            filters['hours'] = hours
        if min_confidence:
            filters['min_confidence'] = min_confidence
        
        # Get events
        events = metadata_manager.get_events(**filters)
        
        # Apply offset
        if offset > 0:
            events = events[offset:]
        
        # Limit results
        if len(events) > limit:
            events = events[:limit]
        
        # Format for API response
        formatted_events = []
        for event in events:
            event_data = event.to_dict()
            event_data['formatted_timestamp'] = datetime.fromtimestamp(event.timestamp).isoformat()
            event_data['file_exists'] = (config.storage.clips_dir / event.filename).exists()
            formatted_events.append(event_data)
        
        return jsonify({
            'status': 'success',
            'data': {
                'events': formatted_events,
                'pagination': {
                    'total': len(formatted_events),
                    'limit': limit,
                    'offset': offset,
                    'has_more': len(events) == limit
                },
                'filters_applied': {
                    'event_type': event_type,
                    'hours': hours,
                    'min_confidence': min_confidence
                }
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'events_retrieval_failed',
            'message': str(e)
        }), 500

@app.route(f'{API_PREFIX}/events/<event_id>')
@require_auth
def get_event_details(event_id: str):
    """Get detailed information about a specific event"""
    try:
        event = metadata_manager.get_event_by_id(event_id)
        
        if not event:
            return jsonify({
                'status': 'error',
                'error': 'event_not_found',
                'message': f'Event {event_id} not found'
            }), 404
        
        event_data = event.to_dict()
        event_data['formatted_timestamp'] = datetime.fromtimestamp(event.timestamp).isoformat()
        
        # Add file information
        clip_path = config.storage.clips_dir / event.filename
        event_data['file_info'] = {
            'exists': clip_path.exists(),
            'size_mb': clip_path.stat().st_size / (1024 * 1024) if clip_path.exists() else 0,
            'download_url': f"{API_PREFIX}/events/{event_id}/download"
        }
        
        return jsonify({
            'status': 'success',
            'data': event_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'event_retrieval_failed',
            'message': str(e)
        }), 500

@app.route(f'{API_PREFIX}/events/<event_id>/download')
@require_auth
def download_event_clip(event_id: str):
    """Download video clip for a specific event"""
    try:
        event = metadata_manager.get_event_by_id(event_id)
        
        if not event:
            return jsonify({
                'status': 'error',
                'error': 'event_not_found',
                'message': f'Event {event_id} not found'
            }), 404
        
        clip_path = config.storage.clips_dir / event.filename
        
        if not clip_path.exists():
            return jsonify({
                'status': 'error',
                'error': 'file_not_found',
                'message': 'Video clip file not found'
            }), 404
        
        return send_file(
            clip_path,
            as_attachment=True,
            download_name=f"{event.event_type}_{event.event_id}.mp4",
            mimetype='video/mp4'
        )
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'download_failed',
            'message': str(e)
        }), 500

# Analytics Endpoints
@app.route(f'{API_PREFIX}/analytics/summary')
@require_auth
def get_analytics_summary():
    """Get comprehensive analytics summary"""
    try:
        days = request.args.get('days', 7, type=int)
        analytics = metadata_manager.get_analytics(days)
        
        return jsonify({
            'status': 'success',
            'data': analytics,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'analytics_failed',
            'message': str(e)
        }), 500

# GPS Endpoints
@app.route(f'{API_PREFIX}/gps/current')
@require_auth
def get_current_gps():
    """Get current GPS location"""
    try:
        location = gps_simulator.get_current_location()
        
        return jsonify({
            'status': 'success',
            'data': location.to_dict(),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'gps_failed',
            'message': str(e)
        }), 500

@app.route(f'{API_PREFIX}/gps/stats')
@require_auth  
def get_gps_stats():
    """Get GPS and location statistics"""
    try:
        stats = gps_simulator.get_location_stats()
        
        return jsonify({
            'status': 'success',
            'data': stats,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'gps_stats_failed',
            'message': str(e)
        }), 500

# Health Monitoring Endpoints
@app.route(f'{API_PREFIX}/health/status')
def get_health_status():
    """Get system health status (public endpoint)"""
    try:
        health_data = {
            'api_status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'version': API_VERSION,
            'uptime_seconds': time.time() - getattr(app, 'start_time', time.time())
        }
        
        if recorder_instance:
            health_data['recorder_status'] = 'running' if recorder_instance.is_running else 'stopped'
            if recorder_instance.is_running:
                system_status = recorder_instance.get_system_status()
                health_data['system_health'] = system_status.get('health_report', {})
        else:
            health_data['recorder_status'] = 'not_initialized'
        
        return jsonify({
            'status': 'success',
            'data': health_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': 'health_check_failed',
            'message': str(e)
        }), 500

def run_api_server():
    """Run the Flask API server with production settings"""
    app.start_time = time.time()
    
    print(f"""
    🚀 Event-Based Video Recording System API v{API_VERSION}
    ======================================================
    
    📡 Server: http://{config.api.host}:{config.api.port}
    📖 Documentation: http://{config.api.host}:{config.api.port}{API_PREFIX}/docs
    🔒 Authentication: {'Required' if config.api.auth_required else 'Disabled'}
    📊 Rate Limiting: {config.api.rate_limit}
    
    Ready to accept connections!
    """)
    
    # Run with production settings
    app.run(
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug,
        threaded=True,
        use_reloader=False  # Disable in production
    )

if __name__ == '__main__':
    run_api_server()