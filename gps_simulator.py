"""
Advanced GPS and Location Management System
Realistic GPS simulation with route tracking, speed calculation, and geofencing
"""

import random
import time
import math
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging

from config import config

@dataclass
class GPSCoordinate:
    """GPS coordinate with metadata"""
    latitude: float
    longitude: float
    accuracy: float = 5.0  # meters
    timestamp: float = field(default_factory=time.time)
    altitude: Optional[float] = None
    speed_kmh: Optional[float] = None
    heading: Optional[float] = None  # degrees from north
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'accuracy': self.accuracy,
            'timestamp': self.timestamp,
            'altitude': self.altitude,
            'speed_kmh': self.speed_kmh,
            'heading': self.heading,
            'formatted_time': datetime.fromtimestamp(self.timestamp).isoformat()
        }

@dataclass
class RoutePoint:
    """Point along a simulated route"""
    coordinates: GPSCoordinate
    distance_from_start: float = 0.0  # kilometers
    estimated_arrival_time: Optional[float] = None
    road_type: str = "urban"  # urban, highway, rural
    speed_limit: int = 50  # km/h

class GPSSimulator:
    """
    Advanced GPS simulator with realistic movement patterns,
    route following, and fleet management features
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.base_lat = config.gps.default_lat
        self.base_lon = config.gps.default_lon
        self.accuracy = config.gps.accuracy_meters
        
        # Movement simulation
        self.current_position = GPSCoordinate(
            latitude=self.base_lat,
            longitude=self.base_lon,
            accuracy=self.accuracy
        )
        
        # Route and movement parameters
        self.is_moving = False
        self.current_speed = 0.0  # km/h
        self.current_heading = 0.0  # degrees
        self.movement_pattern = "random_walk"  # random_walk, route_following, stationary
        
        # Route management
        self.predefined_routes = self._load_predefined_routes()
        self.current_route = None
        self.route_progress = 0.0
        
        # Movement history
        self.position_history = [self.current_position]
        self.max_history_size = 1000
        
        # Simulation state
        self.last_update_time = time.time()
        self.update_interval = config.gps.update_interval
        
        # Traffic and environmental simulation
        self.traffic_conditions = {
            'congestion_level': 0.0,  # 0.0 to 1.0
            'weather_impact': 0.0,    # 0.0 to 1.0 (affects speed)
            'time_of_day_factor': 1.0  # affects traffic
        }
        
        self.logger.info(f"GPS Simulator initialized at {self.base_lat:.6f}, {self.base_lon:.6f}")
    
    def get_current_location(self) -> GPSCoordinate:
        """Get current GPS location with realistic updates"""
        current_time = time.time()
        
        # Update position if enough time has passed
        if current_time - self.last_update_time >= self.update_interval:
            self._update_position()
            self.last_update_time = current_time
        
        # Add GPS accuracy noise
        noisy_position = self._add_gps_noise(self.current_position)
        
        return noisy_position
    
    def _update_position(self):
        """Update current position based on movement pattern"""
        if self.movement_pattern == "stationary":
            return
        elif self.movement_pattern == "random_walk":
            self._update_random_walk()
        elif self.movement_pattern == "route_following":
            self._update_route_following()
        
        # Update environmental factors
        self._update_environmental_factors()
        
        # Add to history
        self.position_history.append(self.current_position)
        
        # Limit history size
        if len(self.position_history) > self.max_history_size:
            self.position_history = self.position_history[-self.max_history_size//2:]
    
    def _update_random_walk(self):
        """Simulate random vehicle movement with realistic constraints"""
        # Speed variation based on road conditions
        base_speed = random.uniform(30, 80)  # km/h
        speed_factor = 1.0 - (self.traffic_conditions['congestion_level'] * 0.5)
        speed_factor *= (1.0 - self.traffic_conditions['weather_impact'] * 0.3)
        
        self.current_speed = base_speed * speed_factor
        
        # Heading changes (vehicles don't turn abruptly)
        max_heading_change = 15.0  # degrees per update
        heading_change = random.uniform(-max_heading_change, max_heading_change)
        self.current_heading = (self.current_heading + heading_change) % 360
        
        # Calculate distance moved
        time_delta_hours = self.update_interval / 3600  # convert seconds to hours
        distance_km = self.current_speed * time_delta_hours
        
        # Convert to lat/lon changes
        lat_change, lon_change = self._distance_to_coordinates(
            distance_km, self.current_heading
        )
        
        # Update position
        new_lat = self.current_position.latitude + lat_change
        new_lon = self.current_position.longitude + lon_change
        
        # Keep within reasonable bounds (simulate staying in city)
        max_distance = 0.1  # ~10km from base
        if self._calculate_distance(new_lat, new_lon, self.base_lat, self.base_lon) > max_distance:
            # Turn back towards base
            self.current_heading = self._calculate_bearing(
                new_lat, new_lon, self.base_lat, self.base_lon
            )
        
        self.current_position = GPSCoordinate(
            latitude=new_lat,
            longitude=new_lon,
            accuracy=self.accuracy,
            speed_kmh=self.current_speed,
            heading=self.current_heading,
            altitude=random.uniform(100, 200)  # meters
        )
    
    def _update_route_following(self):
        """Update position following a predefined route"""
        if not self.current_route:
            self._select_random_route()
            return
        
        # Progress along route
        route_points = self.current_route['points']
        current_index = int(self.route_progress * (len(route_points) - 1))
        
        if current_index >= len(route_points) - 1:
            # Route completed, select new route
            self._select_random_route()
            return
        
        # Get current and next route points
        current_point = route_points[current_index]
        next_point = route_points[current_index + 1]
        
        # Calculate position between points
        progress_between_points = (self.route_progress * (len(route_points) - 1)) % 1.0
        
        lat = current_point['lat'] + (next_point['lat'] - current_point['lat']) * progress_between_points
        lon = current_point['lon'] + (next_point['lon'] - current_point['lon']) * progress_between_points
        
        # Calculate speed and heading
        heading = self._calculate_bearing(
            current_point['lat'], current_point['lon'],
            next_point['lat'], next_point['lon']
        )
        
        # Speed based on road type and conditions
        speed_limit = current_point.get('speed_limit', 50)
        traffic_factor = 1.0 - self.traffic_conditions['congestion_level'] * 0.6
        speed = speed_limit * traffic_factor * random.uniform(0.8, 1.1)
        
        self.current_position = GPSCoordinate(
            latitude=lat,
            longitude=lon,
            accuracy=self.accuracy,
            speed_kmh=speed,
            heading=heading
        )
        
        # Update route progress
        distance_per_update = (speed * self.update_interval / 3600) / self.current_route['total_distance_km']
        self.route_progress += distance_per_update
    
    def _load_predefined_routes(self) -> Dict:
        """Load predefined routes for realistic movement simulation"""
        # Delhi area routes (can be loaded from JSON file in production)
        routes = {
            'connaught_place_to_airport': {
                'name': 'Connaught Place to IGI Airport',
                'total_distance_km': 18.5,
                'estimated_time_minutes': 45,
                'points': [
                    {'lat': 28.6304, 'lon': 77.2177, 'speed_limit': 40, 'road_type': 'urban'},
                    {'lat': 28.6200, 'lon': 77.2100, 'speed_limit': 60, 'road_type': 'arterial'},
                    {'lat': 28.6000, 'lon': 77.1800, 'speed_limit': 80, 'road_type': 'highway'},
                    {'lat': 28.5665, 'lon': 77.1030, 'speed_limit': 60, 'road_type': 'airport'}
                ]
            },
            'red_fort_to_lotus_temple': {
                'name': 'Red Fort to Lotus Temple',
                'total_distance_km': 12.3,
                'estimated_time_minutes': 30,
                'points': [
                    {'lat': 28.6562, 'lon': 77.2410, 'speed_limit': 30, 'road_type': 'urban'},
                    {'lat': 28.6400, 'lon': 77.2300, 'speed_limit': 50, 'road_type': 'arterial'},
                    {'lat': 28.6100, 'lon': 77.2200, 'speed_limit': 60, 'road_type': 'arterial'},
                    {'lat': 28.5535, 'lon': 77.2588, 'speed_limit': 40, 'road_type': 'urban'}
                ]
            },
            'gurgaon_circuit': {
                'name': 'Gurgaon Business District Circuit',
                'total_distance_km': 25.0,
                'estimated_time_minutes': 60,
                'points': [
                    {'lat': 28.4595, 'lon': 77.0266, 'speed_limit': 60, 'road_type': 'arterial'},
                    {'lat': 28.4700, 'lon': 77.0400, 'speed_limit': 40, 'road_type': 'urban'},
                    {'lat': 28.4800, 'lon': 77.0600, 'speed_limit': 50, 'road_type': 'arterial'},
                    {'lat': 28.4900, 'lon': 77.0500, 'speed_limit': 60, 'road_type': 'arterial'},
                    {'lat': 28.4595, 'lon': 77.0266, 'speed_limit': 40, 'road_type': 'urban'}
                ]
            }
        }
        
        return routes
    
    def _select_random_route(self):
        """Select a random route for following"""
        route_names = list(self.predefined_routes.keys())
        selected_route = random.choice(route_names)
        self.current_route = self.predefined_routes[selected_route]
        self.route_progress = 0.0
        self.logger.info(f"Selected route: {self.current_route['name']}")
    
    def _update_environmental_factors(self):
        """Update traffic and environmental conditions"""
        current_hour = time.localtime().tm_hour
        
        # Time of day traffic patterns
        if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:  # Rush hours
            self.traffic_conditions['congestion_level'] = random.uniform(0.6, 0.9)
            self.traffic_conditions['time_of_day_factor'] = 0.7
        elif 10 <= current_hour <= 16:  # Daytime
            self.traffic_conditions['congestion_level'] = random.uniform(0.3, 0.6)
            self.traffic_conditions['time_of_day_factor'] = 0.9
        else:  # Night/early morning
            self.traffic_conditions['congestion_level'] = random.uniform(0.1, 0.3)
            self.traffic_conditions['time_of_day_factor'] = 1.1
        
        # Random weather effects
        self.traffic_conditions['weather_impact'] = random.uniform(0.0, 0.3)
    
    def _add_gps_noise(self, position: GPSCoordinate) -> GPSCoordinate:
        """Add realistic GPS accuracy noise"""
        # GPS accuracy affects the noise level
        accuracy_factor = self.accuracy / 5.0  # 5m is baseline
        
        lat_noise = random.gauss(0, accuracy_factor * 0.00001)  # ~1.1m per 0.00001 degree
        lon_noise = random.gauss(0, accuracy_factor * 0.00001)
        
        return GPSCoordinate(
            latitude=position.latitude + lat_noise,
            longitude=position.longitude + lon_noise,
            accuracy=position.accuracy + random.uniform(-1, 1),
            timestamp=time.time(),
            speed_kmh=position.speed_kmh,
            heading=position.heading,
            altitude=position.altitude
        )
    
    def _distance_to_coordinates(self, distance_km: float, heading_degrees: float) -> Tuple[float, float]:
        """Convert distance and heading to lat/lon changes"""
        # Approximate conversion (good enough for simulation)
        # 1 degree latitude ≈ 111 km
        # 1 degree longitude ≈ 111 km * cos(latitude)
        
        heading_rad = math.radians(heading_degrees)
        
        lat_change = (distance_km * math.cos(heading_rad)) / 111.0
        lon_change = (distance_km * math.sin(heading_rad)) / (111.0 * math.cos(math.radians(self.current_position.latitude)))
        
        return lat_change, lon_change
    
    def _calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in kilometers
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2) * math.sin(dlat/2) + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(dlon/2) * math.sin(dlon/2))
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _calculate_bearing(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate bearing from point 1 to point 2"""
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        dlon_rad = math.radians(lon2 - lon1)
        
        y = math.sin(dlon_rad) * math.cos(lat2_rad)
        x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
        
        bearing_rad = math.atan2(y, x)
        bearing_deg = math.degrees(bearing_rad)
        
        return (bearing_deg + 360) % 360
    
    def set_movement_pattern(self, pattern: str):
        """Set movement pattern: 'stationary', 'random_walk', 'route_following'"""
        if pattern in ['stationary', 'random_walk', 'route_following']:
            self.movement_pattern = pattern
            self.logger.info(f"Movement pattern set to: {pattern}")
            
            if pattern == 'route_following' and not self.current_route:
                self._select_random_route()
        else:
            self.logger.error(f"Invalid movement pattern: {pattern}")
    
    def get_location_stats(self) -> Dict:
        """Get comprehensive location statistics"""
        total_distance = 0.0
        if len(self.position_history) > 1:
            for i in range(1, len(self.position_history)):
                prev_pos = self.position_history[i-1]
                curr_pos = self.position_history[i]
                total_distance += self._calculate_distance(
                    prev_pos.latitude, prev_pos.longitude,
                    curr_pos.latitude, curr_pos.longitude
                )
        
        # Calculate average speed from history
        avg_speed = 0.0
        if len(self.position_history) > 0:
            speeds = [pos.speed_kmh for pos in self.position_history if pos.speed_kmh]
            avg_speed = sum(speeds) / len(speeds) if speeds else 0.0
        
        return {
            'current_position': self.current_position.to_dict(),
            'movement_pattern': self.movement_pattern,
            'current_speed_kmh': self.current_speed,
            'current_heading_degrees': self.current_heading,
            'total_distance_km': round(total_distance, 2),
            'average_speed_kmh': round(avg_speed, 1),
            'position_history_size': len(self.position_history),
            'traffic_conditions': self.traffic_conditions,
            'current_route': self.current_route['name'] if self.current_route else None,
            'route_progress_percent': round(self.route_progress * 100, 1) if self.current_route else 0
        }
    
    def export_location_history(self, filename: Optional[str] = None) -> str:
        """Export location history to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"gps_history_{timestamp}.json"
        
        filepath = config.storage.base_dir / filename
        
        history_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_points': len(self.position_history),
            'movement_pattern': self.movement_pattern,
            'positions': [pos.to_dict() for pos in self.position_history]
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
        
        self.logger.info(f"Location history exported to {filepath}")
        return str(filepath)

# Global GPS simulator instance
_gps_simulator = None

def get_gps_simulator() -> GPSSimulator:
    """Get global GPS simulator instance (singleton pattern)"""
    global _gps_simulator
    if _gps_simulator is None:
        _gps_simulator = GPSSimulator()
    return _gps_simulator

def get_gps() -> Dict:
    """Get current GPS coordinates (backward compatibility)"""
    simulator = get_gps_simulator()
    current_location = simulator.get_current_location()
    
    return {
        'latitude': current_location.latitude,
        'longitude': current_location.longitude,
        'accuracy': current_location.accuracy,
        'timestamp': current_location.timestamp,
        'speed_kmh': current_location.speed_kmh,
        'heading': current_location.heading
    }

def get_current_location() -> GPSCoordinate:
    """Get current GPS location as GPSCoordinate object"""
    return get_gps_simulator().get_current_location()

# Export classes and functions
__all__ = [
    'GPSCoordinate', 'RoutePoint', 'GPSSimulator',
    'get_gps_simulator', 'get_gps', 'get_current_location'
]