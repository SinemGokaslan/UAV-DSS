"""
UAV Mission Prioritization Decision Support System - Data Collection Module
Real Flight Data from OpenSky Network API
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import time
import os

class OpenSkyDataCollector:
    """Collect flight data from OpenSky Network API"""
    
    def __init__(self):
        self.base_url = "https://opensky-network.org/api"
        
    def get_flights_in_area(self, lat_min=39.0, lat_max=42.0, 
                           lon_min=26.0, lon_max=45.0, max_flights=20):
        """Get active flights in specified area"""
        try:
            url = f"{self.base_url}/states/all"
            params = {
                'lamin': lat_min,
                'lomin': lon_min,
                'lamax': lat_max,
                'lomax': lon_max
            }
            
            print("Connecting to OpenSky Network API...")
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and 'states' in data and data['states']:
                    flights = []
                    for state in data['states'][:max_flights]:
                        flight = {
                            'icao24': state[0],
                            'callsign': state[1].strip() if state[1] else 'UNKNOWN',
                            'origin_country': state[2],
                            'longitude': state[5] if state[5] else 0,
                            'latitude': state[6] if state[6] else 0,
                            'altitude': state[7] if state[7] else 0,
                            'velocity': state[9] if state[9] else 0,
                            'heading': state[10] if state[10] else 0,
                            'vertical_rate': state[11] if state[11] else 0
                        }
                        flights.append(flight)
                    
                    df = pd.DataFrame(flights)
                    print(f"   ✓ {len(flights)} gerçek uçuş verisi toplandı")
                    return df
                else:
                    print(" Flights not found in the specified area. Using sample data...")
                    return self._create_sample_flight_data(max_flights)
            else:
                print(f" API did not reply (Status: {response.status_code})")
                return self._create_sample_flight_data(max_flights)
                
        except Exception as e:
            print(f"OpenSky connection error: {str(e)}")
            print("Creating sample flight data...")
            return self._create_sample_flight_data(max_flights)
    
    def _create_sample_flight_data(self, n=20):
        """Create sample flight data"""
        np.random.seed(42)
        data = {
            'icao24': [f'UAV{i:03d}' for i in range(n)],
            'callsign': [f'FLIGHT{i:03d}' for i in range(n)],
            'origin_country': ['Turkey'] * n,
            'longitude': np.random.uniform(27, 44, n),
            'latitude': np.random.uniform(36, 42, n),
            'altitude': np.random.uniform(500, 5000, n),
            'velocity': np.random.uniform(50, 250, n),
            'heading': np.random.uniform(0, 360, n),
            'vertical_rate': np.random.uniform(-5, 5, n)
        }
        print(f"   ✓ {n} Simulated flight data created")
        return pd.DataFrame(data)


class WeatherDataCollector:
    """Weather data collector using Open-Meteo API"""
    
    def get_weather_for_locations(self, locations):
        """Weather data for given locations"""
        try:
            weather_data = []
            print("Getting weather data from Open-Meteo API...")
            
            for idx, (lat, lon) in enumerate(locations):
                url = "https://api.open-meteo.com/v1/forecast"
                params = {
                    'latitude': lat,
                    'longitude': lon,
                    'current': 'temperature_2m,wind_speed_10m,wind_direction_10m,weather_code',
                    'timezone': 'Europe/Istanbul'
                }
                
                try:
                    response = requests.get(url, params=params, timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        current = data.get('current', {})
                        
                        weather = {
                            'location_id': idx,
                            'latitude': round(lat, 4),
                            'longitude': round(lon, 4),
                            'temperature': round(current.get('temperature_2m', 15), 1),
                            'wind_speed': round(current.get('wind_speed_10m', 10), 1),
                            'wind_direction': round(current.get('wind_direction_10m', 180), 0),
                            'weather_code': current.get('weather_code', 0),
                            'weather_condition': self._get_weather_condition(current.get('weather_code', 0))
                        }
                        weather_data.append(weather)
                    else:
                        weather_data.append(self._default_weather(idx, lat, lon))
                    
                    time.sleep(0.2)  # Rate limiting
                    
                except:
                    weather_data.append(self._default_weather(idx, lat, lon))
            
            df = pd.DataFrame(weather_data)
            print(f"   ✓ {len(weather_data)} Get weather data for locations")
            return df
            
        except Exception as e:
            print(f" Weather Error: {str(e)}")
            return self._create_sample_weather_data(len(locations))
    
    def _get_weather_condition(self, code):
        """WMO weather code explaining"""
        conditions = {
            0: 'Clear', 1: 'Clear', 2: 'Partly Cloudy', 3: 'Cloudy',
            45: 'Foggy', 48: 'Foggy', 51: 'Light Drizzle', 
            61: 'Light Rain', 63: 'Moderate Rain', 65: 'Heavy Rain',
            71: 'Light Snow', 73: 'Moderate Snow', 75: 'Heavy Snow',
            95: 'Stormy'
        }
        return conditions.get(code, 'Clear')
    
    def _default_weather(self, idx, lat, lon):
        """Default weather"""
        return {
            'location_id': idx,
            'latitude': round(lat, 4),
            'longitude': round(lon, 4),
            'temperature': round(15 + np.random.uniform(-5, 10), 1),
            'wind_speed': round(10 + np.random.uniform(-5, 15), 1),
            'wind_direction': round(np.random.uniform(0, 360), 0),
            'weather_code': 0,
            'weather_condition': 'Clear'
        }
    
    def _create_sample_weather_data(self, n):
        """Simulation weather data"""
        np.random.seed(42)
        conditions = ['Clear', 'Partly Cloudy', 'Cloudy', 'Light Rain']
        data = {
            'location_id': range(n),
            'temperature': np.round(np.random.uniform(5, 30, n), 1),
            'wind_speed': np.round(np.random.uniform(0, 40, n), 1),
            'wind_direction': np.round(np.random.uniform(0, 360, n), 0),
            'weather_condition': np.random.choice(conditions, n)
        }
        return pd.DataFrame(data)


class MissionDataGenerator:
    """Mission Data Generator"""
    
    def generate_missions(self, n_missions=15):
        """Generate mission data"""
        np.random.seed(42)
        
        mission_types = ['Border Patrol', 'Search and Rescue', 'Reconnaissance', 
                        'Disaster Assessment', 'Surveillance']
        
        urgency_levels = ['Critical', 'High', 'Medium', 'Low']
        urgency_scores = {'Critical': 10, 'High': 7, 'Medium': 5, 'Low': 3}
        
        missions = []
        for i in range(n_missions):
            mission_type = np.random.choice(mission_types)
            urgency = np.random.choice(urgency_levels, p=[0.2, 0.3, 0.3, 0.2])
            
            mission = {
                'mission_id': f'M{i+1:03d}',
                'mission_type': mission_type,
                'latitude': round(np.random.uniform(36, 42), 4),
                'longitude': round(np.random.uniform(27, 44), 4),
                'urgency_level': urgency,
                'urgency_score': round(urgency_scores[urgency] + np.random.uniform(-0.5, 0.5), 2),
                'threat_level': round(np.random.uniform(1, 10), 2),
                'civilian_density': round(np.random.uniform(0, 10), 2),
                'required_sensor': np.random.choice(['Thermal', 'HD Camera', 'Night Vision', 'Multi-Spectral']),
                'estimated_duration': int(np.random.uniform(30, 180)),
                'created_time': datetime.now() - timedelta(hours=np.random.randint(0, 24))
            }
            missions.append(mission)
        
        df = pd.DataFrame(missions)
        print(f"   ✓ {n_missions} mission created")
        return df
    
    def generate_uav_fleet(self, n_uavs=8):
        """Generate UAV fleet"""
        np.random.seed(42)
        
        uav_models = ['Bayraktar TB2', 'Anka-S', 'Vestel Karayel', 'TAI Aksungur']
        sensor_types = ['Thermal', 'HD Camera', 'Night Vision', 'Multi-Spectral']
        
        specs = {
            'Bayraktar TB2': {'range': 150, 'endurance': 27, 'max_altitude': 8230},
            'Anka-S': {'range': 200, 'endurance': 24, 'max_altitude': 9144},
            'Vestel Karayel': {'range': 150, 'endurance': 20, 'max_altitude': 6706},
            'TAI Aksungur': {'range': 250, 'endurance': 49, 'max_altitude': 12192}
        }
        
        uavs = []
        for i in range(n_uavs):
            model = np.random.choice(uav_models)
            base_specs = specs[model]
            
            uav = {
                'uav_id': f'UAV{i+1:02d}',
                'model': model,
                'current_latitude': round(np.random.uniform(36, 42), 4),
                'current_longitude': round(np.random.uniform(27, 44), 4),
                'fuel_level': round(np.random.uniform(40, 100), 1),
                'max_range': base_specs['range'],
                'endurance': base_specs['endurance'],
                'max_altitude': base_specs['max_altitude'],
                'sensor_type': np.random.choice(sensor_types),
                'status': np.random.choice(['Ready', 'On Mission', 'Under Maintenance'], p=[0.6, 0.3, 0.1]),
                'flight_hours': round(np.random.uniform(0, 1000), 1)
            }
            uavs.append(uav)
        
        df = pd.DataFrame(uavs)
        print(f" {n_uavs} UAVs created")
        return df


def collect_all_data(output_dir='data'):
    """Collect and save all data"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("UAV MISSION PRIORITIZATION - DATA COLLECTION SYSTEM")
    print("="*70 + "\n")
    
    # 1. Flight data (20 → 200)
    print("Flight Data")
    opensky = OpenSkyDataCollector()
    flight_data = opensky.get_flights_in_area(max_flights=200)
    
    # 2. Weather data (8 → 50 locations)
    print("\n Weather Data")
    weather = WeatherDataCollector()
    # 50 random locations within Turkey
    sample_locations = []
    np.random.seed(42)
    for i in range(50):
        lat = np.random.uniform(36, 42)
        lon = np.random.uniform(26, 45)
        sample_locations.append((lat, lon))
    weather_data = weather.get_weather_for_locations(sample_locations)
    
    # 3. Mission data (1000 → 10000)
    print("\n Mission Data")
    mission_gen = MissionDataGenerator()
    print(" Creating 10,000 missions (this may take ~5-10 seconds)...")
    missions = mission_gen.generate_missions(n_missions=10000)
    
    # 4. UAV fleet (8 → 50)
    print("\n  UAV Fleet Data")
    uav_fleet = mission_gen.generate_uav_fleet(n_uavs=50)
    
    # Save
    print("\n" + "="*70)
    print("SAVING DATA...")
    print("="*70 + "\n")
    
    flight_data.to_csv(f'{output_dir}/flight_data.csv', index=False)
    print(f" {output_dir}/flight_data.csv")
    
    weather_data.to_csv(f'{output_dir}/weather.csv', index=False)
    print(f" {output_dir}/weather.csv")
    
    missions.to_csv(f'{output_dir}/missions.csv', index=False)
    print(f" {output_dir}/missions.csv")
    
    uav_fleet.to_csv(f'{output_dir}/uav_fleet.csv', index=False)
    print(f" {output_dir}/uav_fleet.csv")
    
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETED!")
    print("="*70)
    print(f"\n Summary:")
    print(f"   • {len(flight_data)} flight records")
    print(f"   • {len(weather_data)} weather records")
    print(f"   • {len(missions)} missions")
    print(f"   • {len(uav_fleet)} UAVs\n")
    
    return {
        'flights': flight_data,
        'weather': weather_data,
        'missions': missions,
        'uav_fleet': uav_fleet
    }


if __name__ == "__main__":
    # Collect all data
    data = collect_all_data()
    
    # Show summaries
    print("="*70)
    print("DATA SUMMARIES")
    print("="*70)
    
    print("\n MISSIONS (First 5):")
    print(data['missions'][['mission_id', 'mission_type', 'urgency_level', 
                            'threat_level', 'civilian_density']].head())
    
    print("\n UAV FLEET (First 5):")
    print(data['uav_fleet'][['uav_id', 'model', 'fuel_level', 
                             'sensor_type', 'status']].head())
    
    print("\n WEATHER (First 5):")
    print(data['weather'][['temperature', 'wind_speed', 'weather_condition']].head())
    
    print("\n FLIGHT DATA (First 5):")
    print(data['flights'][['callsign', 'altitude', 'velocity']].head())
    
    print("\n" + "="*70)
    print("Data is ready in the 'data/' folder!")
    print("="*70 + "\n")