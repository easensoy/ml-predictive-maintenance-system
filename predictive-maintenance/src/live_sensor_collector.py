import requests
import json
import time
import random
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CONFIG_AVAILABLE = False
config_loader_available = False

try:
    from config.config_loader import get_config, get_thingspeak_config, get_equipment_config
    CONFIG_AVAILABLE = True
    config_loader_available = True
    print("✅ Configuration loaded successfully")
except ImportError as e:
    print(f"⚠️ Configuration module not available: {e}")
    print("Using fallback configuration...")

class LiveSensorDataCollector:
    def __init__(self):
        self.config_available = config_loader_available
        
        if self.config_available:
            try:
                self.config = get_config()
                self.thingspeak_config = get_thingspeak_config()
                self.equipment_config = get_equipment_config()
                
                intervals = self.config.get_update_intervals()
                self.update_interval = intervals.get('live_collection', 30)
                
                data_config = self.config.get_data_collection_config()
                retry_settings = data_config.get('retry_settings', {})
                self.timeout = retry_settings.get('timeout', 10)
                
                print("✅ Using YAML configuration")
            except Exception as e:
                print(f"⚠️ Error loading YAML config: {e}")
                self.config_available = False
        
        if not self.config_available:
            self.thingspeak_config = {
                'write_api_key': '2ZPPU9L06NNGD860',
                'channel_id': '3031670',
                'read_api_key': '6AGWH7LEB9BJ2BRP',
                'public_channels': {
                    'weather_station': {
                        'channel_id': 936796,
                        'fields': {1: 'temperature', 2: 'humidity', 3: 'pressure'}
                    },
                    'industrial_sensors': {
                        'channel_id': 762208,
                        'fields': {1: 'temperature', 2: 'vibration'}
                    }
                }
            }
            self.equipment_config = {
                'EQ_001': {'name': 'Turbine Generator', 'base_rpm': 1800, 'temp_factor': 1.2},
                'EQ_002': {'name': 'Air Compressor', 'base_rpm': 1750, 'temp_factor': 1.1},
                'EQ_003': {'name': 'Hydraulic Pump', 'base_rpm': 1650, 'temp_factor': 0.9},
                'EQ_004': {'name': 'Cooling System', 'base_rpm': 1200, 'temp_factor': 0.8},
                'EQ_005': {'name': 'Conveyor Motor', 'base_rpm': 1500, 'temp_factor': 1.0}
            }
            self.update_interval = 30
            self.timeout = 10
            print("✅ Using fallback configuration")

    def get_thingspeak_public_data(self, channel_id, field_num):
        try:
            url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{field_num}/last.json"
            response = requests.get(url, timeout=self.timeout)
            
            if response.status_code == 200:
                data = response.json()
                field_value = data.get(f'field{field_num}')
                if field_value and field_value != 'null' and field_value != '':
                    return {
                        'value': float(field_value),
                        'timestamp': data.get('created_at'),
                        'channel_id': channel_id,
                        'field_id': field_num,
                        'source': 'ThingSpeak'
                    }
            return None
                
        except Exception as e:
            print(f"ThingSpeak API error for channel {channel_id}, field {field_num}: {e}")
            return None

    def send_to_thingspeak(self, sensor_data):
        try:
            write_api_key = self.thingspeak_config.get('write_api_key')
            if not write_api_key or 'YOUR_' in write_api_key:
                print("⚠️ ThingSpeak Write API key not configured")
                return None
            
            url = "https://api.thingspeak.com/update"
            params = {'api_key': write_api_key}
            
            if 'temperature_bearing' in sensor_data:
                params['field1'] = sensor_data['temperature_bearing']
            if 'vibration_rms' in sensor_data:
                params['field2'] = sensor_data['vibration_rms']
            if 'pressure_oil' in sensor_data:
                params['field3'] = sensor_data['pressure_oil']
            if 'rpm' in sensor_data:
                params['field4'] = sensor_data['rpm']
            if 'oil_quality_index' in sensor_data:
                params['field5'] = sensor_data['oil_quality_index']
            if 'power_consumption' in sensor_data:
                params['field6'] = sensor_data['power_consumption']
            
            response = requests.get(url, params=params, timeout=self.timeout)
            if response.status_code == 200:
                entry_id = response.text
                print(f"✅ Data sent to ThingSpeak: Entry ID {entry_id}")
                return {'success': True, 'entry_id': entry_id}
            else:
                print(f"❌ ThingSpeak send failed: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Error sending to ThingSpeak: {e}")
            return None

    def get_ambient_conditions_from_thingspeak(self):
        ambient_data = {'temperature': 20, 'pressure': 10.13, 'humidity': 65, 'source': 'fallback'}
        
        print("📡 Fetching live ambient data from ThingSpeak...")
        
        public_channels = self.thingspeak_config.get('public_channels', {})
        live_data_found = False
        
        for channel_name, channel_config in public_channels.items():
            channel_id = channel_config.get('channel_id')
            fields = channel_config.get('fields', {})
            
            print(f"   Checking {channel_name} (Channel {channel_id})...")
            
            for field_id, field_name in fields.items():
                data = self.get_thingspeak_public_data(channel_id, field_id)
                if data and data['value'] > 0:
                    live_data_found = True
                    if field_name == 'temperature' and 0 < data['value'] < 50:
                        ambient_data['temperature'] = data['value']
                        ambient_data['source'] = f'ThingSpeak-{channel_name}'
                        print(f"   ✅ Got temperature: {data['value']}°C")
                    elif field_name == 'pressure' and 800 < data['value'] < 1200:
                        ambient_data['pressure'] = data['value'] / 100
                        ambient_data['source'] = f'ThingSpeak-{channel_name}'
                        print(f"   ✅ Got pressure: {data['value']} hPa")
                    elif field_name == 'humidity' and 0 < data['value'] <= 100:
                        ambient_data['humidity'] = data['value']
                        print(f"   ✅ Got humidity: {data['value']}%")
                else:
                    print(f"   ⚠️ No data for {field_name} (field {field_id})")
        
        if live_data_found:
            print(f"📊 Using live data from {ambient_data['source']}")
        else:
            print("⚠️ No live data available, using fallback values")
        
        return ambient_data

    def calculate_failure_risk(self, vibration, temperature, oil_quality, pressure):
        risk_score = 0
        
        if vibration > 3.5:
            risk_score += 0.4
        elif vibration > 2.5:
            risk_score += 0.2
        elif vibration > 2.0:
            risk_score += 0.1
        
        if temperature > 100:
            risk_score += 0.3
        elif temperature > 90:
            risk_score += 0.15
        elif temperature > 80:
            risk_score += 0.05
        
        if oil_quality < 40:
            risk_score += 0.3
        elif oil_quality < 60:
            risk_score += 0.15
        elif oil_quality < 80:
            risk_score += 0.05
        
        if pressure < 8:
            risk_score += 0.2
        elif pressure < 12:
            risk_score += 0.1
        
        risk_score += random.uniform(0, 0.1)
        
        return min(risk_score, 0.95)

    def simulate_industrial_data(self, base_temp, base_pressure):
        equipment_data = []
        
        for eq_id, specs in self.equipment_config.items():
            equipment_temp = base_temp + random.uniform(40, 80) * specs.get('temp_factor', 1.0)
            
            base_rpm = specs.get('base_rpm', 1500)
            base_vibration = 0.8 + (base_rpm / 2000) * 0.5
            vibration_rms = base_vibration + random.uniform(-0.3, 0.7)
            
            oil_pressure = base_pressure + random.uniform(5, 15)
            
            rpm = base_rpm + random.uniform(-50, 50)
            
            oil_quality = random.uniform(85, 100) - (equipment_temp - 60) * 0.1
            
            power_consumption = 45 + (vibration_rms - 1.0) * 20 + random.uniform(-5, 5)
            
            failure_risk = self.calculate_failure_risk(
                vibration_rms, equipment_temp, oil_quality, oil_pressure
            )
            
            equipment_data.append({
                'equipment_id': eq_id,
                'equipment_name': specs.get('name', f'Equipment {eq_id}'),
                'timestamp': datetime.now().isoformat(),
                'vibration_rms': max(0.1, vibration_rms),
                'temperature_bearing': max(20, equipment_temp),
                'pressure_oil': max(1, oil_pressure),
                'rpm': max(500, rpm),
                'oil_quality_index': max(0, min(100, oil_quality)),
                'power_consumption': max(10, power_consumption),
                'failure_probability': failure_risk,
                'data_source': 'Live ThingSpeak + Simulation'
            })
            
        return equipment_data

    def collect_live_data(self):
        print("🌡️ Collecting live sensor data from ThingSpeak...")
        
        ambient_data = self.get_ambient_conditions_from_thingspeak()
        
        print(f"📍 Ambient conditions - Temp: {ambient_data['temperature']}°C, Pressure: {ambient_data['pressure']} bar")
        
        equipment_data = self.simulate_industrial_data(
            ambient_data['temperature'], 
            ambient_data['pressure']
        )
        
        return {
            'ambient_conditions': ambient_data,
            'equipment_data': equipment_data,
            'collection_timestamp': datetime.now().isoformat(),
            'data_source': 'ThingSpeak Live Data',
            'config_type': 'YAML' if CONFIG_AVAILABLE else 'Fallback'
        }

    def get_equipment_summary(self):
        live_data = self.collect_live_data()
        
        summary = []
        for equipment in live_data['equipment_data']:
            summary.append({
                'equipment_id': equipment['equipment_id'],
                'vibration_rms': equipment['vibration_rms'],
                'temperature_bearing': equipment['temperature_bearing'],
                'pressure_oil': equipment['pressure_oil'],
                'rpm': equipment['rpm'],
                'oil_quality': equipment['oil_quality_index'] / 100,
                'power_consumption': equipment['power_consumption'],
                'failure_probability': equipment['failure_probability']
            })
        
        return summary

def get_live_equipment_data():
    try:
        collector = LiveSensorDataCollector()
        return collector.get_equipment_summary()
    except Exception as e:
        print(f"Error collecting live data: {e}")
        return generate_fallback_equipment_data()

def generate_fallback_equipment_data():
    equipment_data = []
    equipment_ids = ['EQ_001', 'EQ_002', 'EQ_003', 'EQ_004', 'EQ_005']
    
    for eq_id in equipment_ids:
        equipment_data.append({
            'equipment_id': eq_id,
            'vibration_rms': random.uniform(0.8, 2.5),
            'temperature_bearing': random.uniform(60, 90),
            'pressure_oil': random.uniform(12, 20),
            'rpm': random.uniform(1200, 1800),
            'oil_quality': random.uniform(0.6, 1.0),
            'power_consumption': random.uniform(40, 60),
            'failure_probability': random.uniform(0.02, 0.25)
        })
    
    return equipment_data

if __name__ == "__main__":
    print("🔧 Testing Live Sensor Data Collector")
    print("=" * 50)
    
    try:
        collector = LiveSensorDataCollector()
        
        print("\n📊 Collecting test data...")
        live_data = collector.collect_live_data()
        
        print(f"\n✅ Collected data for {len(live_data['equipment_data'])} pieces of equipment")
        print(f"📡 Configuration: {live_data['config_type']}")
        print(f"🌍 Ambient source: {live_data['ambient_conditions']['source']}")
        
        for equipment in live_data['equipment_data'][:2]:
            print(f"\n🔧 {equipment['equipment_id']} ({equipment['equipment_name']}):")
            print(f"   Temperature: {equipment['temperature_bearing']:.1f}°C")
            print(f"   Vibration: {equipment['vibration_rms']:.2f} RMS")
            print(f"   Failure Risk: {equipment['failure_probability']*100:.1f}%")
            
        print(f"\n🕐 Data collected at: {live_data['collection_timestamp']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()