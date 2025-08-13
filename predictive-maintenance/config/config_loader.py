import yaml
import os
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_file="config/api_keys.yaml"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_file}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
                return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
    
    def get_thingspeak_config(self):
        return self.config.get('thingspeak', {})
    
    def get_openweather_config(self):
        return self.config.get('openweather', {})
    
    def get_equipment_config(self):
        return self.config.get('equipment', {})
    
    def get_data_collection_config(self):
        return self.config.get('data_collection', {})
    
    def get_thresholds(self):
        return self.config.get('thresholds', {})
    
    def get_sensor_fields(self):
        return self.config.get('sensor_fields', {})
    
    def get_system_config(self):
        return self.config.get('system', {})
    
    def get_update_intervals(self):
        data_config = self.get_data_collection_config()
        return data_config.get('update_intervals', {})
    
    def is_debug_enabled(self):
        system_config = self.get_system_config()
        return system_config.get('debug', False)
    
    def validate_config(self):
        errors = []
        
        thingspeak = self.get_thingspeak_config()
        if not thingspeak.get('write_api_key') or thingspeak.get('write_api_key') == 'YOUR_THINGSPEAK_WRITE_API_KEY':
            errors.append("ThingSpeak Write API Key not configured")
        
        if not thingspeak.get('channel_id'):
            errors.append("ThingSpeak Channel ID not configured")
        
        openweather = self.get_openweather_config()
        if not openweather.get('api_key') or openweather.get('api_key') == 'YOUR_OPENWEATHER_API_KEY':
            errors.append("OpenWeatherMap API Key not configured")
        
        equipment = self.get_equipment_config()
        if not equipment:
            errors.append("No equipment configuration found")
        
        if errors:
            print("⚠️  Configuration Warnings:")
            for error in errors:
                print(f"   - {error}")
            print("\n💡 Update your config/api_keys.yaml file with the correct API keys")
            return False
        
        print("✅ Configuration validation passed!")
        return True
    
    def print_config_summary(self):
        print("📋 Configuration Summary:")
        print("=" * 40)
        
        thingspeak = self.get_thingspeak_config()
        print(f"📡 ThingSpeak Channel: {thingspeak.get('channel_id', 'Not configured')}")
        print(f"   Write API Key: {'✅ Configured' if thingspeak.get('write_api_key') and 'YOUR_' not in thingspeak.get('write_api_key', '') else '❌ Not configured'}")
        
        openweather = self.get_openweather_config()
        print(f"🌤️  OpenWeather API: {'✅ Configured' if openweather.get('api_key') and 'YOUR_' not in openweather.get('api_key', '') else '❌ Not configured'}")
        
        equipment = self.get_equipment_config()
        print(f"🔧 Equipment Count: {len(equipment)}")
        
        intervals = self.get_update_intervals()
        print(f"⏰ Live Data Interval: {intervals.get('live_collection', 30)}s")
        print(f"📊 Dashboard Refresh: {intervals.get('dashboard_refresh', 10)}s")

_config_loader = None

def get_config():
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def reload_config():
    global _config_loader
    _config_loader = ConfigLoader()
    return _config_loader

def get_thingspeak_config():
    return get_config().get_thingspeak_config()

def get_openweather_config():
    return get_config().get_openweather_config()

def get_equipment_config():
    return get_config().get_equipment_config()

def validate_configuration():
    return get_config().validate_config()

