# config/config_loader.py
"""
YAML Configuration Loader for Predictive Maintenance System
"""

import yaml
import os
from pathlib import Path

class ConfigLoader:
    def __init__(self, config_file="config/api_keys.yaml"):
        self.config_file = config_file
        self.config = self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
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
        """Get ThingSpeak configuration"""
        return self.config.get('thingspeak', {})
    
    def get_openweather_config(self):
        """Get OpenWeatherMap configuration"""
        return self.config.get('openweather', {})
    
    def get_equipment_config(self):
        """Get equipment configuration"""
        return self.config.get('equipment', {})
    
    def get_data_collection_config(self):
        """Get data collection settings"""
        return self.config.get('data_collection', {})
    
    def get_thresholds(self):
        """Get alert thresholds"""
        return self.config.get('thresholds', {})
    
    def get_sensor_fields(self):
        """Get sensor field mappings"""
        return self.config.get('sensor_fields', {})
    
    def get_system_config(self):
        """Get system settings"""
        return self.config.get('system', {})
    
    def get_update_intervals(self):
        """Get update interval settings"""
        data_config = self.get_data_collection_config()
        return data_config.get('update_intervals', {})
    
    def is_debug_enabled(self):
        """Check if debug mode is enabled"""
        system_config = self.get_system_config()
        return system_config.get('debug', False)
    
    def validate_config(self):
        """Validate required configuration values"""
        errors = []
        
        # Check ThingSpeak config
        thingspeak = self.get_thingspeak_config()
        if not thingspeak.get('write_api_key') or thingspeak.get('write_api_key') == 'YOUR_THINGSPEAK_WRITE_API_KEY':
            errors.append("ThingSpeak Write API Key not configured")
        
        if not thingspeak.get('channel_id'):
            errors.append("ThingSpeak Channel ID not configured")
        
        # Check OpenWeather config
        openweather = self.get_openweather_config()
        if not openweather.get('api_key') or openweather.get('api_key') == 'YOUR_OPENWEATHER_API_KEY':
            errors.append("OpenWeatherMap API Key not configured")
        
        # Check equipment config
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
        """Print a summary of the loaded configuration"""
        print("📋 Configuration Summary:")
        print("=" * 40)
        
        # ThingSpeak
        thingspeak = self.get_thingspeak_config()
        print(f"📡 ThingSpeak Channel: {thingspeak.get('channel_id', 'Not configured')}")
        print(f"   Write API Key: {'✅ Configured' if thingspeak.get('write_api_key') and 'YOUR_' not in thingspeak.get('write_api_key', '') else '❌ Not configured'}")
        
        # OpenWeather
        openweather = self.get_openweather_config()
        print(f"🌤️  OpenWeather API: {'✅ Configured' if openweather.get('api_key') and 'YOUR_' not in openweather.get('api_key', '') else '❌ Not configured'}")
        
        # Equipment
        equipment = self.get_equipment_config()
        print(f"🔧 Equipment Count: {len(equipment)}")
        
        # Update intervals
        intervals = self.get_update_intervals()
        print(f"⏰ Live Data Interval: {intervals.get('live_collection', 30)}s")
        print(f"📊 Dashboard Refresh: {intervals.get('dashboard_refresh', 10)}s")

# Global config instance
_config_loader = None

def get_config():
    """Get global configuration instance"""
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader()
    return _config_loader

def reload_config():
    """Reload configuration from file"""
    global _config_loader
    _config_loader = ConfigLoader()
    return _config_loader

# Convenience functions for common config access
def get_thingspeak_config():
    return get_config().get_thingspeak_config()

def get_openweather_config():
    return get_config().get_openweather_config()

def get_equipment_config():
    return get_config().get_equipment_config()

def validate_configuration():
    return get_config().validate_config()

if __name__ == "__main__":
    # Test the configuration loader
    try:
        config = ConfigLoader()
        config.print_config_summary()
        config.validate_config()
    except Exception as e:
        print(f"❌ Configuration Error: {e}")