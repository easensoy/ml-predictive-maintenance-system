import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import random

def generate_sensor_data(num_samples=10000, num_equipments=50):
    np.random.seed(42)
    random.seed(42)
    
    data = []
    start_date = datetime.now() - timedelta(days=365)
    
    for equipment_id in range(1, num_equipments + 1):
        base_vibration = random.uniform(0.5, 2.0)
        base_temperature = random.uniform(65, 85)
        base_pressure = random.uniform(15, 25)
        
        failure_point = random.randint(int(num_samples * 0.7), num_samples)
        
        for i in range(num_samples // num_equipments):
            timestamp = start_date + timedelta(hours=i)
            
            degradation_factor = 1 + (i / failure_point) * 0.5 if i < failure_point else 1.5
            
            vibration = base_vibration * degradation_factor + np.random.normal(0, 0.1)
            temperature = base_temperature * degradation_factor + np.random.normal(0, 2)
            pressure = base_pressure / degradation_factor + np.random.normal(0, 0.5)
            
            rpm = 1800 - (i / failure_point) * 100 + np.random.normal(0, 10)
            
            oil_quality = 100 - (i / failure_point) * 80 + np.random.normal(0, 2)
            
            power_consumption = 50 * degradation_factor + np.random.normal(0, 2)
            
            time_to_failure = failure_point - i
            failure_soon = 1 if time_to_failure <= 24 and time_to_failure > 0 and i > 50  else 0
            
            data.append({
                'timestamp': timestamp,
                'equipment_id': f'EQ_{equipment_id:03d}',
                'vibration_rms': max(0, vibration),
                'temperature_bearing': max(0, temperature),
                'pressure_oil': max(0, pressure),
                'rpm': max(0, rpm),
                'oil_quality_index': max(0, min(100, oil_quality)),
                'power_consumption': max(0, power_consumption),
                'failure_within_24h': failure_soon
            })
    
    df = pd.DataFrame(data)
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    df.to_csv('data/raw/sensor_data.csv', index=False)
    print(f"Generated {len(df)} sensor readings for {num_equipments} pieces of equipment")
    
    return df

if __name__ == "__main__":
    generate_sensor_data()
