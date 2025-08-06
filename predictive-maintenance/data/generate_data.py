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
        
        # Create multiple failure cycles per equipment
        samples_per_equipment = num_samples // num_equipments
        failure_cycles = random.randint(2, 4)  # 2-4 failure cycles per equipment
        
        for cycle in range(failure_cycles):
            cycle_start = cycle * (samples_per_equipment // failure_cycles)
            cycle_length = samples_per_equipment // failure_cycles
            failure_point = cycle_start + random.randint(int(cycle_length * 0.6), cycle_length - 30)
            
            for i in range(cycle_start, min(cycle_start + cycle_length, samples_per_equipment)):
                timestamp = start_date + timedelta(hours=i)
                
                # Calculate degradation - increases as we approach failure
                time_to_failure = failure_point - i
                if time_to_failure > 0:
                    degradation_factor = 1 + (1 - (time_to_failure / cycle_length)) * 1.5
                else:
                    degradation_factor = 2.5  # Failed state
                
                # Generate sensor readings with degradation
                vibration = base_vibration * degradation_factor + np.random.normal(0, 0.2)
                temperature = base_temperature * degradation_factor + np.random.normal(0, 3)
                pressure = base_pressure / degradation_factor + np.random.normal(0, 1)
                rpm = 1800 - (degradation_factor - 1) * 150 + np.random.normal(0, 20)
                oil_quality = 100 - (degradation_factor - 1) * 40 + np.random.normal(0, 5)
                power_consumption = 50 * degradation_factor + np.random.normal(0, 3)
                
                # Label as failure if within 24 hours of failure point
                failure_within_24h = 1 if 0 < time_to_failure <= 24 else 0
                
                data.append({
                    'timestamp': timestamp,
                    'equipment_id': f'EQ_{equipment_id:03d}',
                    'vibration_rms': max(0.1, vibration),
                    'temperature_bearing': max(20, temperature),
                    'pressure_oil': max(1, pressure),
                    'rpm': max(500, rpm),
                    'oil_quality_index': max(0, min(100, oil_quality)),
                    'power_consumption': max(10, power_consumption),
                    'failure_within_24h': failure_within_24h
                })
    
    df = pd.DataFrame(data)
    
    # Ensure we have a reasonable failure rate (5-15%)
    failure_rate = df['failure_within_24h'].mean()
    print(f"Generated failure rate: {failure_rate:.2%}")
    
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    df.to_csv('data/raw/sensor_data.csv', index=False)
    print(f"Generated {len(df)} sensor readings for {num_equipments} pieces of equipment")
    print(f"Failure samples: {df['failure_within_24h'].sum()}")
    
    return df

if __name__ == "__main__":
    generate_sensor_data()