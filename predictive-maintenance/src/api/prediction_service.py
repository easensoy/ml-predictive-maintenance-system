import torch
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import os

class PredictionService:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.equipment_encoder = None
        self.model_metadata = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load trained PyTorch model and preprocessing artifacts"""
        try:
            # Load PyTorch model
            if os.path.exists('models/best_model.pth'):
                from src.models.lstm_model import PredictiveMaintenanceLSTM
                
                # Load model checkpoint
                checkpoint = torch.load('models/best_model.pth', map_location=self.device)
                
                # Initialize model with saved parameters
                model_params = checkpoint.get('model_params', {})
                self.model = PredictiveMaintenanceLSTM(
                    input_size=model_params.get('input_size', 19),
                    hidden_size=model_params.get('hidden_size', 64),
                    num_layers=model_params.get('num_layers', 2)
                )
                
                # Load model weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                print("✓ PyTorch model loaded successfully")
            else:
                print("✗ PyTorch model file not found - using rule-based predictions")
            
            # Load preprocessors
            if os.path.exists('models/scaler.pkl'):
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
                print("✓ Scaler loaded")
            
            if os.path.exists('models/equipment_encoder.pkl'):
                with open('models/equipment_encoder.pkl', 'rb') as f:
                    self.equipment_encoder = pickle.load(f)
                print("✓ Equipment encoder loaded")
            
            if os.path.exists('models/model_metadata.json'):
                with open('models/model_metadata.json', 'r') as f:
                    self.model_metadata = json.load(f)
                print("✓ Model metadata loaded")
                    
        except Exception as e:
            print(f"✗ Error loading model artifacts: {e} - using rule-based predictions")
            self.model = None
    
    def predict_single(self, sensor_data):
        """Make prediction for single equipment"""
        try:
            if self.model is not None:
                # Use PyTorch model prediction
                sequence_data = self._create_sequence(sensor_data)
                
                with torch.no_grad():
                    sequence_tensor = torch.FloatTensor(sequence_data).to(self.device)
                    if sequence_tensor.dim() == 2:
                        sequence_tensor = sequence_tensor.unsqueeze(0)  # Add batch dimension
                    
                    output = self.model(sequence_tensor)
                    probability = torch.sigmoid(output).cpu().item()
            else:
                # Fallback to rule-based prediction
                probability = self._rule_based_prediction(sensor_data)
            
            risk_level = self._get_risk_level(probability)
            
            return {
                'equipment_id': sensor_data.get('equipment_id', 'Unknown'),
                'failure_probability': round(probability, 4),
                'risk_level': risk_level,
                'prediction_time': datetime.now().isoformat(),
                'recommended_action': self._get_recommendation(risk_level),
                'model_type': 'PyTorch LSTM' if self.model else 'Rule-based'
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def predict_batch(self, equipment_list):
        """Make predictions for multiple equipment"""
        return [self.predict_single(eq) for eq in equipment_list]
    
    def _create_sequence(self, sensor_data):
        """Create sequence data for PyTorch model input"""
        # Create 24-hour sequence with some variation to simulate historical data
        sequence = []
        base_values = [
            sensor_data.get('vibration_rms', 1.0),
            sensor_data.get('temperature_bearing', 70.0),
            sensor_data.get('pressure_oil', 20.0),
            sensor_data.get('rpm', 1800.0),
            sensor_data.get('oil_quality_index', 80.0),
            sensor_data.get('power_consumption', 50.0)
        ]
        
        for i in range(24):
            hour_data = []
            # Add sensor readings with small temporal variations
            for val in base_values:
                # Add some noise to simulate realistic temporal variation
                variation = np.random.normal(0, val * 0.02)
                hour_data.append(val + variation)
            
            # Add time and equipment features to reach 19 dimensions
            hour_data.extend([
                i,  # hour of day (0-23)
                i % 7,  # day of week (0-6)
                (i // 24) % 30,  # day of month approximation
                1.0,  # equipment type encoding
                0, 1, 0, 0, 0, 0, 0, 0, 0, 0  # one-hot encoded features (padding)
            ])
            sequence.append(hour_data)
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            sequence = self.scaler.transform(sequence)
        
        return np.array(sequence, dtype=np.float32)
    
    def _rule_based_prediction(self, sensor_data):
        """Enhanced rule-based prediction when ML model unavailable"""
        vibration = sensor_data.get('vibration_rms', 1.0)
        temperature = sensor_data.get('temperature_bearing', 70.0)
        oil_quality = sensor_data.get('oil_quality_index', 80.0)
        pressure = sensor_data.get('pressure_oil', 20.0)
        rpm = sensor_data.get('rpm', 1800.0)
        power = sensor_data.get('power_consumption', 50.0)
        
        # Advanced multi-factor risk scoring
        risk_score = 0
        
        # Vibration analysis (most critical factor)
        if vibration > 3.5: risk_score += 0.4
        elif vibration > 3.0: risk_score += 0.3
        elif vibration > 2.5: risk_score += 0.2
        elif vibration > 2.0: risk_score += 0.1
        
        # Temperature analysis
        if temperature > 100: risk_score += 0.3
        elif temperature > 90: risk_score += 0.2
        elif temperature > 80: risk_score += 0.1
        
        # Oil quality analysis (inverse relationship)
        if oil_quality < 20: risk_score += 0.3
        elif oil_quality < 40: risk_score += 0.2
        elif oil_quality < 60: risk_score += 0.1
        
        # Pressure analysis (low pressure indicates problems)
        if pressure < 8: risk_score += 0.2
        elif pressure < 12: risk_score += 0.15
        elif pressure < 16: risk_score += 0.1
        
        # RPM deviation analysis
        normal_rpm = 1800
        rpm_deviation = abs(rpm - normal_rpm) / normal_rpm
        if rpm_deviation > 0.2: risk_score += 0.15
        elif rpm_deviation > 0.1: risk_score += 0.1
        elif rpm_deviation > 0.05: risk_score += 0.05
        
        # Power consumption analysis
        normal_power = 50
        power_deviation = abs(power - normal_power) / normal_power
        if power_deviation > 0.3: risk_score += 0.1
        elif power_deviation > 0.2: risk_score += 0.05
        
        # Combined risk factors (multiplicative effects)
        if vibration > 2.5 and temperature > 85:
            risk_score += 0.15  # Combined thermal and mechanical stress
        
        if oil_quality < 40 and pressure < 15:
            risk_score += 0.1  # Lubrication system failure
        
        # Cap maximum risk and apply sigmoid-like smoothing
        risk_score = min(risk_score, 1.0)
        return risk_score * 0.9  # Scale to prevent overconfident predictions
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level categories"""
        if probability >= 0.8: return 'CRITICAL'
        elif probability >= 0.6: return 'HIGH'
        elif probability >= 0.3: return 'MEDIUM'
        else: return 'LOW'
    
    def _get_recommendation(self, risk_level):
        """Get maintenance recommendation based on risk level"""
        recommendations = {
            'CRITICAL': 'IMMEDIATE SHUTDOWN - Schedule emergency maintenance within 4 hours',
            'HIGH': 'URGENT - Schedule maintenance within 24 hours', 
            'MEDIUM': 'CAUTION - Schedule preventive maintenance within 7 days',
            'LOW': 'NORMAL - Continue regular monitoring'
        }
        return recommendations.get(risk_level, 'Unknown risk level')
    
    def get_model_info(self):
        """Return model metadata and performance information"""
        base_info = {
            'model_type': 'PyTorch LSTM + Attention Neural Network',
            'framework': 'PyTorch',
            'prediction_horizon': '24 hours advance warning',
            'input_features': 19,
            'sequence_length': 24,
            'device': str(self.device),
            'status': 'Production Ready' if self.model else 'Rule-based Mode'
        }
        
        if self.model_metadata:
            base_info.update(self.model_metadata)
        
        return base_info