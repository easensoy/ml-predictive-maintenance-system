import torch
import numpy as np
import joblib
import json
import os
from datetime import datetime
import logging
from typing import Dict, List, Any

from ..models.lstm_model import PredictiveMaintenanceLSTM

logger = logging.getLogger(__name__)

class PredictionService:
    def __init__(self, model_path='models/best_model.pth'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.equipment_encoder = None
        self.model_metadata = None
        self.prediction_count = 0
        self.load_model()
    
    def load_model(self):
        try:
            if not os.path.exists(self.model_path):
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            checkpoint = torch.load(self.model_path, map_location='cpu')
            model_config = checkpoint['model_config']
            
            self.model = PredictiveMaintenanceLSTM(
                input_size=model_config['input_size'],
                hidden_size=model_config['hidden_size'],
                num_layers=model_config['num_layers']
            )
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            self.scaler = joblib.load('models/scaler.pkl')
            self.equipment_encoder = joblib.load('models/equipment_encoder.pkl')
            
            with open('models/training_history.json', 'r') as f:
                history = json.load(f)
                self.model_metadata = {
                    'training_timestamp': history['timestamp'],
                    'final_val_metrics': history['val_metrics'][-1] if history['val_metrics'] else None
                }
            
            logger.info("Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
    
    def is_model_loaded(self):
        return self.model is not None
    
    def preprocess_sensor_data(self, sensor_data: List[Dict]) -> np.ndarray:
        feature_names = [
            'vibration_rms', 'temperature_bearing', 'pressure_oil',
            'rpm', 'oil_quality_index', 'power_consumption'
        ]
        
        features = []
        for reading in sensor_data:
            feature_row = [reading.get(name, 0) for name in feature_names]
            features.append(feature_row)
        
        features = np.array(features)
        
        if len(features) >= 6:
            rolling_features = []
            for i in range(len(features)):
                start_idx = max(0, i - 5)
                window = features[start_idx:i+1]
                rolling_mean = np.mean(window, axis=0)
                rolling_std = np.std(window, axis=0)
                combined = np.concatenate([features[i], rolling_mean, rolling_std])
                rolling_features.append(combined)
            features = np.array(rolling_features)
        else:
            rolling_stats = np.zeros((len(features), len(feature_names) * 2))
            features = np.concatenate([features, rolling_stats], axis=1)
        
        equipment_col = np.zeros((len(features), 1))
        features = np.concatenate([features, equipment_col], axis=1)
        
        return features
    
    def predict_single(self, equipment_id: str, sensor_data: List[Dict]) -> Dict[str, Any]:
        if not self.is_model_loaded():
            raise RuntimeError("Model not loaded")
        
        try:
            features = self.preprocess_sensor_data(sensor_data)
            sequence_length = 24
            if len(features) < sequence_length:
                last_reading = features[-1] if len(features) > 0 else np.zeros(features.shape[1])
                padding = np.tile(last_reading, (sequence_length - len(features), 1))
                features = np.vstack([padding, features])
            elif len(features) > sequence_length:
                features = features[-sequence_length:]
            
            features_scaled = self.scaler.transform(features)
            input_tensor = torch.FloatTensor(features_scaled).unsqueeze(0)
            
            with torch.no_grad():
                prediction = self.model(input_tensor)
                failure_probability = prediction.item()
            
            if failure_probability >= 0.8:
                risk_level = "CRITICAL"
                recommendation = "Immediate maintenance required"
            elif failure_probability >= 0.6:
                risk_level = "HIGH"
                recommendation = "Schedule maintenance within 24 hours"
            elif failure_probability >= 0.3:
                risk_level = "MEDIUM"
                recommendation = "Monitor closely, schedule maintenance soon"
            else:
                risk_level = "LOW"
                recommendation = "Normal operation"
            
            self.prediction_count += 1
            
            result = {
                'equipment_id': equipment_id,
                'failure_probability': round(failure_probability, 4),
                'risk_level': risk_level,
                'recommendation': recommendation,
                'prediction_timestamp': datetime.now().isoformat(),
                'data_points_used': len(sensor_data),
                'confidence': 'HIGH' if len(sensor_data) >= 24 else 'MEDIUM'
            }
            
            return result
        except Exception as e:
            logger.error(f"Prediction error for {equipment_id}: {str(e)}")
            raise
    
    def get_model_metrics(self) -> Dict[str, Any]:
        if not self.model_metadata:
            return {'error': 'No model metadata available'}
        
        return {
            'model_info': {
                'training_date': self.model_metadata.get('training_timestamp'),
                'predictions_made': self.prediction_count,
                'status': 'active'
            },
            'performance_metrics': self.model_metadata.get('final_val_met
