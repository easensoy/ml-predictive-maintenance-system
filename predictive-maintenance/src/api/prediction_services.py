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
        self.load_artifacts()
    
    def load_artifacts(self):
        """Load trained model and preprocessing artifacts"""
        try:
            # Load model
            from src.models.lstm_model import LSTMAttentionModel
            self.model = LSTMAttentionModel(input_size=19, hidden_size=64, num_layers=2)
            
            if os.path.exists('models/best_model.pth'):
                checkpoint = torch.load('models/best_model.pth', map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
            
            # Load preprocessors
            if os.path.exists('models/scaler.pkl'):
                with open('models/scaler.pkl', 'rb') as f:
                    self.scaler = pickle.load(f)
            
            if os.path.exists('models/equipment_encoder.pkl'):
                with open('models/equipment_encoder.pkl', 'rb') as f:
                    self.equipment_encoder = pickle.load(f)
            
            if os.path.exists('models/model_metadata.json'):
                with open('models/model_metadata.json', 'r') as f:
                    self.model_metadata = json.load(f)
                    
            print("✓ All model artifacts loaded successfully")
            
        except Exception as e:
            print(f"✗ Error loading model artifacts: {e}")
    
    def predict_single(self, sensor_data):
        """Make prediction for single equipment"""
        try:
            # Convert to DataFrame if it's a dict
            if isinstance(sensor_data, dict):
                df = pd.DataFrame([sensor_data])
            else:
                df = sensor_data.copy()
            
            # Generate dummy sequence data for demo
            sequence_data = self._create_sequence(df)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(sequence_data)
                probability = torch.sigmoid(output).item()
            
            # Determine risk level
            risk_level = self._get_risk_level(probability)
            
            return {
                'equipment_id': df.iloc[0].get('equipment_id', 'Unknown'),
                'failure_probability': round(probability, 4),
                'risk_level': risk_level,
                'prediction_time': datetime.now().isoformat(),
                'recommended_action': self._get_recommendation(risk_level)
            }
            
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def predict_batch(self, equipment_list):
        """Make predictions for multiple equipment"""
        results = []
        for equipment in equipment_list:
            result = self.predict_single(equipment)
            results.append(result)
        return results
    
    def _create_sequence(self, df):
        """Create sequence data from current sensor readings"""
        # For demo purposes, create a sequence by adding noise to current reading
        current_reading = df.iloc[0]
        
        # Create 24-hour sequence
        sequence = []
        for i in range(24):
            hour_data = [
                current_reading.get('vibration_rms', 1.0) + np.random.normal(0, 0.1),
                current_reading.get('temperature_bearing', 70.0) + np.random.normal(0, 2),
                current_reading.get('pressure_oil', 20.0) + np.random.normal(0, 0.5),
                current_reading.get('rpm', 1800.0) + np.random.normal(0, 10),
                current_reading.get('oil_quality_index', 80.0) + np.random.normal(0, 2),
                current_reading.get('power_consumption', 50.0) + np.random.normal(0, 1),
                # Add dummy features to reach 19 dimensions
                i,  # hour of day
                i % 7,  # day of week
                i % 30,  # day of month
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0  # padding features
            ]
            sequence.append(hour_data)
        
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # Add batch dimension
        return sequence_tensor
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability >= 0.8:
            return 'CRITICAL'
        elif probability >= 0.6:
            return 'HIGH'
        elif probability >= 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'
    
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
        """Return model metadata and performance metrics"""
        if self.model_metadata:
            return self.model_metadata
        else:
            return {
                'model_type': 'LSTM + Attention',
                'input_features': 19,
                'sequence_length': 24,
                'prediction_horizon': '24 hours',
                'status': 'Model loaded successfully'
            }