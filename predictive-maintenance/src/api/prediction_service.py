import torch
import numpy as np
import pandas as pd
import pickle
import json
import os
from datetime import datetime

class PredictionService:
    def __init__(self, model_path='models/best_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.equipment_encoder = None
        self.model_metadata = {}
        self._load_artifacts(model_path)
    
    def _load_artifacts(self, model_path):
        artifacts = {
            model_path: self._load_model,
            'models/scaler.pkl': lambda p: setattr(self, 'scaler', self._load_pickle(p)),
            'models/equipment_encoder.pkl': lambda p: setattr(self, 'equipment_encoder', self._load_pickle(p)),
            'models/model_metadata.json': lambda p: setattr(self, 'model_metadata', self._load_json(p))
        }
        
        for path, loader in artifacts.items():
            if os.path.exists(path):
                try:
                    loader(path)
                except Exception:
                    pass
    
    def _load_model(self, path):
        from src.models.lstm_model import PredictiveMaintenanceLSTM
        checkpoint = torch.load(path, map_location=self.device)
        params = checkpoint.get('model_params', {})
        
        self.model = PredictiveMaintenanceLSTM(
            input_size=params.get('input_size', 19),
            hidden_size=params.get('hidden_size', 64),
            num_layers=params.get('num_layers', 2)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device).eval()
    
    def _load_pickle(self, path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    
    def _load_json(self, path):
        with open(path, 'r') as f:
            return json.load(f)
    
    def predict_single(self, data):
        equipment_id = data.get('equipment_id', 'Unknown')
        
        if self.model and self.scaler:
            probability = self._model_predict(data)
        else:
            probability = self._rule_predict(data)
        
        risk_level = self._get_risk_level(probability)
        
        return {
            'equipment_id': equipment_id,
            'failure_probability': round(probability, 4),
            'risk_level': risk_level,
            'prediction_time': datetime.now().isoformat(),
            'recommended_action': self._get_recommendation(risk_level)
        }
    
    def _model_predict(self, data):
        sequence = self._create_sequence(data)
        with torch.no_grad():
            tensor = torch.FloatTensor(sequence).to(self.device)
            if tensor.dim() == 2:
                tensor = tensor.unsqueeze(0)
            output = self.model(tensor)
            return torch.sigmoid(output).cpu().item()
    
    def _rule_predict(self, data):
        risk = 0.0
        
        vibration = data.get('vibration_rms', 1.0)
        temp = data.get('temperature_bearing', 70)
        pressure = data.get('pressure_oil', 15)
        oil_quality = data.get('oil_quality_index', 85)
        
        thresholds = [
            (vibration > 3.0, 0.4), (vibration > 2.5, 0.2), (vibration > 2.0, 0.1),
            (temp > 95, 0.3), (temp > 85, 0.15), (temp > 75, 0.05),
            (oil_quality < 40, 0.3), (oil_quality < 60, 0.15), (oil_quality < 80, 0.05),
            (pressure < 8, 0.2), (pressure < 12, 0.1),
            (vibration > 2.5 and temp > 85, 0.15),
            (oil_quality < 40 and pressure < 15, 0.1)
        ]
        
        for condition, weight in thresholds:
            if condition:
                risk += weight
        
        return min(risk * 0.9, 1.0)
    
    def _create_sequence(self, data):
        features = ['vibration_rms', 'temperature_bearing', 'pressure_oil', 
                   'rpm', 'oil_quality_index', 'power_consumption']
        
        if isinstance(data, dict):
            sequence = np.array([[data.get(f, 0) for f in features]])
            sequence = np.repeat(sequence, 24, axis=0)
        else:
            sequence = np.array([[item.get(f, 0) for f in features] for item in data[:24]])
            if len(sequence) < 24:
                sequence = np.pad(sequence, ((0, 24-len(sequence)), (0, 0)), mode='edge')
        
        if self.scaler:
            sequence = self.scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)
        
        return sequence
    
    def _get_risk_level(self, probability):
        if probability >= 0.8: return 'CRITICAL'
        elif probability >= 0.6: return 'HIGH'
        elif probability >= 0.3: return 'MEDIUM'
        else: return 'LOW'
    
    def _get_recommendation(self, risk_level):
        return {
            'CRITICAL': 'IMMEDIATE SHUTDOWN - Schedule emergency maintenance within 4 hours',
            'HIGH': 'URGENT - Schedule maintenance within 24 hours',
            'MEDIUM': 'CAUTION - Schedule preventive maintenance within 7 days',
            'LOW': 'NORMAL - Continue regular monitoring'
        }.get(risk_level, 'Unknown risk level')
    
    def predict_batch(self, equipment_list):
        return [self.predict_single(equipment) for equipment in equipment_list]
    
    def is_model_loaded(self):
        return self.model is not None
    
    def get_model_info(self):
        info = {
            'model_type': 'PyTorch LSTM + Attention',
            'status': 'Production' if self.model else 'Rule-based',
            'device': str(self.device)
        }
        info.update(self.model_metadata)
        return info