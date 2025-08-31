import torch
import numpy as np
import pandas as pd
import pickle
import json
import os
import yaml
from datetime import datetime

class PredictionService:
    def __init__(self, model_path=None, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = None
        
        model_path = model_path or (self.config['paths']['model'] if self.config else 'models/best_model.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.equipment_encoder = None
        self.model_metadata = {}
        self._load_artifacts(model_path)
    
    def _load_artifacts(self, model_path):
        if self.config:
            paths = self.config['paths']
            artifacts = {
                model_path: self._load_model,
                paths['scaler']: lambda p: setattr(self, 'scaler', self._load_pickle(p)),
                paths['equipment_encoder']: lambda p: setattr(self, 'equipment_encoder', self._load_pickle(p)),
                paths['model_metadata']: lambda p: setattr(self, 'model_metadata', self._load_json(p))
            }
        else:
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
        
        if self.config:
            defaults = self.config['model_defaults']
            self.model = PredictiveMaintenanceLSTM(
                input_size=params.get('input_size', defaults['input_size']),
                hidden_size=params.get('hidden_size', defaults['hidden_size']),
                num_layers=params.get('num_layers', defaults['num_layers'])
            )
        else:
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
            'failure_probability': round(probability, self.config['output']['probability_decimals'] if self.config else 4),
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
        
        if self.config:
            defaults = self.config['rule_based']['defaults']
            vibration = data.get('vibration_rms', defaults['vibration_rms'])
            temp = data.get('temperature_bearing', defaults['temperature_bearing'])
            pressure = data.get('pressure_oil', defaults['pressure_oil'])
            oil_quality = data.get('oil_quality_index', defaults['oil_quality_index'])
        else:
            vibration = data.get('vibration_rms', 1.0)
            temp = data.get('temperature_bearing', 70)
            pressure = data.get('pressure_oil', 15)
            oil_quality = data.get('oil_quality_index', 85)
        
        if self.config:
            thresholds_config = self.config['rule_based']['thresholds']
            
            # Vibration thresholds
            for threshold in thresholds_config['vibration']:
                if vibration > threshold['condition']:
                    risk += threshold['weight']
            
            # Temperature thresholds
            for threshold in thresholds_config['temperature']:
                if temp > threshold['condition']:
                    risk += threshold['weight']
            
            # Oil quality thresholds
            for threshold in thresholds_config['oil_quality']:
                if oil_quality < threshold['condition']:
                    risk += threshold['weight']
            
            # Pressure thresholds
            for threshold in thresholds_config['pressure']:
                if pressure < threshold['condition']:
                    risk += threshold['weight']
            
            # Combined conditions
            combined_vib_temp = thresholds_config['combined_vibration_temp']
            if (vibration > combined_vib_temp['vibration_threshold'] and 
                temp > combined_vib_temp['temperature_threshold']):
                risk += combined_vib_temp['weight']
            
            combined_oil_press = thresholds_config['combined_oil_pressure']
            if (oil_quality < combined_oil_press['oil_quality_threshold'] and 
                pressure < combined_oil_press['pressure_threshold']):
                risk += combined_oil_press['weight']
            
            multiplier = self.config['rule_based']['risk_multiplier']
            max_risk = self.config['rule_based']['max_risk']
            return min(risk * multiplier, max_risk)
        else:
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
        if self.config:
            features = self.config['sequence']['features']
            seq_length = self.config['sequence']['sequence_length']
            padding_mode = self.config['sequence']['padding_mode']
        else:
            features = ['vibration_rms', 'temperature_bearing', 'pressure_oil', 
                       'rpm', 'oil_quality_index', 'power_consumption']
            seq_length = 24
            padding_mode = 'edge'
        
        if isinstance(data, dict):
            sequence = np.array([[data.get(f, 0) for f in features]])
            sequence = np.repeat(sequence, seq_length, axis=0)
        else:
            sequence = np.array([[item.get(f, 0) for f in features] for item in data[:seq_length]])
            if len(sequence) < seq_length:
                sequence = np.pad(sequence, ((0, seq_length-len(sequence)), (0, 0)), mode=padding_mode)
        
        if self.scaler:
            sequence = self.scaler.transform(sequence.reshape(-1, sequence.shape[-1])).reshape(sequence.shape)
        
        return sequence
    
    def _get_risk_level(self, probability):
        if self.config:
            levels = self.config['risk_levels']
            if probability >= levels['critical']: return 'CRITICAL'
            elif probability >= levels['high']: return 'HIGH'
            elif probability >= levels['medium']: return 'MEDIUM'
            else: return 'LOW'
        else:
            if probability >= 0.8: return 'CRITICAL'
            elif probability >= 0.6: return 'HIGH'
            elif probability >= 0.3: return 'MEDIUM'
            else: return 'LOW'
    
    def _get_recommendation(self, risk_level):
        if self.config:
            recommendations = self.config['recommendations']
            return recommendations.get(risk_level, 'Unknown risk level')
        else:
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