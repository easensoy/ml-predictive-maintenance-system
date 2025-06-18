import pytest
import torch
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock

from src.api.prediction_service import PredictionService

class TestPredictionService:
    
    @patch('torch.load')
    @patch('joblib.load')
    def test_service_initialization_success(self, mock_joblib_load, mock_torch_load, temp_model_dir):
        mock_model_checkpoint = {
            'model_state_dict': {},
            'model_config': {
                'input_size': 13,
                'hidden_size': 64,
                'num_layers': 2
            }
        }
        mock_torch_load.return_value = mock_model_checkpoint
        
        mock_scaler = Mock()
        mock_encoder = Mock()
        mock_joblib_load.side_effect = [mock_scaler, mock_encoder]
        
        mock_history = {
            'timestamp': '2023-01-01T00:00:00',
            'val_metrics': [{'accuracy': 0.85, 'f1_score': 0.82}]
        }
        
        with patch('builtins.open', create=True) as mock_open:
            with patch('json.load', return_value=mock_history):
                service = PredictionService(model_path='fake_path.pth')
        
        assert service.is_model_loaded() == True
        assert service.scaler == mock_scaler
        assert service.equipment_encoder == mock_encoder
    
    def test_service_initialization_failure(self):
        service = PredictionService(model_path='nonexistent_model.pth')
        
        assert service.is_model_loaded() == False
        assert service.model is None
    
    def test_preprocess_sensor_data_basic(self):
        service = PredictionService()
        
        sensor_data = [
            {
                'vibration_rms': 1.5,
                'temperature_bearing': 75.0,
                'pressure_oil': 20.0,
                'rpm': 1800,
                'oil_quality_index': 85,
                'power_consumption': 50.0
            }
        ] * 24
        
        result = service.preprocess_sensor_data(sensor_data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 24
        assert result.shape[1] > 6
    
    def test_preprocess_sensor_data_insufficient_data(self):
        service = PredictionService()
        
        sensor_data = [
            {
                'vibration_rms': 1.5,
                'temperature_bearing': 75.0,
                'pressure_oil': 20.0,
                'rpm': 1800,
                'oil_quality_index': 85,
                'power_consumption': 50.0
            }
        ] * 12
        
        result = service.preprocess_sensor_data(sensor_data)
        
        assert result.shape[0] == 24
    
    def test_preprocess_sensor_data_missing_fields(self):
        service = PredictionService()
        
        sensor_data = [
            {
                'vibration_rms': 1.5,
                'temperature_bearing': 75.0,
            }
        ] * 24
        
        result = service.preprocess_sensor_data(sensor_data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 24
    
    @patch('torch.no_grad')
    def test_predict_single_success(self, mock_no_grad):
        service = PredictionService()
        service.model = Mock()
        service.scaler = Mock()
        service.equipment_encoder = Mock()
        
        mock_prediction = torch.tensor([0.75])
        service.model.return_value = mock_prediction
        
        service.scaler.transform.return_value = np.random.randn(24, 13)
        
        equipment_id = "TEST_EQ_001"
        sensor_data = [
            {
                'vibration_rms': 1.5,
                'temperature_bearing': 75.0,
                'pressure_oil': 20.0,
                'rpm': 1800,
                'oil_quality_index': 85,
                'power_consumption': 50.0
            }
        ] * 24
        
        result = service.predict_single(equipment_id, sensor_data)
        
        assert 'equipment_id' in result
        assert 'failure_probability' in result
        assert 'risk_level' in result
        assert 'recommendation' in result
        assert 'prediction_timestamp' in result
        
        assert result['equipment_id'] == equipment_id
        assert result['failure_probability'] == 0.75
        assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
    
    def test_predict_single_model_not_loaded(self):
        service = PredictionService()
        
        equipment_id = "TEST_EQ_001"
        sensor_data = [{'vibration_rms': 1.5}] * 24
        
        with pytest.raises(RuntimeError, match="Model not loaded"):
            service.predict_single(equipment_id, sensor_data)
    
    def test_risk_level_categorization(self):
        service = PredictionService()
        service.model = Mock()
        service.scaler = Mock()
        service.equipment_encoder = Mock()
        service.scaler.transform.return_value = np.random.randn(24, 13)
        
        test_cases = [
            (0.1, 'LOW'),
            (0.4, 'MEDIUM'),
            (0.7, 'HIGH'),
            (0.9, 'CRITICAL')
        ]
        
        for probability, expected_risk in test_cases:
            service.model.return_value = torch.tensor([probability])
            
            result = service.predict_single("TEST_EQ", [{}] * 24)
            assert result['risk_level'] == expected_risk
    
    def test_prediction_counter(self):
        service = PredictionService()
        service.model = Mock()
        service.scaler = Mock()
        service.equipment_encoder = Mock()
        service.scaler.transform.return_value = np.random.randn(24, 13)
        service.model.return_value = torch.tensor([0.5])
        
        initial_count = service.prediction_count
        
        service.predict_single("TEST_EQ", [{}] * 24)
        
        assert service.prediction_count == initial_count + 1
    
    def test_get_model_metrics(self):
        service = PredictionService()
        service.model_metadata = {
            'training_timestamp': '2023-01-01T00:00:00',
            'final_val_metrics': {'accuracy': 0.85, 'f1_score': 0.82}
        }
        service.prediction_count = 100
        
        metrics = service.get_model_metrics()
        
        assert 'model_info' in metrics
        assert 'performance_metrics' in metrics
        assert 'system_metrics' in metrics
        
        assert metrics['model_info']['predictions_made'] == 100
        assert metrics['performance_metrics']['accuracy'] == 0.85
        assert metrics['system_metrics']['model_loaded'] == False
    
    def test_confidence_assessment(self):
        service = PredictionService()
        service.model = Mock()
        service.scaler = Mock()
        service.equipment_encoder = Mock()
        service.scaler.transform.return_value = np.random.randn(24, 13)
        service.model.return_value = torch.tensor([0.5])
        
        full_data = [{}] * 24
        result_full = service.predict_single("TEST_EQ", full_data)
        assert result_full['confidence'] == 'HIGH'
        
        partial_data = [{}] * 12
        result_partial = service.predict_single("TEST_EQ", partial_data)
        assert result_partial['confidence'] == 'MEDIUM'