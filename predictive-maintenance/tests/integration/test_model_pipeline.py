import pytest
import numpy as np
import torch
import tempfile
import os
import time
from unittest.mock import patch

from src.utils.data_preprocessing import DataPreprocessor
from src.models.lstm_model import PredictiveMaintenanceLSTM
from src.api.prediction_service import PredictionService
from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.data_drift import DataDriftDetector

class TestModelPipeline:
    
    @pytest.mark.integration
    def test_end_to_end_prediction_pipeline(self, sample_sensor_data, temp_model_dir):
        data_file = os.path.join(temp_model_dir, 'pipeline_test_data.csv')
        sample_sensor_data.to_csv(data_file, index=False)
        
        original_cwd = os.getcwd()
        os.chdir(temp_model_dir)
        
        try:
            preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test = preprocessor.prepare_data(data_file)
            
            input_size = X_train.shape[2]
            model = PredictiveMaintenanceLSTM(input_size=input_size, hidden_size=16)
            
            model_checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': input_size,
                    'hidden_size': 16,
                    'num_layers': 2
                },
                'timestamp': '2023-01-01T00:00:00'
            }
            
            torch.save(model_checkpoint, 'models/best_model.pth')
            
            training_history = {
                'timestamp': '2023-01-01T00:00:00',
                'val_metrics': [{'accuracy': 0.85, 'f1_score': 0.82}]
            }
            
            import json
            with open('models/training_history.json', 'w') as f:
                json.dump(training_history, f)
            
            prediction_service = PredictionService()
            
            assert prediction_service.is_model_loaded() == True
            
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
            
            result = prediction_service.predict_single('TEST_EQ_001', sensor_data)
            
            assert 'equipment_id' in result
            assert 'failure_probability' in result
            assert 'risk_level' in result
            assert 'recommendation' in result
            
            assert result['equipment_id'] == 'TEST_EQ_001'
            assert 0 <= result['failure_probability'] <= 1
            assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_monitoring_integration(self, temp_model_dir):
        monitor = ModelMonitor(metrics_file=f"{temp_model_dir}/monitor_test.json")
        
        prediction_scenarios = [
            ('EQ_001', 0.2, 0, 'HIGH'),
            ('EQ_002', 0.8, 1, 'HIGH'),
            ('EQ_003', 0.7, 0, 'HIGH'),
            ('EQ_004', 0.3, 1, 'MEDIUM'),
            ('EQ_005', 0.9, 1, 'HIGH'),
        ]
        
        for eq_id, prediction, actual, confidence in prediction_scenarios:
            monitor.log_prediction(eq_id, prediction, actual, confidence)
        
        assert len(monitor.predictions_log) == 5
        
        health = monitor.get_system_health()
        assert 'status' in health
        assert 'health_score' in health
        assert 0 <= health['health_score'] <= 100
        
        summary = monitor.get_performance_summary(days=1)
        assert summary['total_predictions'] == 5
        assert summary['predictions_with_feedback'] == 5
    
    @pytest.mark.integration
    def test_data_drift_integration(self, sample_sensor_data):
        feature_columns = ['vibration_rms', 'temperature_bearing', 'pressure_oil',
                          'rpm', 'oil_quality_index', 'power_consumption']
        
        reference_data = sample_sensor_data[feature_columns].values
        
        drift_detector = DataDriftDetector()
        drift_detector.set_reference_data(reference_data)
        
        similar_data = reference_data + np.random.normal(0, 0.1, reference_data.shape)
        drift_result = drift_detector.detect_drift(similar_data)
        
        assert drift_result['overall_drift_detected'] == False
        
        shifted_data = reference_data + 2.0
        drift_result_shifted = drift_detector.detect_drift(shifted_data)
        
        assert drift_result_shifted['overall_drift_detected'] == True
        
        summary = drift_detector.get_drift_summary(drift_result_shifted)
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    @pytest.mark.integration
    def test_model_consistency_across_restarts(self, sample_sensor_data, temp_model_dir):
        data_file = os.path.join(temp_model_dir, 'consistency_test_data.csv')
        sample_sensor_data.to_csv(data_file, index=False)
        
        original_cwd = os.getcwd()
        os.chdir(temp_model_dir)
        
        try:
            preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test = preprocessor.prepare_data(data_file)
            
            input_size = X_train.shape[2]
            model = PredictiveMaintenanceLSTM(input_size=input_size, hidden_size=16)
            
            model_checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': input_size,
                    'hidden_size': 16,
                    'num_layers': 2
                }
            }
            torch.save(model_checkpoint, 'models/best_model.pth')
            
            import json
            with open('models/training_history.json', 'w') as f:
                json.dump({'timestamp': '2023-01-01', 'val_metrics': [{}]}, f)
            
            service1 = PredictionService()
            
            test_sensor_data = [
                {
                    'vibration_rms': 1.5,
                    'temperature_bearing': 75.0,
                    'pressure_oil': 20.0,
                    'rpm': 1800,
                    'oil_quality_index': 85,
                    'power_consumption': 50.0
                }
            ] * 24
            
            result1 = service1.predict_single('TEST_EQ', test_sensor_data)
            
            service2 = PredictionService()
            result2 = service2.predict_single('TEST_EQ', test_sensor_data)
            
            assert result1['failure_probability'] == result2['failure_probability']
            assert result1['risk_level'] == result2['risk_level']
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.integration 
    def test_error_propagation_and_recovery(self, temp_model_dir):
        service = PredictionService(model_path='nonexistent_model.pth')
        
        assert service.is_model_loaded() == False
        
        with pytest.raises(RuntimeError):
            service.predict_single('TEST_EQ', [{}] * 24)
        
        monitor = ModelMonitor(metrics_file=f"{temp_model_dir}/error_test.json")
        
        monitor.log_prediction('EQ_001', 0.5)
        assert len(monitor.predictions_log) == 1
        
        health = monitor.get_system_health()
        assert isinstance(health, dict)
        assert 'status' in health
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_performance_under_load(self, temp_model_dir):
        original_cwd = os.getcwd()
        os.chdir(temp_model_dir)
        
        try:
            input_size = 13
            model = PredictiveMaintenanceLSTM(input_size=input_size, hidden_size=16)
            
            model_checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'input_size': input_size,
                    'hidden_size': 16,
                    'num_layers': 2
                }
            }
            torch.save(model_checkpoint, 'models/best_model.pth')
            
            import joblib
            from sklearn.preprocessing import StandardScaler, LabelEncoder
            
            scaler = StandardScaler()
            scaler.scale_ = np.ones(input_size)
            scaler.mean_ = np.zeros(input_size)
            scaler.var_ = np.ones(input_size)
            joblib.dump(scaler, 'models/scaler.pkl')
            
            encoder = LabelEncoder()
            encoder.classes_ = np.array(['EQ_001'])
            joblib.dump(encoder, 'models/equipment_encoder.pkl')
            
            import json
            with open('models/training_history.json', 'w') as f:
                json.dump({'timestamp': '2023-01-01', 'val_metrics': [{}]}, f)
            
            service = PredictionService()
            
            num_predictions = 100
            start_time = time.time()
            
            for i in range(num_predictions):
                sensor_data = [{'vibration_rms': 1.5 + i * 0.01}] * 24
                result = service.predict_single(f'EQ_{i:03d}', sensor_data)
                
                assert 0 <= result['failure_probability'] <= 1
                assert result['risk_level'] in ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            
            end_time = time.time()
            total_time = end_time - start_time
            avg_time_per_prediction = total_time / num_predictions
            
            assert avg_time_per_prediction < 0.1
            
        finally:
            os.chdir(original_cwd)