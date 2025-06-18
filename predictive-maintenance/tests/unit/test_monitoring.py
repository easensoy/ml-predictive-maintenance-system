import pytest
import numpy as np
import json
import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch, Mock

from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.data_drift import DataDriftDetector

class TestModelMonitor:
    
    def test_monitor_initialization(self, temp_model_dir):
        metrics_file = f"{temp_model_dir}/test_metrics.json"
        monitor = ModelMonitor(metrics_file=metrics_file)
        
        assert monitor.metrics_file == metrics_file
        assert len(monitor.predictions_log) == 0
        assert len(monitor.performance_metrics) == 0
        assert len(monitor.alerts) == 0
    
    def test_log_prediction_basic(self, temp_model_dir):
        metrics_file = f"{temp_model_dir}/test_metrics.json"
        monitor = ModelMonitor(metrics_file=metrics_file)
        
        monitor.log_prediction(
            equipment_id="TEST_EQ_001",
            prediction=0.75,
            actual=1,
            confidence="HIGH"
        )
        
        assert len(monitor.predictions_log) == 1
        
        log_entry = monitor.predictions_log[0]
        assert log_entry['equipment_id'] == "TEST_EQ_001"
        assert log_entry['prediction'] == 0.75
        assert log_entry['actual'] == 1
        assert log_entry['confidence'] == "HIGH"
        assert 'timestamp' in log_entry
        assert 'day' in log_entry
    
    def test_daily_metrics_calculation(self, temp_model_dir):
        metrics_file = f"{temp_model_dir}/test_metrics.json"
        monitor = ModelMonitor(metrics_file=metrics_file)
        
        test_cases = [
            ("EQ_001", 0.8, 1),
            ("EQ_002", 0.2, 0),
            ("EQ_003", 0.9, 1),
            ("EQ_004", 0.3, 0),
            ("EQ_005", 0.7, 0),
        ]
        
        for eq_id, pred, actual in test_cases:
            monitor.log_prediction(eq_id, pred, actual, "HIGH")
        
        today = datetime.now().strftime('%Y-%m-%d')
        assert today in monitor.performance_metrics
        
        metrics = monitor.performance_metrics[today]
        assert metrics['total_predictions'] == 5
        assert metrics['predictions_with_feedback'] == 5
        assert metrics['accuracy'] == 0.8
    
    def test_alert_generation_high_risk_surge(self, temp_model_dir):
        metrics_file = f"{temp_model_dir}/test_metrics.json"
        monitor = ModelMonitor(metrics_file=metrics_file)
        
        for i in range(6):
            monitor.log_prediction(f"EQ_{i:03d}", 0.85, confidence="HIGH")
        
        high_risk_alerts = [a for a in monitor.alerts if a['type'] == 'HIGH_RISK_SURGE']
        assert len(high_risk_alerts) > 0
    
    def test_system_health_calculation(self, temp_model_dir):
        metrics_file = f"{temp_model_dir}/test_metrics.json"
        monitor = ModelMonitor(metrics_file=metrics_file)
        
        for i in range(20):
            prediction = 0.2 + (i % 3) * 0.2
            monitor.log_prediction(f"EQ_{i:03d}", prediction, confidence="HIGH")
        
        health = monitor.get_system_health()
        
        assert 'status' in health
        assert 'health_score' in health
        assert 'total_predictions_24h' in health
        assert health['health_score'] >= 0
        assert health['health_score'] <= 100
        assert health['status'] in ['HEALTHY', 'WARNING', 'CRITICAL']
    
    @patch('builtins.open', create=True)
    @patch('json.dump')
    def test_metrics_persistence(self, mock_json_dump, mock_open, temp_model_dir):
        metrics_file = f"{temp_model_dir}/test_metrics.json"
        monitor = ModelMonitor(metrics_file=metrics_file)
        
        monitor.log_prediction("EQ_001", 0.5, 1, "HIGH")
        monitor.save_metrics()
        
        mock_open.assert_called_with(metrics_file, 'w')
        mock_json_dump.assert_called_once()
        
        saved_data = mock_json_dump.call_args[0][0]
        assert 'performance_metrics' in saved_data
        assert 'alerts' in saved_data
        assert 'last_updated' in saved_data


class TestDataDriftDetector:
    
    def test_detector_initialization(self):
        detector = DataDriftDetector(reference_window_size=500)
        
        assert detector.reference_window_size == 500
        assert detector.drift_threshold == 0.05
        assert len(detector.feature_names) > 0
        assert len(detector.reference_data) == 0
    
    def test_set_reference_data(self):
        detector = DataDriftDetector()
        
        np.random.seed(42)
        reference_data = np.random.randn(1000, 6)
        
        detector.set_reference_data(reference_data)
        
        assert len(detector.reference_data) == 6
        
        for feature in detector.feature_names:
            if feature in detector.reference_data:
                ref_stats = detector.reference_data[feature]
                assert 'mean' in ref_stats
                assert 'std' in ref_stats
                assert 'min' in ref_stats
                assert 'max' in ref_stats
                assert 'distribution' in ref_stats
    
    def test_no_drift_detection(self):
        detector = DataDriftDetector()
        
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (1000, 6))
        detector.set_reference_data(reference_data)
        
        np.random.seed(123)
        new_data = np.random.normal(0, 1, (100, 6))
        
        result = detector.detect_drift(new_data)
        
        assert 'overall_drift_detected' in result
        assert result['overall_drift_detected'] == False
    
    def test_mean_shift_drift_detection(self):
        detector = DataDriftDetector()
        
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (1000, 6))
        detector.set_reference_data(reference_data)
        
        np.random.seed(123)
        new_data = np.random.normal(2, 1, (100, 6))
        
        result = detector.detect_drift(new_data)
        
        assert result['overall_drift_detected'] == True
        
        features_with_drift = [
            feature for feature, results in result['features'].items()
            if results.get('drift_detected', False)
        ]
        assert len(features_with_drift) > 0
    
    def test_variance_change_detection(self):
        detector = DataDriftDetector()
        
        np.random.seed(42)
        reference_data = np.random.normal(0, 1, (1000, 6))
        detector.set_reference_data(reference_data)
        
        np.random.seed(123)
        new_data = np.random.normal(0, 5, (100, 6))
        
        result = detector.detect_drift(new_data)
        
        assert result['overall_drift_detected'] == True
    
    def test_range_expansion_detection(self):
        detector = DataDriftDetector()
        
        np.random.seed(42)
        reference_data = np.random.uniform(-1, 1, (1000, 6))
        detector.set_reference_data(reference_data)
        
        new_data = np.array([
            [5, 0, 0, 0, 0, 0],
            [-5, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])
        
        result = detector.detect_drift(new_data)
        
        first_feature = detector.feature_names[0]
        if first_feature in result['features']:
            range_test = result['features'][first_feature]['tests'].get('range_drift')
            if range_test:
                assert range_test['drift_detected'] == True
    
    def test_drift_summary_generation(self):
        detector = DataDriftDetector()
        
        drift_result = {
            'overall_drift_detected': True,
            'features': {
                'vibration_rms': {
                    'drift_detected': True,
                    'tests': {
                        'mean_shift': {'drift_detected': True},
                        'variance_change': {'drift_detected': False}
                    }
                },
                'temperature_bearing': {
                    'drift_detected': False,
                    'tests': {}
                }
            }
        }
        
        summary = detector.get_drift_summary(drift_result)
        
        assert isinstance(summary, str)
        assert 'vibration_rms' in summary
        assert 'mean_shift' in summary
        
        no_drift_result = {'overall_drift_detected': False}
        no_drift_summary = detector.get_drift_summary(no_drift_result)
        assert 'No significant data drift detected' in no_drift_summary
    
    def test_adaptive_reference_update(self):
        detector = DataDriftDetector()
        
        np.random.seed(42)
        initial_data = np.random.normal(0, 1, (1000, 6))
        detector.set_reference_data(initial_data)
        
        initial_mean = detector.reference_data[detector.feature_names[0]]['mean']
        
        new_data = np.random.normal(0.1, 1, (100, 6))
        detector.update_reference_data(new_data, adaptive=True)
        
        updated_mean = detector.reference_data[detector.feature_names[0]]['mean']
        
        assert initial_mean < updated_mean < 0.1