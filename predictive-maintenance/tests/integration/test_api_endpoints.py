import pytest
import json
import os
from unittest.mock import patch, Mock

class TestAPIEndpoints:
    
    def test_health_endpoint(self, flask_test_client):
        response = flask_test_client.get('/api/health')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'timestamp' in data
        assert 'model_loaded' in data
        assert 'version' in data
        
        assert data['status'] == 'healthy'
        assert isinstance(data['model_loaded'], bool)
    
    @patch('src.api.prediction_service.PredictionService.predict_single')
    def test_prediction_endpoint_success(self, mock_predict, flask_test_client, sample_api_data):
        mock_predict.return_value = {
            'equipment_id': 'TEST_EQ_001',
            'failure_probability': 0.75,
            'risk_level': 'HIGH',
            'recommendation': 'Schedule maintenance within 24 hours',
            'prediction_timestamp': '2023-01-01T12:00:00',
            'data_points_used': 24,
            'confidence': 'HIGH'
        }
        
        response = flask_test_client.post(
            '/api/predict',
            data=json.dumps(sample_api_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'equipment_id' in data
        assert 'failure_probability' in data
        assert 'risk_level' in data
        assert 'recommendation' in data
        
        assert data['equipment_id'] == 'TEST_EQ_001'
        assert data['failure_probability'] == 0.75
        assert data['risk_level'] == 'HIGH'
    
    def test_prediction_endpoint_missing_data(self, flask_test_client):
        response = flask_test_client.post('/api/predict')
        assert response.status_code == 400
        
        data = json.loads(response.data)
        assert 'error' in data
        
        incomplete_data = {'equipment_id': 'TEST_EQ_001'}
        
        response = flask_test_client.post(
            '/api/predict',
            data=json.dumps(incomplete_data),
            content_type='application/json'
        )
        
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
        assert 'Missing required fields' in data['error']
    
    @patch('src.api.prediction_service.PredictionService.predict_single')
    def test_prediction_endpoint_service_error(self, mock_predict, flask_test_client, sample_api_data):
        mock_predict.side_effect = Exception("Model not loaded")
        
        response = flask_test_client.post(
            '/api/predict',
            data=json.dumps(sample_api_data),
            content_type='application/json'
        )
        
        assert response.status_code == 500
        
        data = json.loads(response.data)
        assert 'error' in data
        assert data['error'] == 'Internal server error'
    
    @patch('src.api.prediction_service.PredictionService.predict_single')
    def test_batch_prediction_endpoint(self, mock_predict, flask_test_client):
        mock_predict.side_effect = [
            {
                'equipment_id': 'EQ_001',
                'failure_probability': 0.3,
                'risk_level': 'LOW',
                'recommendation': 'Normal operation'
            },
            {
                'equipment_id': 'EQ_002',
                'failure_probability': 0.8,
                'risk_level': 'CRITICAL',
                'recommendation': 'Immediate maintenance required'
            }
        ]
        
        batch_data = {
            'equipments': [
                {
                    'equipment_id': 'EQ_001',
                    'sensor_data': [{'vibration_rms': 1.0}] * 24
                },
                {
                    'equipment_id': 'EQ_002',
                    'sensor_data': [{'vibration_rms': 3.0}] * 24
                }
            ]
        }
        
        response = flask_test_client.post(
            '/api/predict/batch',
            data=json.dumps(batch_data),
            content_type='application/json'
        )
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'results' in data
        assert len(data['results']) == 2
        
        results = data['results']
        assert results[0]['equipment_id'] == 'EQ_001'
        assert results[0]['risk_level'] == 'LOW'
        assert results[1]['equipment_id'] == 'EQ_002'
        assert results[1]['risk_level'] == 'CRITICAL'
    
    @patch('src.api.prediction_service.PredictionService.get_model_metrics')
    def test_model_metrics_endpoint(self, mock_get_metrics, flask_test_client):
        mock_metrics = {
            'model_info': {
                'training_date': '2023-01-01T00:00:00',
                'predictions_made': 1000,
                'status': 'active'
            },
            'performance_metrics': {
                'accuracy': 0.87,
                'precision': 0.84,
                'recall': 0.91,
                'f1_score': 0.87
            },
            'system_metrics': {
                'model_loaded': True,
                'last_prediction': '2023-01-01T12:00:00'
            }
        }
        
        mock_get_metrics.return_value = mock_metrics
        
        response = flask_test_client.get('/api/model/metrics')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'model_info' in data
        assert 'performance_metrics' in data
        assert 'system_metrics' in data
        
        assert data['model_info']['predictions_made'] == 1000
        assert data['performance_metrics']['accuracy'] == 0.87
    
    def test_retrain_endpoint(self, flask_test_client):
        response = flask_test_client.post('/api/model/retrain')
        
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'message' in data
        assert 'status' in data
        assert 'timestamp' in data
        
        assert 'retraining' in data['message'].lower()
        assert data['status'] == 'queued'
    
    def test_cors_headers(self, flask_test_client):
        response = flask_test_client.options('/api/health')
        
        assert 'Access-Control-Allow-Origin' in response.headers
    
    def test_404_handling(self, flask_test_client):
        response = flask_test_client.get('/api/nonexistent')
        
        assert response.status_code == 404
        
        data = json.loads(response.data)
        assert 'error' in data
        assert 'not found' in data['error'].lower()
    
    def test_content_type_validation(self, flask_test_client):
        response = flask_test_client.post(
            '/api/predict',
            data='{"equipment_id": "TEST"}',
            content_type='text/plain'
        )
        
        assert response.status_code in [400, 500]