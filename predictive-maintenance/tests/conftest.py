import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
import sys
from unittest.mock import Mock
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

@pytest.fixture
def sample_sensor_data():
    np.random.seed(42)
    n_samples = 100
    data = {
        'timestamp': [datetime.now() - timedelta(hours=i) for i in range(n_samples)],
        'equipment_id': ['TEST_EQ_001'] * n_samples,
        'vibration_rms': np.random.normal(1.5, 0.3, n_samples),
        'temperature_bearing': np.random.normal(75, 5, n_samples),
        'pressure_oil': np.random.normal(20, 2, n_samples),
        'rpm': np.random.normal(1800, 50, n_samples),
        'oil_quality_index': np.random.normal(85, 10, n_samples),
        'power_consumption': np.random.normal(50, 5, n_samples),
        'failure_within_24h': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    df = pd.DataFrame(data)
    df['vibration_rms'] = np.clip(df['vibration_rms'], 0.1, 10)
    df['temperature_bearing'] = np.clip(df['temperature_bearing'], 20, 120)
    df['pressure_oil'] = np.clip(df['pressure_oil'], 5, 30)
    df['rpm'] = np.clip(df['rpm'], 1000, 2500)
    df['oil_quality_index'] = np.clip(df['oil_quality_index'], 0, 100)
    df['power_consumption'] = np.clip(df['power_consumption'], 10, 100)
    return df

@pytest.fixture
def sample_sequences():
    batch_size = 10
    sequence_length = 24
    num_features = 13
    X = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
    y = np.random.choice([0, 1], batch_size)
    return torch.FloatTensor(X), torch.LongTensor(y)

@pytest.fixture
def temp_model_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = os.path.join(temp_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        data_dir = os.path.join(temp_dir, 'data', 'raw')
        os.makedirs(data_dir, exist_ok=True)
        yield temp_dir

@pytest.fixture
def mock_trained_model():
    mock_model = Mock()
    mock_model.eval.return_value = None
    mock_model.return_value = torch.tensor([0.7])
    return mock_model

@pytest.fixture
def sample_api_data():
    return {
        'equipment_id': 'TEST_EQ_001',
        'sensor_data': [
            {
                'vibration_rms': 1.5,
                'temperature_bearing': 75.0,
                'pressure_oil': 20.0,
                'rpm': 1800,
                'oil_quality_index': 85,
                'power_consumption': 50.0
            }
        ] * 24
    }

@pytest.fixture
def flask_test_client():
    from src.api.app import app
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
