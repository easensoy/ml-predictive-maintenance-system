import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from unittest.mock import patch, Mock

from src.utils.data_preprocessing import DataPreprocessor

class TestDataPreprocessor:
    
    def test_initialization(self):
        preprocessor = DataPreprocessor()
        
        assert preprocessor.sequence_length == 24
        assert preprocessor.scaler is not None
        assert preprocessor.equipment_encoder is not None
    
    def test_load_and_preprocess_basic(self, sample_sensor_data, temp_model_dir):
        temp_file = os.path.join(temp_model_dir, 'test_data.csv')
        sample_sensor_data.to_csv(temp_file, index=False)
        
        preprocessor = DataPreprocessor()
        
        result_df = preprocessor.load_and_preprocess(temp_file)
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) > 0
        assert 'equipment_encoded' in result_df.columns
        
        rolling_columns = [col for col in result_df.columns if 'rolling' in col]
        assert len(rolling_columns) > 0
        
        assert result_df['equipment_encoded'].dtype in [np.int32, np.int64]
    
    def test_create_sequences_shape(self, sample_sensor_data, temp_model_dir):
        temp_file = os.path.join(temp_model_dir, 'test_data.csv')
        sample_sensor_data.to_csv(temp_file, index=False)
        
        preprocessor = DataPreprocessor()
        df = preprocessor.load_and_preprocess(temp_file)
        
        X, y, equipment_ids = preprocessor.create_sequences(df)
        
        assert len(X.shape) == 3
        assert X.shape[1] == 24
        assert X.shape[2] > 6
        
        assert len(y) == len(X)
        assert len(equipment_ids) == len(X)
        assert all(isinstance(eq_id, str) for eq_id in equipment_ids)
    
    def test_scale_features_consistency(self, sample_sequences):
        X_train, y_train = sample_sequences
        X_test = X_train.clone()
        
        preprocessor = DataPreprocessor()
        
        X_train_scaled, X_test_scaled = preprocessor.scale_features(
            X_train.numpy(), X_test.numpy()
        )
        
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape
        
        flattened_train = X_train_scaled.reshape(-1, X_train_scaled.shape[-1])
        means = np.mean(flattened_train, axis=0)
        stds = np.std(flattened_train, axis=0)
        
        assert np.allclose(means, 0, atol=0.1)
        assert np.allclose(stds, 1, atol=0.2)
    
    def test_prepare_data_end_to_end(self, sample_sensor_data, temp_model_dir):
        temp_file = os.path.join(temp_model_dir, 'test_data.csv')
        sample_sensor_data.to_csv(temp_file, index=False)
        
        preprocessor = DataPreprocessor()
        
        X_train, X_test, y_train, y_test = preprocessor.prepare_data(temp_file)
        
        assert isinstance(X_train, np.ndarray)
        assert isinstance(X_test, np.ndarray)
        assert isinstance(y_train, np.ndarray)
        assert isinstance(y_test, np.ndarray)
        
        total_samples = len(X_train) + len(X_test)
        assert total_samples > 0
        assert len(X_test) / total_samples == pytest.approx(0.2, abs=0.1)
        
        train_positive_rate = np.mean(y_train)
        test_positive_rate = np.mean(y_test)
        assert abs(train_positive_rate - test_positive_rate) < 0.3
    
    def test_edge_cases(self, temp_model_dir):
        minimal_data = pd.DataFrame({
            'timestamp': pd.date_range('2023-01-01', periods=30, freq='H'),
            'equipment_id': ['EQ_001'] * 30,
            'vibration_rms': [1.0] * 30,
            'temperature_bearing': [70.0] * 30,
            'pressure_oil': [20.0] * 30,
            'rpm': [1800] * 30,
            'oil_quality_index': [85] * 30,
            'power_consumption': [50.0] * 30,
            'failure_within_24h': [0] * 30
        })
        
        temp_file = os.path.join(temp_model_dir, 'minimal_data.csv')
        minimal_data.to_csv(temp_file, index=False)
        
        preprocessor = DataPreprocessor()
        
        result_df = preprocessor.load_and_preprocess(temp_file)
        assert len(result_df) > 0
        
        X, y, equipment_ids = preprocessor.create_sequences(result_df)
        assert len(X) >= 0
    
    @patch('joblib.dump')
    def test_preprocessor_saving(self, mock_dump, sample_sensor_data, temp_model_dir):
        temp_file = os.path.join(temp_model_dir, 'test_data.csv')
        sample_sensor_data.to_csv(temp_file, index=False)
        
        preprocessor = DataPreprocessor()
        preprocessor.prepare_data(temp_file)
        
        assert mock_dump.call_count == 2
        
        calls = mock_dump.call_args_list
        saved_objects = [call[0][0] for call in calls]
        
        assert preprocessor.scaler in saved_objects
        assert preprocessor.equipment_encoder in saved_objects