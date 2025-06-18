import pytest
import os
import tempfile
import torch
import pandas as pd
from unittest.mock import patch

from data.generate_data import generate_sensor_data
from src.utils.data_preprocessing import DataPreprocessor
from src.models.lstm_model import PredictiveMaintenanceLSTM
from src.models.trainer import ModelTrainer

class TestTrainingPipeline:
    
    @pytest.mark.slow
    def test_complete_training_pipeline(self, temp_model_dir):
        df = generate_sensor_data(num_samples=1000, num_equipments=10)
        
        data_file = os.path.join(temp_model_dir, 'test_sensor_data.csv')
        df.to_csv(data_file, index=False)
        
        assert os.path.exists(data_file)
        assert len(df) > 0
        assert 'failure_within_24h' in df.columns
        
        original_cwd = os.getcwd()
        os.chdir(temp_model_dir)
        
        try:
            preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test = preprocessor.prepare_data(data_file)
            
            assert len(X_train) > 0
            assert len(X_test) > 0
            assert X_train.shape[1] == 24
            assert X_train.shape[2] > 6
            
            assert os.path.exists('models/scaler.pkl')
            assert os.path.exists('models/equipment_encoder.pkl')
            
            input_size = X_train.shape[2]
            model = PredictiveMaintenanceLSTM(
                input_size=input_size,
                hidden_size=32,
                num_layers=1,
                dropout=0.1
            )
            
            trainer = ModelTrainer(model, device='cpu')
            
            final_metrics = trainer.train(
                X_train, y_train, X_test, y_test,
                batch_size=16,
                epochs=5,
                learning_rate=0.01
            )
            
            assert 'accuracy' in final_metrics
            assert 'f1_score' in final_metrics
            assert final_metrics['accuracy'] >= 0
            assert final_metrics['f1_score'] >= 0
            
            assert os.path.exists('models/best_model.pth')
            assert os.path.exists('models/training_history.json')
            
            model.eval()
            with torch.no_grad():
                test_batch = torch.FloatTensor(X_test[:5])
                predictions = model(test_batch)
                
                assert predictions.shape == (5,)
                assert torch.all(predictions >= 0)
                assert torch.all(predictions <= 1)
            
        finally:
            os.chdir(original_cwd)
    
    def test_data_generation_consistency(self, temp_model_dir):
        df1 = generate_sensor_data(num_samples=500, num_equipments=5)
        df2 = generate_sensor_data(num_samples=500, num_equipments=5)
        
        assert len(df1) == len(df2)
        assert list(df1.columns) == list(df2.columns)
        
        for df in [df1, df2]:
            assert df['vibration_rms'].notna().all()
            assert df['temperature_bearing'].notna().all()
            assert df['failure_within_24h'].notna().all()
            
            assert df['vibration_rms'].min() >= 0
            assert df['temperature_bearing'].between(20, 120).all()
            assert df['oil_quality_index'].between(0, 100).all()
            assert df['failure_within_24h'].isin([0, 1]).all()
            
            failure_rate = df['failure_within_24h'].mean()
            assert 0.05 <= failure_rate <= 0.20
    
    def test_preprocessing_pipeline_integration(self, sample_sensor_data, temp_model_dir):
        data_file = os.path.join(temp_model_dir, 'integration_test_data.csv')
        sample_sensor_data.to_csv(data_file, index=False)
        
        original_cwd = os.getcwd()
        os.chdir(temp_model_dir)
        
        try:
            preprocessor = DataPreprocessor()
            X_train, X_test, y_train, y_test = preprocessor.prepare_data(data_file)
            
            X_train_tensor = torch.FloatTensor(X_train)
            y_train_tensor = torch.LongTensor(y_train)
            
            input_size = X_train.shape[2]
            model = PredictiveMaintenanceLSTM(input_size=input_size, hidden_size=16)
            
            with torch.no_grad():
                output = model(X_train_tensor[:5])
                assert output.shape == (5,)
                assert torch.all(output >= 0) and torch.all(output <= 1)
            
        finally:
            os.chdir(original_cwd)
    
    @pytest.mark.slow
    def test_model_persistence_and_loading(self, temp_model_dir):
        batch_size, seq_len, input_size = 50, 24, 13
        X_train = torch.randn(batch_size, seq_len, input_size)
        y_train = torch.randint(0, 2, (batch_size,))
        X_test = torch.randn(20, seq_len, input_size)
        y_test = torch.randint(0, 2, (20,))
        
        original_cwd = os.getcwd()
        os.chdir(temp_model_dir)
        
        try:
            model = PredictiveMaintenanceLSTM(input_size=input_size, hidden_size=16)
            trainer = ModelTrainer(model, device='cpu')
            
            trainer.train(
                X_train.numpy(), y_train.numpy(), 
                X_test.numpy(), y_test.numpy(),
                epochs=3, batch_size=10
            )
            
            model.eval()
            with torch.no_grad():
                original_prediction = model(X_test[:1])
            
            checkpoint = torch.load('models/best_model.pth', map_location='cpu')
            
            loaded_model = PredictiveMaintenanceLSTM(
                input_size=checkpoint['model_config']['input_size'],
                hidden_size=checkpoint['model_config']['hidden_size'],
                num_layers=checkpoint['model_config']['num_layers']
            )
            loaded_model.load_state_dict(checkpoint['model_state_dict'])
            loaded_model.eval()
            
            with torch.no_grad():
                loaded_prediction = loaded_model(X_test[:1])
            
            assert torch.allclose(original_prediction, loaded_prediction, atol=1e-6)
            
        finally:
            os.chdir(original_cwd)