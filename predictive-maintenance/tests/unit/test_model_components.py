import pytest
import torch
import torch.nn as nn
import numpy as np
import os
from unittest.mock import Mock, patch, MagicMock

from src.models.lstm_model import PredictiveMaintenanceLSTM, ModelMetrics
from src.models.trainer import ModelTrainer

class TestPredictiveMaintenanceLSTM:
    
    def test_model_initialization(self):
        input_size = 13
        hidden_size = 32
        num_layers = 2
        
        model = PredictiveMaintenanceLSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers
        )
        
        assert isinstance(model.lstm, nn.LSTM)
        assert model.lstm.input_size == input_size
        assert model.lstm.hidden_size == hidden_size
        assert model.lstm.num_layers == num_layers
        
        assert isinstance(model.attention, nn.MultiheadAttention)
        assert isinstance(model.classifier, nn.Sequential)
    
    def test_forward_pass_shape(self, sample_sequences):
        X, y = sample_sequences
        batch_size, sequence_length, input_size = X.shape
        
        model = PredictiveMaintenanceLSTM(input_size=input_size)
        model.eval()
        
        with torch.no_grad():
            output = model(X)
        
        assert output.shape == (batch_size,)
        assert torch.all(output >= 0) and torch.all(output <= 1)
    
    def test_model_training_mode(self, sample_sequences):
        X, y = sample_sequences
        input_size = X.shape[2]
        
        model = PredictiveMaintenanceLSTM(input_size=input_size, dropout=0.5)
        
        model.train()
        output1 = model(X)
        output2 = model(X)
        
        assert not torch.allclose(output1, output2, atol=1e-6)
        
        model.eval()
        with torch.no_grad():
            output3 = model(X)
            output4 = model(X)
        
        assert torch.allclose(output3, output4)
    
    def test_model_with_different_input_sizes(self):
        test_cases = [
            (6, 16, 10),
            (13, 32, 5),
            (20, 64, 1),
        ]
        
        for input_size, hidden_size, batch_size in test_cases:
            model = PredictiveMaintenanceLSTM(
                input_size=input_size,
                hidden_size=hidden_size
            )
            
            X = torch.randn(batch_size, 24, input_size)
            
            output = model(X)
            assert output.shape == (batch_size,)
    
    def test_gradient_flow(self, sample_sequences):
        X, y = sample_sequences
        input_size = X.shape[2]
        
        model = PredictiveMaintenanceLSTM(input_size=input_size)
        criterion = nn.BCELoss()
        
        output = model(X)
        loss = criterion(output, y.float())
        
        loss.backward()
        
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None and torch.any(param.grad != 0):
                has_gradients = True
                break
        
        assert has_gradients


class TestModelMetrics:
    
    def test_metrics_initialization(self):
        metrics = ModelMetrics()
        
        assert len(metrics.predictions) == 0
        assert len(metrics.targets) == 0
        assert len(metrics.losses) == 0
    
    def test_metrics_update(self):
        metrics = ModelMetrics()
        
        predictions = torch.tensor([0.8, 0.3, 0.9, 0.1])
        targets = torch.tensor([1, 0, 1, 0])
        loss = 0.5
        
        metrics.update(predictions, targets, loss)
        
        assert len(metrics.predictions) == 4
        assert len(metrics.targets) == 4
        assert len(metrics.losses) == 1
    
    def test_perfect_predictions_metrics(self):
        metrics = ModelMetrics()
        
        predictions = torch.tensor([0.9, 0.9, 0.1, 0.1])
        targets = torch.tensor([1, 1, 0, 0])
        loss = 0.0
        
        metrics.update(predictions, targets, loss)
        result = metrics.get_metrics()
        
        assert result['accuracy'] == 1.0
        assert result['precision'] == 1.0
        assert result['recall'] == 1.0
        assert result['f1_score'] == 1.0
        assert result['false_alarm_rate'] == 0.0
    
    def test_worst_case_predictions_metrics(self):
        metrics = ModelMetrics()
        
        predictions = torch.tensor([0.1, 0.1, 0.9, 0.9])
        targets = torch.tensor([1, 1, 0, 0])
        loss = 2.0
        
        metrics.update(predictions, targets, loss)
        result = metrics.get_metrics()
        
        assert result['accuracy'] == 0.0
        assert result['false_alarm_rate'] == 1.0
    
    def test_edge_case_no_positives(self):
        metrics = ModelMetrics()
        
        predictions = torch.tensor([0.3, 0.7, 0.2, 0.4])
        targets = torch.tensor([0, 0, 0, 0])
        loss = 1.0
        
        metrics.update(predictions, targets, loss)
        result = metrics.get_metrics()
        
        assert isinstance(result['accuracy'], float)
        assert isinstance(result['precision'], float)
        assert isinstance(result['recall'], float)


class TestModelTrainer:
    
    def test_trainer_initialization(self, sample_sequences):
        X, y = sample_sequences
        input_size = X.shape[2]
        
        model = PredictiveMaintenanceLSTM(input_size=input_size)
        trainer = ModelTrainer(model, device='cpu')
        
        assert trainer.model == model
        assert trainer.device == torch.device('cpu')
        assert len(trainer.train_losses) == 0
        assert len(trainer.val_losses) == 0
    
    @patch('torch.save')
    def test_save_model(self, mock_save, sample_sequences, temp_model_dir):
        X, y = sample_sequences
        input_size = X.shape[2]
        
        model = PredictiveMaintenanceLSTM(input_size=input_size)
        trainer = ModelTrainer(model, device='cpu')
        
        save_path = os.path.join(temp_model_dir, 'test_model.pth')
        trainer.save_model(save_path)
        
        mock_save.assert_called_once()
        
        saved_data = mock_save.call_args[0][0]
        
        assert 'model_state_dict' in saved_data
        assert 'model_config' in saved_data
        assert 'timestamp' in saved_data
    
    @patch('matplotlib.pyplot.savefig')
    @patch('matplotlib.pyplot.close')
    def test_plot_training_curves(self, mock_close, mock_savefig, sample_sequences):
        X, y = sample_sequences
        input_size = X.shape[2]
        
        model = PredictiveMaintenanceLSTM(input_size=input_size)
        trainer = ModelTrainer(model, device='cpu')
        
        trainer.train_losses = [1.0, 0.8, 0.6, 0.4]
        trainer.val_losses = [1.1, 0.9, 0.7, 0.5]
        trainer.train_metrics = [
            {'accuracy': 0.6, 'f1_score': 0.5, 'false_alarm_rate': 0.4, 'recall': 0.6},
            {'accuracy': 0.7, 'f1_score': 0.6, 'false_alarm_rate': 0.3, 'recall': 0.7},
            {'accuracy': 0.8, 'f1_score': 0.7, 'false_alarm_rate': 0.2, 'recall': 0.8},
            {'accuracy': 0.9, 'f1_score': 0.8, 'false_alarm_rate': 0.1, 'recall': 0.9}
        ]
        trainer.val_metrics = trainer.train_metrics.copy()
        
        trainer.plot_training_curves()
        
        mock_savefig.assert_called_once()
        mock_close.assert_called_once()