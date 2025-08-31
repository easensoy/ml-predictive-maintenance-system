import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import yaml
import os

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        scores = torch.softmax(self.attention_weights(lstm_output), dim=1)
        return torch.sum(lstm_output * scores, dim=1)

class PredictiveMaintenanceLSTM(nn.Module):
    def __init__(self, config_path=None, input_size=None, hidden_size=None, num_layers=None, dropout=None):
        if config_path:
            config = self._load_config(config_path)
            input_size = input_size or config['model']['input_size']
            hidden_size = hidden_size or config['model']['hidden_size']
            num_layers = num_layers or config['model']['num_layers']
            dropout = dropout or config['model']['dropout']
            self.config = config
        else:
            input_size = input_size or 19
            hidden_size = hidden_size or 64
            num_layers = num_layers or 2
            dropout = dropout or 0.2
            self.config = None
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        min_layers = self.config['training']['dropout_min_layers'] if self.config else 1
        batch_first = self.config['training']['batch_first'] if self.config else True
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout if num_layers > min_layers else 0, 
                           batch_first=batch_first)
        self.attention = AttentionLayer(hidden_size)
        
        hidden_1 = self.config['classifier']['hidden_layer_1'] if self.config else 32
        hidden_2 = self.config['classifier']['hidden_layer_2'] if self.config else 16
        output_size = self.config['classifier']['output_layer'] if self.config else 1
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
            nn.Linear(hidden_2, output_size)
        )
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                fill_value = self.config['training']['weight_init_bias_fill'] if self.config else 0
                param.data.fill_(fill_value)
    
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        attended = self.attention(lstm_out)
        return self.classifier(attended).squeeze(-1)
    
    def _load_config(self, config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

class MetricsTracker:
    def __init__(self, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.threshold = config['metrics']['threshold']
            self.zero_division_default = config['metrics']['zero_division_default']
        else:
            self.threshold = 0.5
            self.zero_division_default = 0
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, output, target, loss):
        pred = (torch.sigmoid(output) > self.threshold).cpu().numpy()
        self.predictions.extend(pred)
        self.targets.extend(target.cpu().numpy())
        self.losses.append(loss)
    
    def compute_metrics(self):
        if not self.predictions:
            return {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1_score', 'false_alarm_rate', 'avg_loss']}
        
        preds, targets = np.array(self.predictions), np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, zero_division=self.zero_division_default),
            'recall': recall_score(targets, preds, zero_division=self.zero_division_default),
            'f1_score': f1_score(targets, preds, zero_division=self.zero_division_default),
            'avg_loss': np.mean(self.losses)
        }
        
        neg_count = (targets == 0).sum()
        false_negative_default = self.zero_division_default
        metrics['false_alarm_rate'] = ((preds == 1) & (targets == 0)).sum() / neg_count if neg_count > 0 else false_negative_default
        
        return metrics
    
    def reset(self):
        self.predictions.clear()
        self.targets.clear()
        self.losses.clear()

class EarlyStopping:
    def __init__(self, config_path=None, patience=None, min_delta=None):
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.patience = patience or config['early_stopping']['patience']
            self.min_delta = min_delta or config['early_stopping']['min_delta']
        else:
            self.patience = patience or 7
            self.min_delta = min_delta or 0.001
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        
    def __call__(self, val_loss, model=None):
        if self.best_loss is None or self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)