import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention_weights = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        scores = torch.softmax(self.attention_weights(lstm_output), dim=1)
        return torch.sum(lstm_output * scores, dim=1)

class PredictiveMaintenanceLSTM(nn.Module):
    def __init__(self, input_size=19, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           dropout=dropout if num_layers > 1 else 0, 
                           batch_first=True)
        self.attention = AttentionLayer(hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        attended = self.attention(lstm_out)
        return self.classifier(attended).squeeze(-1)

class MetricsTracker:
    def __init__(self):
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, output, target, loss):
        pred = (torch.sigmoid(output) > 0.5).cpu().numpy()
        self.predictions.extend(pred)
        self.targets.extend(target.cpu().numpy())
        self.losses.append(loss)
    
    def compute_metrics(self):
        if not self.predictions:
            return {k: 0 for k in ['accuracy', 'precision', 'recall', 'f1_score', 'false_alarm_rate', 'avg_loss']}
        
        preds, targets = np.array(self.predictions), np.array(self.targets)
        
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, zero_division=0),
            'recall': recall_score(targets, preds, zero_division=0),
            'f1_score': f1_score(targets, preds, zero_division=0),
            'avg_loss': np.mean(self.losses)
        }
        
        neg_count = (targets == 0).sum()
        metrics['false_alarm_rate'] = ((preds == 1) & (targets == 0)).sum() / neg_count if neg_count > 0 else 0
        
        return metrics
    
    def reset(self):
        self.predictions.clear()
        self.targets.clear()
        self.losses.clear()

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
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