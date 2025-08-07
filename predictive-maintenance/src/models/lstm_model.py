import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class AttentionLayer(nn.Module):
    """Attention mechanism for LSTM"""
    
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.attention_weights = nn.Linear(hidden_size, 1)
        
    def forward(self, lstm_output):
        # lstm_output shape: (batch_size, seq_len, hidden_size)
        attention_scores = self.attention_weights(lstm_output)  # (batch_size, seq_len, 1)
        attention_scores = torch.softmax(attention_scores, dim=1)
        
        # Apply attention weights
        weighted_output = lstm_output * attention_scores  # (batch_size, seq_len, hidden_size)
        attended_output = torch.sum(weighted_output, dim=1)  # (batch_size, hidden_size)
        
        return attended_output

class PredictiveMaintenanceLSTM(nn.Module):
    """LSTM + Attention model for predictive maintenance"""
    
    def __init__(self, input_size=19, hidden_size=64, num_layers=2, dropout=0.2):
        super(PredictiveMaintenanceLSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = AttentionLayer(hidden_size)
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        batch_size = x.size(0)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # Apply attention
        attended_output = self.attention(lstm_out)
        
        # Classification
        output = self.classifier(attended_output)
        
        return output.squeeze(-1)  # Remove last dimension for binary classification

class MetricsTracker:
    """Track training metrics"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.predictions = []
        self.targets = []
        self.losses = []
    
    def update(self, output, target, loss):
        # Convert to binary predictions
        predictions = torch.sigmoid(output) > 0.5
        
        # Store metrics
        self.predictions.extend(predictions.detach().cpu().numpy())
        self.targets.extend(target.detach().cpu().numpy())
        self.losses.append(loss)
    
    def compute_metrics(self):
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        if len(predictions) == 0:
            return {'accuracy': 0, 'precision': 0, 'recall': 0, 'f1_score': 0, 'avg_loss': 0}
        
        # Calculate metrics
        accuracy = accuracy_score(targets, predictions)
        precision = precision_score(targets, predictions, zero_division=0)
        recall = recall_score(targets, predictions, zero_division=0)
        f1 = f1_score(targets, predictions, zero_division=0)
        
        # False alarm rate (false positives / total negatives)
        false_alarm_rate = 0
        if (targets == 0).sum() > 0:
            false_alarm_rate = ((predictions == 1) & (targets == 0)).sum() / (targets == 0).sum()
        
        avg_loss = np.mean(self.losses) if self.losses else 0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_alarm_rate': false_alarm_rate,
            'avg_loss': avg_loss
        }

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)