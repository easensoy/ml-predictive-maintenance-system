import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import json

from .lstm_model import PredictiveMaintenanceLSTM, ModelMetrics

class ModelTrainer:
    
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        metrics = ModelMetrics()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target.float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            metrics.update(output, target, loss.item())
        
        return metrics.get_metrics()
    
    def validate_epoch(self, val_loader, criterion):
        self.model.eval()
        metrics = ModelMetrics()
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target.float())
                metrics.update(output, target, loss.item())
        
        return metrics.get_metrics()
    
    def train(self, X_train, y_train, X_val, y_val, 
              batch_size=32, epochs=100, learning_rate=0.001):
        
        print(f"Training on device: {self.device}")
        
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()])
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 20
        
        print("Starting training...")
        
        for epoch in range(epochs):
            train_metrics = self.train_epoch(train_loader, optimizer, criterion)
            val_metrics = self.validate_epoch(val_loader, criterion)
            scheduler.step(val_metrics['avg_loss'])
            self.train_losses.append(train_metrics['avg_loss'])
            self.val_losses.append(val_metrics['avg_loss'])
            self.train_metrics.append(train_metrics)
            self.val_metrics.append(val_metrics)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"  Train - Loss: {train_metrics['avg_loss']:.4f}, "
                      f"F1: {train_metrics['f1_score']:.4f}, "
                      f"False Alarm Rate: {train_metrics['false_alarm_rate']:.4f}")
                print(f"  Val   - Loss: {val_metrics['avg_loss']:.4f}, "
                      f"F1: {val_metrics['f1_score']:.4f}, "
                      f"False Alarm Rate: {val_metrics['false_alarm_rate']:.4f}")
            
            if val_metrics['avg_loss'] < best_val_loss:
                best_val_loss = val_metrics['avg_loss']
                patience_counter = 0
                self.save_model('models/best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        if os.path.exists('models/best_model.pth'):
            self.load_model('models/best_model.pth')
        self.save_training_history()
        self.plot_training_curves()
        
        print("Training completed!")
        return self.val_metrics[-1]
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_size': self.model.lstm.input_size,
                'hidden_size': self.model.hidden_size,
                'num_layers': self.model.num_layers
            },
            'timestamp': datetime.now().isoformat()
        }, path)
    
    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint
    
    def save_training_history(self):
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs('models', exist_ok=True)
        with open('models/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
    
    def plot_training_curves(self):
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(len(self.train_losses))
        
        ax1.plot(epochs, self.train_losses, label='Training Loss')
        ax1.plot(epochs, self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        train_f1 = [m['f1_score'] for m in self.train_metrics]
        val_f1 = [m['f1_score'] for m in self.val_metrics]
        ax2.plot(epochs, train_f1, label='Training F1')
        ax2.plot(epochs, val_f1, label='Validation F1')
        ax2.set_title('F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        train_far = [m['false_alarm_rate'] for m in self.train_metrics]
        val_far = [m['false_alarm_rate'] for m in self.val_metrics]
        ax3.plot(epochs, train_far, label='Training FAR')
        ax3.plot(epochs, val_far, label='Validation FAR')
        ax3.set_title('False Alarm Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('False Alarm Rate')
        ax3.legend()
        ax3.grid(True)
        
        train_recall = [m['recall'] for m in self.train_metrics]
        val_recall = [m['recall'] for m in self.val_metrics]
        ax4.plot(epochs, train_recall, label='Training Recall')
        ax4.plot(epochs, val_recall, label='Validation Recall')
        ax4.set_title('Recall (Failure Detection Rate)')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Recall')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
