#!/usr/bin/env python3
"""
PyTorch-based Training Script for Predictive Maintenance
Advanced LSTM + Attention model for equipment failure prediction
"""

import os
import sys
import json
import pickle
import argparse
import matplotlib.pyplot as plt
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add src to path
sys.path.append('src')

# Import modules
from data.generate_data import generate_sensor_data
from src.models.lstm_model import PredictiveMaintenanceLSTM, MetricsTracker, EarlyStopping, count_parameters

def create_sequences(data, sequence_length=24):
    """Create sequences for time series prediction"""
    sequences = []
    targets = []
    
    # Group by equipment
    for equipment_id in data['equipment_id'].unique():
        equipment_data = data[data['equipment_id'] == equipment_id].copy()
        equipment_data = equipment_data.sort_values('timestamp')
        
        # Create sequences
        for i in range(len(equipment_data) - sequence_length + 1):
            seq_data = equipment_data.iloc[i:i+sequence_length]
            
            # Extract features (excluding timestamp, equipment_id, target)
            feature_cols = [col for col in equipment_data.columns 
                          if col not in ['timestamp', 'equipment_id', 'failure_within_24h']]
            
            sequence = seq_data[feature_cols].values
            target = seq_data['failure_within_24h'].iloc[-1]  # Use last value as target
            
            sequences.append(sequence)
            targets.append(target)
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

def preprocess_data(data):
    """Preprocess data for training"""
    print("Loading and preprocessing data...")
    
    # Feature engineering
    data['hour'] = pd.to_datetime(data['timestamp']).dt.hour
    data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.dayofweek
    data['day_of_month'] = pd.to_datetime(data['timestamp']).dt.day
    
    # Equipment encoding
    equipment_encoder = LabelEncoder()
    data['equipment_encoded'] = equipment_encoder.fit_transform(data['equipment_id'])
    
    # Create one-hot encoding for equipment (simplified)
    equipment_dummies = pd.get_dummies(data['equipment_encoded'], prefix='equipment')
    data = pd.concat([data, equipment_dummies], axis=1)
    
    # Ensure we have exactly 19 features as expected by model
    feature_cols = [
        'vibration_rms', 'temperature_bearing', 'pressure_oil', 
        'rpm', 'oil_quality_index', 'power_consumption',
        'hour', 'day_of_week', 'day_of_month', 'equipment_encoded'
    ]
    
    # Add padding features to reach 19 dimensions
    for i in range(19 - len(feature_cols)):
        data[f'feature_{i}'] = np.random.normal(0, 0.1, len(data))
        feature_cols.append(f'feature_{i}')
    
    # Keep only necessary columns
    data = data[feature_cols + ['equipment_id', 'timestamp', 'failure_within_24h']]
    
    return data, equipment_encoder

def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train model for one epoch"""
    model.train()
    metrics = MetricsTracker()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        metrics.update(output, target, loss.item())
    
    return metrics.compute_metrics()

def evaluate_model(model, data_loader, criterion, device):
    """Evaluate model performance"""
    model.eval()
    metrics = MetricsTracker()
    
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            metrics.update(output, target, loss.item())
    
    return metrics.compute_metrics()

def plot_training_curves(history, save_path='models/training_curves.png'):
    """Plot training curves"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # F1 Score
    ax2.plot(epochs, history['train_f1'], 'b-', label='Training F1')
    ax2.plot(epochs, history['val_f1'], 'r-', label='Validation F1')
    ax2.set_title('F1 Score')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.legend()
    ax2.grid(True)
    
    # False Alarm Rate
    ax3.plot(epochs, history['train_far'], 'b-', label='Training FAR')
    ax3.plot(epochs, history['val_far'], 'r-', label='Validation FAR')
    ax3.set_title('False Alarm Rate')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('False Alarm Rate')
    ax3.legend()
    ax3.grid(True)
    
    # Recall (Failure Detection Rate)
    ax4.plot(epochs, history['train_recall'], 'b-', label='Training Recall')
    ax4.plot(epochs, history['val_recall'], 'r-', label='Validation Recall')
    ax4.set_title('Recall (Failure Detection Rate)')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Recall')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")

def main():
    parser = argparse.ArgumentParser(description='Train PyTorch Predictive Maintenance Model')
    parser.add_argument('--generate-data', action='store_true', help='Generate synthetic data')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--num-layers', type=int, default=2, help='Number of LSTM layers')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PYTORCH PREDICTIVE MAINTENANCE MODEL TRAINING")
    print("=" * 60)
    print(f"Training started at: {datetime.now()}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 1. DATA PREPARATION
    print("\n1. DATA PREPARATION")
    print("-" * 30)
    
    if args.generate_data:
        print("Generating sensor data...")
        generate_sensor_data(num_samples=10000, num_equipments=50)
    
    # Load data
    if os.path.exists('data/raw/sensor_data.csv'):
        data = pd.read_csv('data/raw/sensor_data.csv')
        print(f"Loaded {len(data)} sensor readings")
    else:
        print("No data found. Please generate data first with --generate-data")
        return
    
    # 2. DATA PREPROCESSING
    print("\n2. DATA PREPROCESSING")
    print("-" * 30)
    
    data, equipment_encoder = preprocess_data(data)
    
    print("Creating sequences...")
    X, y = create_sequences(data, sequence_length=24)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print("Scaling features...")
    # Flatten for scaling, then reshape
    n_samples, n_timesteps, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples, n_timesteps, n_features)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape[0], n_timesteps, n_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape[0], n_timesteps, n_features)
    
    print("Data preparation complete:")
    print(f"  Training samples: {len(X_train_scaled):,}")
    print(f"  Validation samples: {len(X_val_scaled):,}")
    print(f"  Test samples: {len(X_test_scaled):,}")
    print(f"  Feature dimensions: {X_train_scaled.shape}")
    print(f"  Positive samples (failures): {y_train.sum()}/{len(y_train)} ({y_train.mean():.2%})")
    
    # 3. MODEL INITIALIZATION
    print("\n3. MODEL INITIALIZATION")
    print("-" * 30)
    
    model = PredictiveMaintenanceLSTM(
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    print(f"Model architecture: LSTM + Attention")
    print(f"Input size: {n_features}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Sequence length: 24")
    print(f"Model parameters: {count_parameters(model):,}")
    
    # 4. TRAINING SETUP
    print("\n4. TRAINING SETUP")
    print("-" * 30)
    
    # Create data loaders
    train_dataset = TensorDataset(torch.FloatTensor(X_train_scaled), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val_scaled), torch.FloatTensor(y_val))
    test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test))
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Loss function with class weights
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Optimizer with weight decay
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    print(f"Training on device: {device}")
    print(f"Optimizer: Adam (lr={args.learning_rate})")
    print(f"Loss function: BCEWithLogitsLoss (pos_weight={pos_weight.item():.2f})")
    
    # 5. TRAINING LOOP
    print("\n5. MODEL TRAINING")
    print("-" * 30)
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_f1': [], 'val_f1': [],
        'train_far': [], 'val_far': [],
        'train_recall': [], 'val_recall': []
    }
    
    print("Starting training...")
    best_val_f1 = 0
    
    for epoch in range(args.epochs):
        # Training
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_metrics['avg_loss'])
        
        # Store metrics
        history['train_loss'].append(train_metrics['avg_loss'])
        history['val_loss'].append(val_metrics['avg_loss'])
        history['train_f1'].append(train_metrics['f1_score'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['train_far'].append(train_metrics['false_alarm_rate'])
        history['val_far'].append(val_metrics['false_alarm_rate'])
        history['train_recall'].append(train_metrics['recall'])
        history['val_recall'].append(val_metrics['recall'])
        
        # Print progress
        print(f"Epoch {epoch+1:2d}/{args.epochs}")
        print(f"  Train - Loss: {train_metrics['avg_loss']:.4f}, F1: {train_metrics['f1_score']:.4f}, False Alarm Rate: {train_metrics['false_alarm_rate']:.4f}")
        print(f"  Val   - Loss: {val_metrics['avg_loss']:.4f}, F1: {val_metrics['f1_score']:.4f}, False Alarm Rate: {val_metrics['false_alarm_rate']:.4f}")
        
        # Save best model
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_val_f1,
                'model_params': {
                    'input_size': n_features,
                    'hidden_size': args.hidden_size,
                    'num_layers': args.num_layers
                }
            }, 'models/best_model.pth')
        
        # Early stopping check
        if early_stopping(val_metrics['avg_loss'], model):
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    print("Training completed!")
    
    # 6. FINAL EVALUATION
    print("\n6. FINAL EVALUATION")
    print("-" * 30)
    
    # Load best model
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    print("Final Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 7. SAVE ARTIFACTS
    print("\n7. SAVING ARTIFACTS")
    print("-" * 30)
    
    # Save preprocessors
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    with open('models/equipment_encoder.pkl', 'wb') as f:
        pickle.dump(equipment_encoder, f)
    
    # Save training history
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save model metadata
    metadata = {
        'model_type': 'PyTorch LSTM + Attention',
        'framework': 'PyTorch',
        'input_features': int(n_features),
        'sequence_length': 24,
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'prediction_horizon': '24 hours',
        'training_samples': int(len(X_train_scaled)),
        'validation_samples': int(len(X_val_scaled)),
        'test_samples': int(len(X_test_scaled)),
        'epochs_trained': epoch + 1,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'training_time': datetime.now().isoformat(),
        **test_metrics
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Plot training curves
    plot_training_curves(history)
    
    print("Model artifacts saved:")
    print("  - models/best_model.pth")
    print("  - models/scaler.pkl") 
    print("  - models/equipment_encoder.pkl")
    print("  - models/training_history.json")
    print("  - models/model_metadata.json")
    print("  - models/training_curves.png")
    
    print("\n" + "=" * 60)
    print("PYTORCH TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("Next steps:")
    print("1. Test locally: python run_api.py")
    print("2. Deploy to AWS: docker build with PyTorch requirements")
    print("3. Monitor with budget protection enabled")

if __name__ == "__main__":
    main()