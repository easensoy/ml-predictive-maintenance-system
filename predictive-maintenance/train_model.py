#!/usr/bin/env python3

import os
import sys
import json
import pickle
import argparse
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

sys.path.append('src')

from src.models.lstm_model import PredictiveMaintenanceLSTM, MetricsTracker, EarlyStopping, count_parameters
from src.live_sensor_collector import LiveSensorDataCollector

def collect_live_data_for_training(hours=72, interval_minutes=30):
    collector = LiveSensorDataCollector()
    all_data = []
    total_collections = int(hours * 60 / interval_minutes)
    
    print(f"Collecting live data for {hours} hours ({total_collections} collections)...")
    print("This may take a while. Consider using existing data for faster training.")
    
    for i in range(total_collections):
        try:
            live_data = collector.collect_live_data()
            for equipment in live_data['equipment_data']:
                equipment['timestamp'] = live_data['collection_timestamp']
                all_data.append(equipment)
            
            print(f"Collection {i+1}/{total_collections} complete ({len(all_data)} total records)")
            
            if i < total_collections - 1:
                time.sleep(interval_minutes * 60)
        except Exception as e:
            print(f"Error collecting data at iteration {i+1}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No live data collected. Check sensor connections.")
    
    return pd.DataFrame(all_data)

def load_data(data_path, use_live=False, live_hours=72):
    if use_live:
        return collect_live_data_for_training(hours=live_hours)
    elif os.path.exists(data_path):
        return pd.read_csv(data_path)
    else:
        raise FileNotFoundError(f"Data file not found: {data_path}")

def create_sequences(data, seq_len=24):
    sequences, targets = [], []
    feature_cols = [col for col in data.columns if col not in ['timestamp', 'equipment_id', 'failure_within_24h']]
    
    for equipment_id in data['equipment_id'].unique():
        eq_data = data[data['equipment_id'] == equipment_id].sort_values('timestamp')
        for i in range(len(eq_data) - seq_len + 1):
            sequences.append(eq_data[feature_cols].iloc[i:i+seq_len].values)
            targets.append(eq_data['failure_within_24h'].iloc[i+seq_len-1])
    
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

def preprocess_data(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['day_of_week'] = data['timestamp'].dt.dayofweek
    
    encoder = LabelEncoder()
    data['equipment_encoded'] = encoder.fit_transform(data['equipment_id'])
    
    feature_cols = []
    required_cols = ['vibration_rms', 'temperature_bearing', 'pressure_oil', 
                    'rpm', 'power_consumption', 'hour', 'day_of_week', 'equipment_encoded']
    
    for col in required_cols:
        if col in data.columns:
            feature_cols.append(col)
    
    if 'oil_quality_index' in data.columns:
        feature_cols.append('oil_quality_index')
    elif 'oil_quality' in data.columns:
        data['oil_quality_index'] = data['oil_quality'] * 100
        feature_cols.append('oil_quality_index')
    
    if 'failure_within_24h' not in data.columns:
        data['failure_within_24h'] = (data['failure_probability'] > 0.5).astype(int)
    
    return data[feature_cols + ['equipment_id', 'timestamp', 'failure_within_24h']], encoder

def train_epoch(model, loader, optimizer, criterion, device):
            loss = criterion(output, target)
            metrics.update(output, target, loss.item())
    
    return metrics.compute_metrics()

def save_artifacts(model, scaler, encoder, history, test_metrics, args, n_features):
    os.makedirs('models', exist_ok=True)
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('models/equipment_encoder.pkl', 'wb') as f:
        pickle.dump(encoder, f)
    with open('models/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    metadata = {
        'model_type': 'PyTorch LSTM + Attention',
        'input_features': int(n_features),
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'training_time': datetime.now().isoformat(),
        **test_metrics
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size):
    datasets = [
        TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
        for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    ]
    return [DataLoader(ds, batch_size=batch_size, shuffle=(i==0)) for i, ds in enumerate(datasets)]

def scale_data(X_train, X_val, X_test):
    scaler = StandardScaler()
    n_samples, n_timesteps, n_features = X_train.shape
    
    X_train_flat = X_train.reshape(-1, n_features)
    X_val_flat = X_val.reshape(-1, n_features)
    X_test_flat = X_test.reshape(-1, n_features)
    
    X_train_scaled = scaler.fit_transform(X_train_flat).reshape(n_samples, n_timesteps, n_features)
    X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape[0], n_timesteps, n_features)
    X_test_scaled = scaler.transform(X_test_flat).reshape(X_test.shape[0], n_timesteps, n_features)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=str, default='data/raw/sensor_data.csv')
    parser.add_argument('--use-live', action='store_true')
    parser.add_argument('--live-hours', type=int, default=72)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--hidden-size', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for dir_name in ['models', 'data/raw', 'data/processed', 'logs']:
        os.makedirs(dir_name, exist_ok=True)
    
    print(f"Loading data from {'live sensors' if args.use_live else args.data_path}...")
    data = load_data(args.data_path, args.use_live)
    
    if args.use_live:
        data.to_csv('data/processed/live_training_data.csv', index=False)
        print(f"Saved {len(data)} live data points to data/processed/live_training_data.csv")
    data, encoder = preprocess_data(data)
    X, y = create_sequences(data, seq_len=24)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_data(X_train, X_val, X_test)
    train_loader, val_loader, test_loader = create_data_loaders(
        X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, args.batch_size
    )
    
    n_features = X_train_scaled.shape[2]
    model = PredictiveMaintenanceLSTM(
        input_size=n_features,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    ).to(device)
    
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    early_stopping = EarlyStopping(patience=7, min_delta=0.001)
    
    history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
    best_val_f1 = 0
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        scheduler.step(val_metrics['avg_loss'])
        
        for key in history:
            metric_key = key.replace('train_', '').replace('val_', '')
            if key.startswith('train_'):
                history[key].append(train_metrics[metric_key])
            else:
                history[key].append(val_metrics[metric_key])
        
        if val_metrics['f1_score'] > best_val_f1:
            best_val_f1 = val_metrics['f1_score']
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_params': {'input_size': n_features, 'hidden_size': args.hidden_size, 'num_layers': args.num_layers}
            }, 'models/best_model.pth')
        
        if early_stopping(val_metrics['avg_loss'], model):
            break
    
    checkpoint = torch.load('models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    save_artifacts(model, scaler, encoder, history, test_metrics, args, n_features)
    
    data_source = "live sensors" if args.use_live else "CSV file"
    print(f"Training completed using {data_source}. Test F1: {test_metrics['f1_score']:.4f}")

if __name__ == "__main__":
    main()