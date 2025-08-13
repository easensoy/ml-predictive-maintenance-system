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
from src.models.lstm_model import PredictiveMaintenanceLSTM, EarlyStopping
from src.live_sensor_collector import LiveSensorDataCollector

class Trainer:
    def __init__(self, data_path='data/raw/sensor_data.csv', use_live=False, live_hours=72, epochs=20, batch_size=32, learning_rate=0.001, hidden_size=64, num_layers=2, device=None):
        self.data_path = data_path
        self.use_live = use_live
        self.live_hours = live_hours
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = None
        self.encoder = None
        self.history = None
        self.n_features = None
        self.test_metrics = None

    @staticmethod
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

    @staticmethod
    def load_data(data_path, use_live=False, live_hours=72):
        if use_live:
            return Trainer.collect_live_data_for_training(hours=live_hours)
        elif os.path.exists(data_path):
            return pd.read_csv(data_path)
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")

    @staticmethod
    def create_sequences(data, seq_len=24):
        sequences, targets = [], []
        feature_cols = [col for col in data.columns if col not in ['timestamp', 'equipment_id', 'failure_within_24h']]
        for equipment_id in data['equipment_id'].unique():
            eq_data = data[data['equipment_id'] == equipment_id].sort_values('timestamp')
            for i in range(len(eq_data) - seq_len + 1):
                sequences.append(eq_data[feature_cols].iloc[i:i+seq_len].values)
                targets.append(eq_data['failure_within_24h'].iloc[i+seq_len-1])
        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    @staticmethod
    def preprocess_data(data):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data['hour'] = data['timestamp'].dt.hour
        data['day_of_week'] = data['timestamp'].dt.dayofweek
        encoder = LabelEncoder()
        data['equipment_encoded'] = encoder.fit_transform(data['equipment_id'])
        feature_cols = []
        required_cols = ['vibration_rms', 'temperature_bearing', 'pressure_oil', 'rpm', 'power_consumption', 'hour', 'day_of_week', 'equipment_encoded']
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

    @staticmethod
    def create_data_loaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size):
        datasets = [TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y)) for X, y in [(X_train, y_train), (X_val, y_val), (X_test, y_test)]]
        return [DataLoader(ds, batch_size=batch_size, shuffle=(i==0)) for i, ds in enumerate(datasets)]

    @staticmethod
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

    @staticmethod
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

    def run(self):
        for dir_name in ['models', 'data/raw', 'data/processed', 'logs']:
            os.makedirs(dir_name, exist_ok=True)
        print(f"Loading data from {'live sensors' if self.use_live else self.data_path}...")
        data = self.load_data(self.data_path, self.use_live, self.live_hours)
        if self.use_live:
            data.to_csv('data/processed/live_training_data.csv', index=False)
            print(f"Saved {len(data)} live data points to data/processed/live_training_data.csv")
        data, self.encoder = self.preprocess_data(data)
        X, y = self.create_sequences(data, seq_len=24)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
        X_train_scaled, X_val_scaled, X_test_scaled, self.scaler = self.scale_data(X_train, X_val, X_test)
        train_loader, val_loader, test_loader = self.create_data_loaders(X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, self.batch_size)
        self.n_features = X_train_scaled.shape[2]
        self.model = PredictiveMaintenanceLSTM(input_size=self.n_features, hidden_size=self.hidden_size, num_layers=self.num_layers).to(self.device)
        pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(self.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
        early_stopping = EarlyStopping(patience=7, min_delta=0.001)
        self.history = {'train_loss': [], 'val_loss': [], 'train_f1': [], 'val_f1': []}
        best_val_f1 = 0
        for epoch in range(self.epochs):
            self.model.train()
            train_loss, train_outputs, train_targets = 0, [], []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits.squeeze(), yb)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * xb.size(0)
                train_outputs.append(torch.sigmoid(logits.detach().cpu()).numpy())
                train_targets.append(yb.detach().cpu().numpy())
            train_loss /= len(train_loader.dataset)
            train_outputs = np.concatenate(train_outputs)
            train_targets = np.concatenate(train_targets)
            train_pred = (train_outputs > 0.5).astype(int)
            train_f1 = (2 * (train_pred * train_targets).sum()) / (train_pred.sum() + train_targets.sum() + 1e-8)
            self.model.eval()
            val_loss, val_outputs, val_targets = 0, [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = self.model(xb)
                    loss = criterion(logits.squeeze(), yb)
                    val_loss += loss.item() * xb.size(0)
                    val_outputs.append(torch.sigmoid(logits.cpu()).numpy())
                    val_targets.append(yb.cpu().numpy())
            val_loss /= len(val_loader.dataset)
            val_outputs = np.concatenate(val_outputs)
            val_targets = np.concatenate(val_targets)
            val_pred = (val_outputs > 0.5).astype(int)
            val_f1 = (2 * (val_pred * val_targets).sum()) / (val_pred.sum() + val_targets.sum() + 1e-8)
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_f1'].append(train_f1)
            self.history['val_f1'].append(val_f1)
            scheduler.step(val_loss)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'model_params': {'input_size': self.n_features, 'hidden_size': self.hidden_size, 'num_layers': self.num_layers}
                }, 'models/best_model.pth')
            if early_stopping(val_loss, self.model):
                break
        checkpoint = torch.load('models/best_model.pth', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        test_loss, test_outputs, test_targets = 0, [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = self.model(xb)
                loss = criterion(logits.squeeze(), yb)
                test_loss += loss.item() * xb.size(0)
                test_outputs.append(torch.sigmoid(logits.cpu()).numpy())
                test_targets.append(yb.cpu().numpy())
        test_loss /= len(test_loader.dataset)
        test_outputs = np.concatenate(test_outputs)
        test_targets = np.concatenate(test_targets)
        test_pred = (test_outputs > 0.5).astype(int)
        test_f1 = (2 * (test_pred * test_targets).sum()) / (test_pred.sum() + test_targets.sum() + 1e-8)
        self.test_metrics = {'test_loss': float(test_loss), 'f1_score': float(test_f1)}
        self.save_artifacts(self.model, self.scaler, self.encoder, self.history, self.test_metrics, argparse.Namespace(hidden_size=self.hidden_size, num_layers=self.num_layers), self.n_features)
        data_source = "live sensors" if self.use_live else "CSV file"
        print(f"Training completed using {data_source}. Test F1: {self.test_metrics['f1_score']:.4f}")
        return self.test_metrics

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

