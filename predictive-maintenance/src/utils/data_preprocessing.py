import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

class DataPreprocessor:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.equipment_encoder = LabelEncoder()
        self.sequence_length = 24
        
    def load_and_preprocess(self, file_path):
        df = pd.read_csv(file_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['equipment_id', 'timestamp'])
        feature_cols = ['vibration_rms', 'temperature_bearing', 'pressure_oil', 
                       'rpm', 'oil_quality_index', 'power_consumption']
        for col in feature_cols:
            df[f'{col}_rolling_mean_6h'] = df.groupby('equipment_id')[col].rolling(6).mean().values
            df[f'{col}_rolling_std_6h'] = df.groupby('equipment_id')[col].rolling(6).std().values
        df = df.dropna()
        df['equipment_encoded'] = self.equipment_encoder.fit_transform(df['equipment_id'])
        return df
    
    def create_sequences(self, df):
        sequences = []
        labels = []
        equipment_ids = []
        feature_cols = [col for col in df.columns if col not in 
                       ['timestamp', 'equipment_id', 'failure_within_24h']]
        for equipment in df['equipment_id'].unique():
            equipment_data = df[df['equipment_id'] == equipment]
            for i in range(len(equipment_data) - self.sequence_length):
                sequence = equipment_data[feature_cols].iloc[i:i+self.sequence_length].values
                label = equipment_data['failure_within_24h'].iloc[i+self.sequence_length]
                sequences.append(sequence)
                labels.append(label)
                equipment_ids.append(equipment)
        return np.array(sequences), np.array(labels), equipment_ids
    
    def scale_features(self, X_train, X_test):
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])
        X_train_scaled = self.scaler.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler.transform(X_test_reshaped)
        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)
        return X_train_scaled, X_test_scaled
    
    def prepare_data(self, file_path):
        print("Loading and preprocessing data...")
        df = self.load_and_preprocess(file_path)
        print("Creating sequences...")
        X, y, equipment_ids = self.create_sequences(df)
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("Scaling features...")
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.equipment_encoder, 'models/equipment_encoder.pkl')
        print(f"Data preparation complete:")
        print(f"  Training samples: {len(X_train_scaled)}")
        print(f"  Test samples: {len(X_test_scaled)}")
        print(f"  Feature dimensions: {X_train_scaled.shape}")
        print(f"  Positive samples (failures): {y_train.sum()}/{len(y_train)} ({y_train.mean():.2%})")
        return X_train_scaled, X_test_scaled, y_train, y_test
