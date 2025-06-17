#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import numpy as np
from datetime import datetime

sys.path.append('src')

from data.generate_data import generate_sensor_data
from src.utils.data_preprocessing import DataPreprocessor
from src.models.lstm_model import PredictiveMaintenanceLSTM
from src.models.trainer import ModelTrainer
from src.monitoring.model_monitor import model_monitor
from src.monitoring.data_drift import DataDriftDetector

def main():
    parser = argparse.ArgumentParser(description='Train Predictive Maintenance Model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-size', type=int, default=64, help='LSTM hidden size')
    parser.add_argument('--generate-data', action='store_true', help='Generate new training data')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE MODEL TRAINING")
    print("=" * 60)
    print(f"Training started at: {datetime.now()}")
    print()
    
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    print("\n1. DATA PREPARATION")
    print("-" * 30)
    
    if args.generate_data or not os.path.exists('data/raw/sensor_data.csv'):
        print("Generating sensor data...")
        generate_sensor_data(num_samples=10000, num_equipments=50)
    else:
        print("Using existing sensor data...")
    
    print("\n2. DATA PREPROCESSING")
    print("-" * 30)
    
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.prepare_data('data/raw/sensor_data.csv')
    
    print("\n3. MODEL INITIALIZATION")
    print("-" * 30)
    
    input_size = X_train.shape[2]
    print(f"Input size: {input_size}")
    print(f"Hidden size: {args.hidden_size}")
    print(f"Sequence length: {X_train.shape[1]}")
    
    model = PredictiveMaintenanceLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=2,
        dropout=0.2
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n4. MODEL TRAINING")
    print("-" * 30)
    
    trainer = ModelTrainer(model, device=device)
    
    final_metrics = trainer.train(
        X_train, y_train, X_test, y_test,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate
    )
    
    print("\n5. FINAL EVALUATION")
    print("-" * 30)
    
    print("Final Test Metrics:")
    for metric, value in final_metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print("\n6. MONITORING SETUP")
    print("-" * 30)
    
    drift_detector = DataDriftDetector()
    sample_data = X_train[:1000].reshape(-1, X_train.shape[2])
    drift_detector.set_reference_data(sample_data)
    
    print("Data drift detector initialized with training data")
    
    print("\n7. SAVING ARTIFACTS")
    print("-" * 30)
    
    os.makedirs('models', exist_ok=True)
    
    metadata = {
        'training_date': datetime.now().isoformat(),
        'training_params': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'hidden_size': args.hidden_size
        },
        'data_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'input_features': input_size,
            'sequence_length': X_train.shape[1]
        },
        'final_metrics': final_metrics,
        'device_used': str(device)
    }
    
    import json
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model artifacts saved:")
    print("  - models/best_model.pth")
    print("  - models/scaler.pkl")
    print("  - models/equipment_encoder.pkl")
    print("  - models/training_history.json")
    print("  - models/model_metadata.json")
    print("  - models/training_curves.png")
    
    print("\n8. API READINESS CHECK")
    print("-" * 30)
    
    try:
        from src.api.prediction_service import PredictionService
        
        prediction_service = PredictionService()
        
        if prediction_service.is_model_loaded():
            print("✓ Prediction service loaded successfully")
            
            test_sensor_data = [
                {
                    'vibration_rms': 1.5,
                    'temperature_bearing': 75.0,
                    'pressure_oil': 20.0,
                    'rpm': 1800,
                    'oil_quality_index': 85,
                    'power_consumption': 50.0
                }
            ] * 24
            
            result = prediction_service.predict_single('TEST_EQ', test_sensor_data)
            print(f"✓ Test prediction successful: {result['failure_probability']:.3f}")
            
        else:
            print("✗ Prediction service failed to load")
            
    except Exception as e:
        print(f"✗ API test failed: {str(e)}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the API server: python run_api.py")
    print("2. Open http://localhost:5000 in your browser")
    print("3. Test predictions using the web interface")
    print("4. Deploy to AWS using: docker-compose up")

if __name__ == "__main__":
    main()
