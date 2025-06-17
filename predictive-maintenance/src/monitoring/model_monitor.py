import json
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any
import os

logger = logging.getLogger(__name__)

class ModelMonitor:
    
    def __init__(self, metrics_file='monitoring/model_metrics.json'):
        self.metrics_file = metrics_file
        self.predictions_log = []
        self.performance_metrics = {}
        self.alerts = []
        
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        
        self.load_metrics()
    
    def log_prediction(self, equipment_id: str, prediction: float, 
                      actual: float = None, confidence: str = 'HIGH'):
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'equipment_id': equipment_id,
            'prediction': prediction,
            'actual': actual,
            'confidence': confidence,
            'day': datetime.now().strftime('%Y-%m-%d')
        }
        
        self.predictions_log.append(log_entry)
        
        if len(self.predictions_log) > 1000:
            self.predictions_log = self.predictions_log[-1000:]
        
        self.update_daily_metrics()
        
        self.check_alerts(log_entry)
    
    def update_daily_metrics(self):
        today = datetime.now().strftime('%Y-%m-%d')
        today_predictions = [p for p in self.predictions_log if p['day'] == today]
        
        if not today_predictions:
            return
        
        predictions_with_actual = [p for p in today_predictions if p['actual'] is not None]
        
        if predictions_with_actual:
            predictions = np.array([p['prediction'] for p in predictions_with_actual])
            actuals = np.array([p['actual'] for p in predictions_with_actual])
            
            binary_preds = (predictions > 0.5).astype(int)
            
            accuracy = np.mean(binary_preds == actuals)
            
            tp = np.sum((binary_preds == 1) & (actuals == 1))
            tn = np.sum((binary_preds == 0) & (actuals == 0))
            fp = np.sum((binary_preds == 1) & (actuals == 0))
            fn = np.sum((binary_preds == 0) & (actuals == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            self.performance_metrics[today] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'total_predictions': len(today_predictions),
                'predictions_with_feedback': len(predictions_with_actual),
                'avg_prediction': float(np.mean(predictions)),
                'high_risk_predictions': int(np.sum(predictions > 0.8))
            }
        else:
            predictions = np.array([p['prediction'] for p in today_predictions])
            
            self.performance_metrics[today] = {
                'total_predictions': len(today_predictions),
                'predictions_with_feedback': 0,
                'avg_prediction': float(np.mean(predictions)),
                'high_risk_predictions': int(np.sum(predictions > 0.8))
            }
        
        self.save_metrics()
    
    def check_alerts(self, log_entry: Dict):
        recent_predictions = [p for p in self.predictions_log 
                            if datetime.fromisoformat(p['timestamp']) > 
                            datetime.now() - timedelta(hours=1)]
        
        high_risk_count = sum(1 for p in recent_predictions if p['prediction'] > 0.8)
        
        if high_risk_count > 5:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'HIGH_RISK_SURGE',
                'message': f'Unusual number of high-risk predictions: {high_risk_count} in last hour',
                'severity': 'WARNING'
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])
        
        low_confidence_count = sum(1 for p in recent_predictions if p['confidence'] != 'HIGH')
        
        if low_confidence_count > len(recent_predictions) * 0.5 and len(recent_predictions) > 10:
            alert = {
                'timestamp': datetime.now().isoformat(),
                'type': 'LOW_CONFIDENCE',
                'message': f'Model confidence dropping: {low_confidence_count}/{len(recent_predictions)} low confidence predictions',
                'severity': 'WARNING'
            }
            self.alerts.append(alert)
            logger.warning(alert['message'])
        
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_system_health(self) -> Dict[str, Any]:
        recent_predictions = [p for p in self.predictions_log 
                            if datetime.fromisoformat(p['timestamp']) > 
                            datetime.now() - timedelta(hours=24)]
        
        health_score = 100
        
        recent_alerts = [a for a in self.alerts 
                        if datetime.fromisoformat(a['timestamp']) > 
                        datetime.now() - timedelta(hours=24)]
        
        health_score -= len(recent_alerts) * 10
        
        if len(recent_predictions) < 10:
            health_score -= 20
        
        high_risk_ratio = sum(1 for p in recent_predictions if p['prediction'] > 0.8) / max(len(recent_predictions), 1)
        if high_risk_ratio > 0.3:
            health_score -= 30
        
        health_score = max(0, min(100, health_score))
        
        status = 'HEALTHY' if health_score > 80 else 'WARNING' if health_score > 50 else 'CRITICAL'
        
        return {
            'status': status,
            'health_score': health_score,
            'total_predictions_24h': len(recent_predictions),
            'high_risk_predictions_24h': sum(1 for p in recent_predictions if p['prediction'] > 0.8),
            'recent_alerts': len(recent_alerts),
            'last_prediction': recent_predictions[-1]['timestamp'] if recent_predictions else None
        }
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        relevant_metrics = {date: metrics for date, metrics in self.performance_metrics.items() 
                          if date >= cutoff_date}
        
        if not relevant_metrics:
            return {'message': 'No performance data available'}
        
        total_predictions = sum(m.get('total_predictions', 0) for m in relevant_metrics.values())
        total_with_feedback = sum(m.get('predictions_with_feedback', 0) for m in relevant_metrics.values())
        
        days_with_feedback = [m for m in relevant_metrics.values() if m.get('predictions_with_feedback', 0) > 0]
        
        if days_with_feedback:
            avg_accuracy = np.mean([m['accuracy'] for m in days_with_feedback])
            avg_f1 = np.mean([m['f1_score'] for m in days_with_feedback])
        else:
            avg_accuracy = None
            avg_f1 = None
        
        return {
            'period_days': days,
            'total_predictions': total_predictions,
            'predictions_with_feedback': total_with_feedback,
            'feedback_rate': total_with_feedback / max(total_predictions, 1),
            'avg_accuracy': avg_accuracy,
            'avg_f1_score': avg_f1,
            'days_with_data': len(relevant_metrics)
        }
    
    def save_metrics(self):
        data = {
            'performance_metrics': self.performance_metrics,
            'alerts': self.alerts,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(self.metrics_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_metrics(self):
        try:
            if os.path.exists(self.metrics_file):
                with open(self.metrics_file, 'r') as f:
                    data = json.load(f)
                    self.performance_metrics = data.get('performance_metrics', {})
                    self.alerts = data.get('alerts', [])
        except Exception as e:
            logger.error(f"Error loading metrics: {str(e)}")
            self.performance_metrics = {}
            self.alerts = []

model_monitor = ModelMonitor()
