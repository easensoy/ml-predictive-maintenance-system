import numpy as np
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

logger = logging.getLogger(__name__)

class ModelPerformanceTracker:
    def __init__(self, db_path: str = 'model_performance.db'):
        self.db_path = db_path
        self.init_database()
        self.current_session_predictions = []
        
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                equipment_id TEXT NOT NULL,
                predicted_value REAL NOT NULL,
                actual_value REAL,
                prediction_confidence REAL,
                model_version TEXT,
                feature_drift_detected BOOLEAN DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                accuracy REAL,
                precision_score REAL,
                recall REAL,
                f1_score REAL,
                auc_score REAL,
                total_predictions INTEGER,
                drift_warnings INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT NOT NULL,
                resolved BOOLEAN DEFAULT 0
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, equipment_id: str, predicted_value: float, 
                      actual_value: Optional[float] = None, 
                      confidence: float = 0.8, model_version: str = "v1.0"):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, equipment_id, predicted_value, actual_value, prediction_confidence, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (datetime.now().isoformat(), equipment_id, predicted_value, 
              actual_value, confidence, model_version))
        
        conn.commit()
        conn.close()
        
        self.current_session_predictions.append({
            'equipment_id': equipment_id,
            'predicted': predicted_value,
            'actual': actual_value,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
        if len(self.current_session_predictions) > 1000:
            self.current_session_predictions = self.current_session_predictions[-1000:]
    
    def calculate_daily_metrics(self, target_date: Optional[str] = None) -> Dict:
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT predicted_value, actual_value, prediction_confidence
            FROM predictions 
            WHERE date(timestamp) = ? AND actual_value IS NOT NULL
        ''', (target_date,))
        
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            return {'date': target_date, 'message': 'No data with ground truth available'}
        
        predictions = np.array([r[0] for r in results])
        actuals = np.array([r[1] for r in results])
        confidences = np.array([r[2] for r in results])
        
        binary_predictions = (predictions > 0.5).astype(int)
        binary_actuals = actuals.astype(int)
        
        metrics = {
            'date': target_date,
            'accuracy': accuracy_score(binary_actuals, binary_predictions),
            'precision': precision_score(binary_actuals, binary_predictions, zero_division=0),
            'recall': recall_score(binary_actuals, binary_predictions, zero_division=0),
            'f1_score': f1_score(binary_actuals, binary_predictions, zero_division=0),
            'total_predictions': len(results),
            'avg_confidence': float(np.mean(confidences)),
            'high_confidence_ratio': float(np.mean(confidences > 0.8))
        }
        
        try:
            metrics['auc_score'] = roc_auc_score(binary_actuals, predictions)
        except ValueError:
            metrics['auc_score'] = 0.0
        
        self.save_daily_metrics(metrics)
        return metrics
    
    def save_daily_metrics(self, metrics: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO model_metrics 
            (date, accuracy, precision_score, recall, f1_score, auc_score, total_predictions)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (metrics['date'], metrics['accuracy'], metrics['precision'], 
              metrics['recall'], metrics['f1_score'], metrics.get('auc_score', 0), 
              metrics['total_predictions']))
        
        conn.commit()
        conn.close()
    
    def detect_performance_degradation(self, window_days: int = 7) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=window_days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT date, accuracy, f1_score, total_predictions
            FROM model_metrics 
            WHERE date >= ? 
            ORDER BY date
        ''', (cutoff_date,))
        
        results = cursor.fetchall()
        conn.close()
        
        if len(results) < 3:
            return []
        
        alerts = []
        recent_accuracy = [r[1] for r in results[-3:]]
        recent_f1 = [r[2] for r in results[-3:]]
        
        if len(recent_accuracy) >= 3:
            accuracy_trend = np.polyfit(range(len(recent_accuracy)), recent_accuracy, 1)[0]
            f1_trend = np.polyfit(range(len(recent_f1)), recent_f1, 1)[0]
            
            if accuracy_trend < -0.05:
                alerts.append({
                    'type': 'ACCURACY_DECLINE',
                    'message': f'Accuracy declining over {window_days} days (trend: {accuracy_trend:.3f})',
                    'severity': 'HIGH' if accuracy_trend < -0.1 else 'MEDIUM'
                })
            
            if f1_trend < -0.05:
                alerts.append({
                    'type': 'F1_DECLINE',
                    'message': f'F1 score declining over {window_days} days (trend: {f1_trend:.3f})',
                    'severity': 'HIGH' if f1_trend < -0.1 else 'MEDIUM'
                })
        
        current_accuracy = recent_accuracy[-1]
        if current_accuracy < 0.7:
            alerts.append({
                'type': 'LOW_ACCURACY',
                'message': f'Current accuracy below threshold: {current_accuracy:.3f}',
                'severity': 'CRITICAL' if current_accuracy < 0.5 else 'HIGH'
            })
        
        for alert in alerts:
            self.save_alert(alert)
        
        return alerts
    
    def save_alert(self, alert: Dict):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO performance_alerts (timestamp, alert_type, message, severity)
            VALUES (?, ?, ?, ?)
        ''', (datetime.now().isoformat(), alert['type'], alert['message'], alert['severity']))
        
        conn.commit()
        conn.close()
        
        logger.warning(f"Performance alert: {alert['message']}")
    
    def get_model_health_report(self) -> Dict:
        recent_metrics = self.get_recent_metrics(days=7)
        active_alerts = self.get_active_alerts()
        
        if not recent_metrics:
            return {
                'status': 'UNKNOWN',
                'message': 'Insufficient performance data',
                'alerts': active_alerts
            }
        
        latest_metrics = recent_metrics[-1]
        health_score = 100
        
        if latest_metrics.get('accuracy', 0) < 0.8:
            health_score -= 30
        if latest_metrics.get('f1_score', 0) < 0.7:
            health_score -= 25
        if len(active_alerts) > 0:
            health_score -= len(active_alerts) * 15
        
        status = 'HEALTHY' if health_score > 80 else 'WARNING' if health_score > 50 else 'CRITICAL'
        
        return {
            'status': status,
            'health_score': health_score,
            'latest_metrics': latest_metrics,
            'active_alerts': active_alerts,
            'days_of_data': len(recent_metrics)
        }
    
    def get_recent_metrics(self, days: int = 7) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            SELECT * FROM model_metrics WHERE date >= ? ORDER BY date
        ''', (cutoff_date,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results
    
    def get_active_alerts(self) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        recent_cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
        
        cursor.execute('''
            SELECT * FROM performance_alerts 
            WHERE timestamp >= ? AND resolved = 0 
            ORDER BY timestamp DESC
        ''', (recent_cutoff,))
        
        columns = [desc[0] for desc in cursor.description]
        results = [dict(zip(columns, row)) for row in cursor.fetchall()]
        
        conn.close()
        return results

performance_tracker = ModelPerformanceTracker()