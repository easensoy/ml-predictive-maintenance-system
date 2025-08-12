import psutil
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import logging

logger = logging.getLogger(__name__)

class SystemHealthMonitor:
    def __init__(self):
        self.metrics_history = []
        self.alert_thresholds = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_percent': 90.0,
            'prediction_latency': 5.0
        }
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval_seconds: int = 60):
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval_seconds,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        logger.info("System health monitoring started")
    
    def stop_monitoring(self):
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        logger.info("System health monitoring stopped")
    
    def _monitor_loop(self, interval_seconds: int):
        while self.monitoring_active:
            try:
                metrics = self.collect_system_metrics()
                self.metrics_history.append(metrics)
                
                if len(self.metrics_history) > 1440:
                    self.metrics_history = self.metrics_history[-1440:]
                
                self._check_alerts(metrics)
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def collect_system_metrics(self) -> Dict:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024**3),
            'process_count': len(psutil.pids()),
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        }
    
    def _check_alerts(self, metrics: Dict):
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'RESOURCE_THRESHOLD',
                    'metric': metric,
                    'value': metrics[metric],
                    'threshold': threshold,
                    'severity': 'HIGH' if metrics[metric] > threshold * 1.1 else 'MEDIUM'
                }
                self.alerts.append(alert)
                logger.warning(f"Resource alert: {metric} = {metrics[metric]:.1f}% (threshold: {threshold}%)")
        
        if len(self.alerts) > 500:
            self.alerts = self.alerts[-500:]
    
    def get_health_status(self) -> Dict:
        if not self.metrics_history:
            return {'status': 'UNKNOWN', 'message': 'No metrics available'}
        
        latest = self.metrics_history[-1]
        recent_alerts = [a for a in self.alerts 
                        if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)]
        
        health_score = 100
        if latest['cpu_percent'] > 70: health_score -= 20
        if latest['memory_percent'] > 75: health_score -= 20
        if latest['disk_percent'] > 80: health_score -= 30
        if len(recent_alerts) > 3: health_score -= 20
        
        status = 'HEALTHY' if health_score > 80 else 'WARNING' if health_score > 50 else 'CRITICAL'
        
        return {
            'status': status,
            'health_score': health_score,
            'latest_metrics': latest,
            'recent_alerts': len(recent_alerts),
            'uptime_hours': len(self.metrics_history) / 60
        }
    
    def get_performance_trend(self, hours: int = 24) -> Dict:
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_metrics = [m for m in self.metrics_history 
                         if datetime.fromisoformat(m['timestamp']) > cutoff]
        
        if not recent_metrics:
            return {'message': 'No recent metrics available'}
        
        cpu_values = [m['cpu_percent'] for m in recent_metrics]
        memory_values = [m['memory_percent'] for m in recent_metrics]
        
        return {
            'period_hours': hours,
            'avg_cpu_percent': sum(cpu_values) / len(cpu_values),
            'max_cpu_percent': max(cpu_values),
            'avg_memory_percent': sum(memory_values) / len(memory_values),
            'max_memory_percent': max(memory_values),
            'samples_collected': len(recent_metrics)
        }

health_monitor = SystemHealthMonitor()