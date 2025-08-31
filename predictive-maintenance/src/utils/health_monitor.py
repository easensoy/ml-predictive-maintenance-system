import psutil
import time
import json
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import threading
import logging
import os

logger = logging.getLogger(__name__)

class SystemHealthMonitor:
    def __init__(self, config_path: Optional[str] = None):
        self.metrics_history = []
        self.alerts = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set alert thresholds from config
        self.alert_thresholds = {
            'cpu_percent': self.config.get('thresholds', {}).get('cpu_usage', 80.0),
            'memory_percent': self.config.get('thresholds', {}).get('memory_usage', 85.0),
            'disk_percent': self.config.get('thresholds', {}).get('disk_usage', 90.0),
            'prediction_latency': self.config.get('thresholds', {}).get('response_latency', 5.0)
        }
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict:
        """Load configuration from YAML file with fallback to defaults"""
        if config_path is None:
            # Try to find config file in standard location
            current_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(current_dir, '..', '..', 'config', 'health_monitor_config.yaml')
        
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    logger.info(f"Loaded health monitor configuration from {config_path}")
                    return config
            else:
                logger.warning(f"Config file not found at {config_path}, using default values")
                return {}
        except Exception as e:
            logger.error(f"Error loading config file {config_path}: {e}")
            return {}
    
    def start_monitoring(self, interval_seconds: Optional[int] = None):
        if self.monitoring_active:
            return
        
        # Use provided interval or fall back to config or default
        if interval_seconds is None:
            interval_seconds = self.config.get('monitoring', {}).get('interval_seconds', 60)
        
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
        max_history = self.config.get('monitoring', {}).get('max_metrics_history', 1440)
        
        while self.monitoring_active:
            try:
                metrics = self.collect_system_metrics()
                self.metrics_history.append(metrics)
                
                if len(self.metrics_history) > max_history:
                    self.metrics_history = self.metrics_history[-max_history:]
                
                self._check_alerts(metrics)
                time.sleep(interval_seconds)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(interval_seconds)
    
    def collect_system_metrics(self) -> Dict:
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        bytes_to_gb = self.config.get('system', {}).get('bytes_to_gb', 1073741824)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / bytes_to_gb,
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / bytes_to_gb,
            'process_count': len(psutil.pids()),
            'load_average': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        }
    
    def _check_alerts(self, metrics: Dict):
        high_severity_multiplier = self.config.get('alerts', {}).get('high_severity_multiplier', 1.1)
        max_alerts = self.config.get('monitoring', {}).get('max_alerts_history', 500)
        
        for metric, threshold in self.alert_thresholds.items():
            if metric in metrics and metrics[metric] > threshold:
                alert = {
                    'timestamp': datetime.now().isoformat(),
                    'type': 'RESOURCE_THRESHOLD',
                    'metric': metric,
                    'value': metrics[metric],
                    'threshold': threshold,
                    'severity': 'HIGH' if metrics[metric] > threshold * high_severity_multiplier else 'MEDIUM'
                }
                self.alerts.append(alert)
                logger.warning(f"Resource alert: {metric} = {metrics[metric]:.1f}% (threshold: {threshold}%)")
        
        if len(self.alerts) > max_alerts:
            self.alerts = self.alerts[-max_alerts:]
    
    def get_health_status(self) -> Dict:
        if not self.metrics_history:
            return {'status': 'UNKNOWN', 'message': 'No metrics available'}
        
        latest = self.metrics_history[-1]
        recent_window_hours = self.config.get('alerts', {}).get('recent_window_hours', 1)
        recent_alerts = [a for a in self.alerts 
                        if datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=recent_window_hours)]
        
        # Get health scoring configuration
        health_config = self.config.get('health_score', {})
        penalties = health_config.get('penalties', {})
        
        # Calculate health score with configurable penalties
        health_score = 100
        cpu_threshold = self.alert_thresholds['cpu_percent'] * 0.875  # 70% of 80% threshold
        memory_threshold = self.alert_thresholds['memory_percent'] * 0.882  # 75% of 85% threshold  
        disk_threshold = self.alert_thresholds['disk_percent'] * 0.889  # 80% of 90% threshold
        max_alerts_threshold = penalties.get('max_alerts', 3)
        
        if latest['cpu_percent'] > cpu_threshold: 
            health_score -= penalties.get('high_cpu', 20)
        if latest['memory_percent'] > memory_threshold: 
            health_score -= penalties.get('high_memory', 20)
        if latest['disk_percent'] > disk_threshold: 
            health_score -= penalties.get('high_disk', 30)
        if len(recent_alerts) > max_alerts_threshold: 
            health_score -= penalties.get('per_alert', 5) * len(recent_alerts)
        
        # Get status thresholds
        thresholds = health_config.get('thresholds', {})
        healthy_threshold = thresholds.get('excellent', 80)
        warning_threshold = thresholds.get('good', 50)
        
        status = 'HEALTHY' if health_score > healthy_threshold else 'WARNING' if health_score > warning_threshold else 'CRITICAL'
        
        # Get time conversion factor
        time_factor = self.config.get('performance', {}).get('time_conversion_factor', 60)
        
        return {
            'status': status,
            'health_score': health_score,
            'latest_metrics': latest,
            'recent_alerts': len(recent_alerts),
            'uptime_hours': len(self.metrics_history) / time_factor
        }
    
    def get_performance_trend(self, hours: Optional[int] = None) -> Dict:
        # Use provided hours or fall back to config or default
        if hours is None:
            hours = self.config.get('performance', {}).get('trend_window_hours', 24)
            
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

# Global health monitor instance with default configuration
health_monitor = SystemHealthMonitor()