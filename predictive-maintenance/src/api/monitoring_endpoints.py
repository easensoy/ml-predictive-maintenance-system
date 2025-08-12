from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import logging
from ..utils.health_monitor import health_monitor
from ..models.performance_tracker import performance_tracker

logger = logging.getLogger(__name__)

monitoring_bp = Blueprint('monitoring', __name__, url_prefix='/api/monitoring')

@monitoring_bp.route('/health', methods=['GET'])
def get_system_health():
    try:
        health_status = health_monitor.get_health_status()
        model_health = performance_tracker.get_model_health_report()
        
        combined_status = {
            'timestamp': datetime.now().isoformat(),
            'system': health_status,
            'model': model_health,
            'overall_status': _determine_overall_status(health_status, model_health)
        }
        
        return jsonify(combined_status)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'error': 'Health check unavailable'}), 500

@monitoring_bp.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        hours = request.args.get('hours', 24, type=int)
        
        system_trend = health_monitor.get_performance_trend(hours)
        model_metrics = performance_tracker.get_recent_metrics(days=hours//24 or 1)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'period_hours': hours,
            'system_performance': system_trend,
            'model_performance': model_metrics
        })
    except Exception as e:
        logger.error(f"Metrics retrieval failed: {e}")
        return jsonify({'error': 'Metrics unavailable'}), 500

@monitoring_bp.route('/alerts', methods=['GET'])
def get_alerts():
    try:
        active_only = request.args.get('active_only', 'true').lower() == 'true'
        
        model_alerts = performance_tracker.get_active_alerts()
        
        system_alerts = []
        if hasattr(health_monitor, 'alerts'):
            recent_cutoff = datetime.now() - timedelta(hours=24)
            system_alerts = [
                alert for alert in health_monitor.alerts 
                if datetime.fromisoformat(alert['timestamp']) > recent_cutoff
            ]
        
        all_alerts = []
        for alert in model_alerts:
            all_alerts.append({
                'source': 'model',
                'timestamp': alert.get('timestamp'),
                'type': alert.get('alert_type'),
                'message': alert.get('message'),
                'severity': alert.get('severity')
            })
        
        for alert in system_alerts:
            all_alerts.append({
                'source': 'system',
                'timestamp': alert.get('timestamp'),
                'type': alert.get('type'),
                'message': f"{alert.get('metric')}: {alert.get('value'):.1f}% (threshold: {alert.get('threshold'):.1f}%)",
                'severity': alert.get('severity')
            })
        
        all_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'total_alerts': len(all_alerts),
            'alerts': all_alerts[:50]
        })
    except Exception as e:
        logger.error(f"Alert retrieval failed: {e}")
        return jsonify({'error': 'Alerts unavailable'}), 500

@monitoring_bp.route('/performance/degradation', methods=['GET'])
def check_performance_degradation():
    try:
        window_days = request.args.get('window_days', 7, type=int)
        degradation_alerts = performance_tracker.detect_performance_degradation(window_days)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'window_days': window_days,
            'degradation_detected': len(degradation_alerts) > 0,
            'alerts': degradation_alerts
        })
    except Exception as e:
        logger.error(f"Degradation check failed: {e}")
        return jsonify({'error': 'Degradation check unavailable'}), 500

@monitoring_bp.route('/logs/prediction', methods=['POST'])
def log_prediction():
    try:
        data = request.get_json()
        equipment_id = data.get('equipment_id')
        predicted_value = data.get('predicted_value')
        actual_value = data.get('actual_value')
        confidence = data.get('confidence', 0.8)
        model_version = data.get('model_version', 'v1.0')
        
        if not equipment_id or predicted_value is None:
            return jsonify({'error': 'Missing required fields'}), 400
        
        performance_tracker.log_prediction(
            equipment_id=equipment_id,
            predicted_value=predicted_value,
            actual_value=actual_value,
            confidence=confidence,
            model_version=model_version
        )
        
        return jsonify({
            'message': 'Prediction logged successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Prediction logging failed: {e}")
        return jsonify({'error': 'Prediction logging failed'}), 500

@monitoring_bp.route('/dashboard', methods=['GET'])
def get_monitoring_dashboard():
    try:
        health_status = health_monitor.get_health_status()
        model_health = performance_tracker.get_model_health_report()
        recent_alerts = performance_tracker.get_active_alerts()
        
        dashboard_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'system_status': health_status.get('status'),
                'model_status': model_health.get('status'),
                'total_alerts': len(recent_alerts),
                'uptime_hours': health_status.get('uptime_hours', 0)
            },
            'system_metrics': {
                'cpu_percent': health_status.get('latest_metrics', {}).get('cpu_percent', 0),
                'memory_percent': health_status.get('latest_metrics', {}).get('memory_percent', 0),
                'disk_percent': health_status.get('latest_metrics', {}).get('disk_percent', 0)
            },
            'model_metrics': {
                'accuracy': model_health.get('latest_metrics', {}).get('accuracy', 0),
                'f1_score': model_health.get('latest_metrics', {}).get('f1_score', 0),
                'total_predictions': model_health.get('latest_metrics', {}).get('total_predictions', 0)
            },
            'recent_alerts': recent_alerts[:10]
        }
        
        return jsonify(dashboard_data)
    except Exception as e:
        logger.error(f"Dashboard data retrieval failed: {e}")
        return jsonify({'error': 'Dashboard unavailable'}), 500

@monitoring_bp.route('/start', methods=['POST'])
def start_monitoring():
    try:
        interval = request.json.get('interval_seconds', 60) if request.is_json else 60
        health_monitor.start_monitoring(interval)
        
        return jsonify({
            'message': 'Monitoring started successfully',
            'interval_seconds': interval,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        return jsonify({'error': 'Failed to start monitoring'}), 500

@monitoring_bp.route('/stop', methods=['POST'])
def stop_monitoring():
    try:
        health_monitor.stop_monitoring()
        
        return jsonify({
            'message': 'Monitoring stopped successfully',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        return jsonify({'error': 'Failed to stop monitoring'}), 500

def _determine_overall_status(system_health: dict, model_health: dict) -> str:
    system_status = system_health.get('status', 'UNKNOWN')
    model_status = model_health.get('status', 'UNKNOWN')
    
    status_priority = {'CRITICAL': 0, 'WARNING': 1, 'HEALTHY': 2, 'UNKNOWN': 3}
    
    min_priority = min(
        status_priority.get(system_status, 3),
        status_priority.get(model_status, 3)
    )
    
    for status, priority in status_priority.items():
        if priority == min_priority:
            return status
    
    return 'UNKNOWN'