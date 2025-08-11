from flask import Flask, request, jsonify, render_template
from datetime import datetime
import sys
import os
sys.path.append('.')

try:
    from src.api.prediction_service import PredictionService
    prediction_service = PredictionService()
except Exception as e:
    print(f"Warning: Could not load prediction service: {e}")
    prediction_service = None

try:
    from src.live_sensor_collector import get_live_equipment_data, LiveSensorDataCollector
    LIVE_DATA_AVAILABLE = True
    print("✅ Live data collector loaded successfully")
except ImportError as e:
    LIVE_DATA_AVAILABLE = False
    print(f"⚠️ Live data collector not available: {e}")

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'Predictive Maintenance API',
        'model_loaded': prediction_service is not None and prediction_service.model is not None,
        'live_data_available': LIVE_DATA_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/live-status')
def get_live_status():
    return jsonify({
        'live_data_available': LIVE_DATA_AVAILABLE,
        'thingspeak_enabled': True,
        'openweather_enabled': False,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/equipment/summary')
def get_equipment_summary():
    try:
        if LIVE_DATA_AVAILABLE:
            equipment_data = get_live_equipment_data()
            return jsonify(equipment_data)
        else:
            return jsonify({'error': 'Live data collection not available'}), 503
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/live-data')
def get_live_data():
    try:
        if LIVE_DATA_AVAILABLE:
            collector = LiveSensorDataCollector()
            live_data = collector.collect_live_data()
            
            return jsonify({
                'success': True,
                'data': live_data,
                'timestamp': live_data['collection_timestamp']
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Live data collection not available'
            }), 503
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if not prediction_service:
            return jsonify({'error': 'Prediction service not available'}), 503
            
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400
            
        result = prediction_service.predict_single(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    try:
        if not prediction_service:
            return jsonify({'error': 'Prediction service not available'}), 503
            
        data = request.json
        equipment_list = data.get('equipment_list', [])
        results = prediction_service.predict_batch(equipment_list)
        return jsonify({'predictions': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/equipment/demo')
def get_demo_data():
    try:
        import random
        
        demo_data = []
        equipment_ids = ['EQ_001', 'EQ_002', 'EQ_003', 'EQ_004', 'EQ_005']
        equipment_names = ['Turbine Generator', 'Air Compressor', 'Hydraulic Pump', 'Cooling System', 'Conveyor Motor']
        
        for i, equipment_id in enumerate(equipment_ids):
            equipment = {
                'equipment_id': equipment_id,
                'equipment_name': equipment_names[i],
                'vibration_rms': random.uniform(0.8, 2.8),
                'temperature_bearing': random.uniform(65, 95),
                'pressure_oil': random.uniform(10, 18),
                'rpm': random.uniform(1200, 1850),
                'oil_quality_index': random.uniform(45, 98),
                'power_consumption': random.uniform(35, 75),
                'data_source': 'Demo Generation'
            }
            
            if prediction_service and prediction_service.model:
                try:
                    prediction = prediction_service.predict_single({
                        'equipment_id': equipment_id,
                        'sensor_data': [equipment]
                    })
                    equipment.update({
                        'failure_probability': prediction.get('failure_probability', 0.1),
                        'risk_level': prediction.get('risk_level', 'LOW'),
                        'recommended_action': prediction.get('recommendation', 'Normal operation')
                    })
                except:
                    equipment.update({
                        'failure_probability': 0.5 if equipment['vibration_rms'] > 2.0 else 0.2,
                        'risk_level': 'MEDIUM' if equipment['vibration_rms'] > 2.0 else 'LOW',
                        'recommended_action': 'Monitor closely' if equipment['vibration_rms'] > 2.0 else 'Normal operation'
                    })
            else:
                risk_score = 0
                if equipment['vibration_rms'] > 2.5: risk_score += 0.4
                if equipment['temperature_bearing'] > 85: risk_score += 0.3
                if equipment['oil_quality_index'] < 50: risk_score += 0.3
                
                equipment.update({
                    'failure_probability': min(risk_score, 0.95),
                    'risk_level': 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW',
                    'recommended_action': 'Schedule maintenance' if risk_score > 0.6 else 'Monitor' if risk_score > 0.3 else 'Normal operation'
                })
            
            demo_data.append(equipment)
        
        return jsonify(demo_data)
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate demo data: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500