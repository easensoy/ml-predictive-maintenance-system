from flask import Flask, request, jsonify, render_template
import sys
import os
sys.path.append('.')

# Initialize prediction service with error handling
try:
    from src.api.prediction_service import PredictionService
    prediction_service = PredictionService()
except Exception as e:
    print(f"Warning: Could not load prediction service: {e}")
    prediction_service = None

app = Flask(__name__, template_folder='web/templates', static_folder='web/static')

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Equipment monitoring dashboard"""
    return render_template('dashboard.html')

@app.route('/api/health')
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'service': 'Predictive Maintenance API',
        'model_loaded': prediction_service is not None and prediction_service.model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Single equipment prediction"""
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
    """Batch equipment predictions"""
    try:
        if not prediction_service:
            return jsonify({'error': 'Prediction service not available'}), 503
            
        data = request.json
        equipment_list = data.get('equipment_list', [])
        results = prediction_service.predict_batch(equipment_list)
        return jsonify({'predictions': results})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/api/model/info')
def model_info():
    """Get model information and metrics"""
    if prediction_service:
        return jsonify(prediction_service.get_model_info())
    else:
        return jsonify({
            'model_type': 'Demo Mode',
            'status': 'Prediction service unavailable'
        })

@app.route('/api/equipment/demo')
def demo_equipment():
    """Generate demo equipment data for testing"""
    try:
        demo_data = [
            {
                'equipment_id': 'EQ_001',
                'vibration_rms': 1.2,
                'temperature_bearing': 75.0,
                'pressure_oil': 18.5,
                'rpm': 1750.0,
                'oil_quality_index': 85.0,
                'power_consumption': 48.0,
                'status': 'Running'
            },
            {
                'equipment_id': 'EQ_002', 
                'vibration_rms': 2.8,
                'temperature_bearing': 92.0,
                'pressure_oil': 12.0,
                'rpm': 1650.0,
                'oil_quality_index': 45.0,
                'power_consumption': 65.0,
                'status': 'Warning'
            },
            {
                'equipment_id': 'EQ_003',
                'vibration_rms': 0.8,
                'temperature_bearing': 68.0,
                'pressure_oil': 22.0,
                'rpm': 1820.0,
                'oil_quality_index': 90.0,
                'power_consumption': 46.0,
                'status': 'Healthy'
            }
        ]
        
        # Add predictions to each equipment
        for equipment in demo_data:
            if prediction_service:
                try:
                    prediction = prediction_service.predict_single(equipment)
                    equipment.update(prediction)
                except Exception as e:
                    # Fallback to manual risk calculation
                    equipment.update({
                        'failure_probability': 0.5 if equipment['vibration_rms'] > 2.0 else 0.2,
                        'risk_level': 'MEDIUM' if equipment['vibration_rms'] > 2.0 else 'LOW',
                        'recommended_action': 'Monitor closely' if equipment['vibration_rms'] > 2.0 else 'Normal operation'
                    })
            else:
                # Manual risk calculation when service unavailable
                risk_score = 0
                if equipment['vibration_rms'] > 2.5: risk_score += 0.4
                if equipment['temperature_bearing'] > 85: risk_score += 0.3
                if equipment['oil_quality_index'] < 50: risk_score += 0.3
                
                equipment.update({
                    'failure_probability': min(risk_score, 0.95),
                    'risk_level': 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW',
                    'recommended_action': 'Schedule maintenance' if risk_score > 0.6 else 'Monitor' if risk_score > 0.3 else 'Normal operation'
                })
        
        return jsonify(demo_data)
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate demo data: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("PREDICTIVE MAINTENANCE API SERVER")
    print("=" * 60)
    print("Starting server...")
    print("API will be available at: http://localhost:5000")
    print("Dashboard available at: http://localhost:5000/dashboard")
    print("API health check: http://localhost:5000/api/health")
    print("Demo data: http://localhost:5000/api/equipment/demo")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)