from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import logging
import os
from datetime import datetime
import traceback

from .prediction_service import PredictionService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, 
           template_folder='../../web/templates',
           static_folder='../../web/static')
CORS(app)

prediction_service = PredictionService()

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
        'timestamp': datetime.now().isoformat(),
        'model_loaded': prediction_service.is_model_loaded(),
        'version': '1.0.0'
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        required_fields = ['equipment_id', 'sensor_data']
        if not all(field in data for field in required_fields):
            return jsonify({
                'error': f'Missing required fields: {required_fields}'
            }), 400
        result = prediction_service.predict_single(
            equipment_id=data['equipment_id'],
            sensor_data=data['sensor_data']
        )
        logger.info(f"Prediction made for {data['equipment_id']}: {result['failure_probability']:.3f}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    try:
        data = request.get_json()
        if not data or 'equipments' not in data:
            return jsonify({'error': 'No equipment data provided'}), 400
        results = []
        for equipment_data in data['equipments']:
            try:
                result = prediction_service.predict_single(
                    equipment_id=equipment_data['equipment_id'],
                    sensor_data=equipment_data['sensor_data']
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing {equipment_data.get('equipment_id', 'unknown')}: {str(e)}")
                results.append({
                    'equipment_id': equipment_data.get('equipment_id', 'unknown'),
                    'error': str(e)
                })
        return jsonify({'results': results})
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/model/metrics')
def get_model_metrics():
    try:
        metrics = prediction_service.get_model_metrics()
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        return jsonify({'error': 'Could not retrieve metrics'}), 500

@app.route('/api/model/retrain', methods=['POST'])
def trigger_retrain():
    try:
        return jsonify({
            'message': 'Retraining triggered',
            'status': 'queued',
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Retrain error: {str(e)}")
        return jsonify({'error': 'Could not trigger retraining'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
