from datetime import datetime
import sys
import os

from flask import app, jsonify
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.live_sensor_collector import get_live_equipment_data

# Import the live sensor collector
try:
    from src.live_sensor_collector import get_live_equipment_data
    LIVE_DATA_AVAILABLE = True
    print("✅ Live data collector loaded successfully")
except ImportError as e:
    LIVE_DATA_AVAILABLE = False
    print(f"⚠️ Live data collector not available: {e}")

@app.route('/api/equipment/summary')
def get_equipment_summary():
    try:
        equipment_data = get_live_equipment_data()
        return jsonify(equipment_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ADD new route for live data status
@app.route('/api/live-status')
def get_live_status():
    """
    Check live data collection status
    """
    return jsonify({
        'live_data_available': LIVE_DATA_AVAILABLE,
        'thingspeak_enabled': True,
        'openweather_enabled': False,
        'timestamp': datetime.now().isoformat()
    })

# ADD new route for real-time data streaming
@app.route('/api/live-data')
def get_live_data():
    """
    Get current live sensor data
    """
    try:
        if LIVE_DATA_AVAILABLE:
            from src.live_sensor_collector import LiveSensorDataCollector
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

if __name__ == '__main__':
    print("🚀 Starting Predictive Maintenance System with Live Data Integration")
    print("📡 Live data sources: ThingSpeak public channels")
    print("🌐 Dashboard: http://localhost:5000")
    print("📊 Live Data API: http://localhost:5000/api/live-data")
    print("📈 Equipment Summary: http://localhost:5000/api/equipment/summary")
    
    app.run(debug=True, host='0.0.0.0', port=5000)