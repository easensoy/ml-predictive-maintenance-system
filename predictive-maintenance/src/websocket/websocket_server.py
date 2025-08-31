from flask_socketio import SocketIO, emit
from datetime import datetime
import json
import threading
import time
import yaml

class WebSocketManager:
    def __init__(self, app, config_path=None):
        if config_path:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            self.update_interval = config['websocket']['update_interval']
        else:
            self.update_interval = 5
        self.socketio = SocketIO(app, cors_allowed_origins="*")
        self.active_connections = 0
        self.setup_events()
        
    def setup_events(self):
        @self.socketio.on('connect')
        def handle_connect():
            self.active_connections += 1
            emit('status', {'connected': True, 'timestamp': datetime.now().isoformat()})
            
        @self.socketio.on('disconnect')
        def handle_disconnect():
            self.active_connections -= 1
            
        @self.socketio.on('request_live_data')
        def handle_live_data_request():
            self.send_live_data()
            
    def send_live_data(self):
        try:
            from src.live_sensor_collector import get_live_equipment_data
            data = get_live_equipment_data()
            self.socketio.emit('live_data', data)
        except Exception:
            self.socketio.emit('live_data', {'error': 'Data collection failed'})
            
    def send_prediction_update(self, prediction_data):
        self.socketio.emit('prediction_update', prediction_data)
        
    def send_alert(self, alert_data):
        self.socketio.emit('system_alert', alert_data)
        
    def start_periodic_updates(self):
        def update_loop():
            while True:
                if self.active_connections > 0:
                    self.send_live_data()
                time.sleep(self.update_interval)
                
        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        
    def run(self, app, **kwargs):
        self.socketio.run(app, **kwargs)