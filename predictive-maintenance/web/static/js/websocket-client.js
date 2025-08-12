class WebSocketClient {
    constructor(url = window.location.origin) {
        this.socket = io(url);
        this.callbacks = {};
        this.connected = false;
        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.socket.on('connect', () => {
            this.connected = true;
            this.updateConnectionStatus(true);
            this.emit('connected');
        });

        this.socket.on('disconnect', () => {
            this.connected = false;
            this.updateConnectionStatus(false);
            this.emit('disconnected');
        });

        this.socket.on('live_data', (data) => {
            this.emit('liveData', data);
        });

        this.socket.on('prediction_update', (data) => {
            this.emit('predictionUpdate', data);
        });

        this.socket.on('system_alert', (data) => {
            this.emit('systemAlert', data);
        });
    }

    updateConnectionStatus(connected) {
        const indicator = document.querySelector('.live-indicator');
        if (indicator) {
            indicator.textContent = connected ? '🟢 Live' : '🔴 Disconnected';
            indicator.className = `live-indicator ${connected ? 'active' : ''}`;
        }
    }

    on(event, callback) {
        if (!this.callbacks[event]) {
            this.callbacks[event] = [];
        }
        this.callbacks[event].push(callback);
    }

    emit(event, data) {
        if (this.callbacks[event]) {
            this.callbacks[event].forEach(callback => callback(data));
        }
    }

    requestLiveData() {
        if (this.connected) {
            this.socket.emit('request_live_data');
        }
    }

    startPeriodicUpdates(interval = 5000) {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        this.updateInterval = setInterval(() => {
            this.requestLiveData();
        }, interval);
    }

    stopPeriodicUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
            this.updateInterval = null;
        }
    }
}

window.WebSocketClient = WebSocketClient;