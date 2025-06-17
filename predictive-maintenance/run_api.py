#!/usr/bin/env python3
import sys
import os

sys.path.append('src')

from src.api.app import app

if __name__ == '__main__':
    print("Starting Predictive Maintenance API Server...")
    print("API will be available at: http://localhost:5000")
    print("Dashboard available at: http://localhost:5000/dashboard")
    print("API health check: http://localhost:5000/api/health")
    print("\nPress Ctrl+C to stop the server")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
