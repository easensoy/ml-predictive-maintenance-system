import yaml
from run_api import PredictiveMaintenanceAPI

# Load application configuration
try:
    with open('config/application_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    server_config = config['server']
except FileNotFoundError:
    server_config = {'debug': True, 'host': '0.0.0.0', 'port': 5000}

api = PredictiveMaintenanceAPI()
api.register_additional_routes()
api.app.run(
    debug=server_config['debug'],
    host=server_config['host'],
    port=server_config['port']
)