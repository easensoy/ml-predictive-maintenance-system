
from run_api import PredictiveMaintenanceAPI

api = PredictiveMaintenanceAPI()
api.register_additional_routes()
api.app.run(debug=True, host='0.0.0.0', port=5000)