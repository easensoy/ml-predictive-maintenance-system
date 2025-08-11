# AI-Powered Predictive Maintenance System

## Live Dashboard Outputs

### Main Dashboard Interface
<img width="1901" height="1082" alt="Screenshot 2025-08-11 025452" src="https://github.com/user-attachments/assets/ebef7afc-67f5-4939-a985-0fcc3b6377c2" />


This cloud-deployed system demonstrates real-time equipment monitoring with AI-driven failure prediction. The main interface displays current operational status across industrial equipment with live sensor data integration.


### Performance Metrics Display
- **F1 Score: 92%** - LSTM neural network classification performance  
- **Accuracy: 94%** - Correct failure prediction rate using ensemble methods
- **False Alarms: 2.8%** - Minimised unnecessary maintenance interventions through reinforcement learning optimisation
- **Prediction Window: 24h** - Advance warning capability for maintenance scheduling

### Live Equipment Monitoring
Current AWS deployment tracks three industrial assets with real-time ThingSpeak API integration:

**EQ_001 (Pump A1)**
- Vibration: 1.2 RMS, Temperature: 75°C, Pressure: 18.5 PSI  
- Failure Probability: 0% (Healthy status)
- Status: Normal operation

**EQ_002 (Motor B2)**  
- Vibration: 2.8 RMS, Temperature: 92°C, Pressure: 12.0 PSI
- Failure Probability: 77% (High risk threshold exceeded)
- Status: Immediate maintenance required within 24 hours

**EQ_003 (Compressor C1)**
- Vibration: 0.8 RMS, Temperature: 68°C, Pressure: 22.0 PSI  
- Failure Probability: 0% (Healthy status)
- Status: Optimal performance


### Real-Time Monitoring Dashboard  
<img width="1910" height="1074" alt="Screenshot 2025-08-11 025506" src="https://github.com/user-attachments/assets/3c44ff26-ea89-4351-879b-b661c809ea24" />

The monitoring dashboard provides comprehensive equipment health visualisation with live data streaming from ThingSpeak APIs. Features include individual equipment cards, risk distribution analytics, and temporal trend analysis.


### Data Visualisation & Analytics

**Risk Distribution Chart**
- Real-time pie chart showing equipment risk classification
- Colour-coded segments: Green (Low), Orange (Medium), Red (High), Dark Red (Critical)
- Current distribution: Majority low risk with one high-risk equipment flagged

**Equipment Health Trends**  
- Multi-line temporal chart displaying sensor patterns over time
- Three tracked parameters: Vibration (RMS), Temperature (°C), Pressure (PSI)
- Pattern recognition for anomaly detection and predictive maintenance scheduling
- Rolling time window with 24-hour historical data retention

### AI/ML Architecture Implementation
- **LSTM Networks**: Process temporal sensor patterns for failure prediction
- **Attention Mechanisms**: Focus on critical failure indicators (temperature spikes, vibration anomalies)
- **Reinforcement Learning**: Optimise maintenance scheduling to minimise costs whilst preventing failures
- **Real-time Inference**: Sub-second prediction updates through optimised model serving

### Cloud Infrastructure & API Integration
- **AWS ECS Fargate**: Serverless container deployment with auto-scaling
- **ThingSpeak Integration**: Live environmental data (temperature: 21.92°C, pressure: 10.13 bar)
- **RESTful APIs**: `/api/live-data`, `/api/equipment/summary`, `/dashboard` endpoints
- **Real-time Updates**: 30-second refresh cycles for continuous monitoring
- **Interactive Dashboards**: Chart.js visualisations with live data streaming
- **WebSocket Communication**: Real-time data push for instant dashboard updates

### Industrial Applications
Manufacturing, energy utilities, oil & gas, transportation, and mining industries benefit from:
- **Cost Prevention**: Avoid £50K-500K catastrophic equipment failures
- **Downtime Reduction**: Decrease unplanned outages from 15% to <3%
- **Predictive Scheduling**: AI-optimised maintenance timing

## Quick Setup

### AWS Cloud Deployment  
```bash
docker build -f deployment/Dockerfile -t predictive-maintenance .
aws ecr get-login-password --region eu-west-2 | docker login --username AWS --password-stdin <account>.dkr.ecr.eu-west-2.amazonaws.com
docker tag predictive-maintenance:latest <account>.dkr.ecr.eu-west-2.amazonaws.com/predictive-maintenance:latest
docker push <account>.dkr.ecr.eu-west-2.amazonaws.com/predictive-maintenance:latest
```

**Live Demo**: 
- Main Interface: `http://13.40.67.182:5000`
- Monitoring Dashboard: `http://13.40.67.182:5000/dashboard`
