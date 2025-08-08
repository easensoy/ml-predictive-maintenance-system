// Dashboard JavaScript - Clean Version (No innerHTML)
class DashboardManager {
    constructor() {
        this.equipmentData = [];
        this.riskChart = null;
        this.healthChart = null;
        this.init();
    }

    async init() {
        await this.loadDashboardData();
        this.setupEventListeners();
    }

    async loadDashboardData() {
        try {
            const response = await fetch('/api/equipment/demo');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const contentType = response.headers.get('content-type');
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('Response is not JSON format');
            }
            
            this.equipmentData = await response.json();
            this.displayEquipmentOverview();
            this.updateCharts();
        } catch (error) {
            console.error('Error loading dashboard data:', error);
            this.showErrorMessage();
        }
    }

    displayEquipmentOverview() {
        const container = document.getElementById('equipmentOverview');
        this.clearContainer(container);

        this.equipmentData.forEach(equipment => {
            const card = this.createEquipmentCard(equipment);
            container.appendChild(card);
        });
    }

    clearContainer(container) {
        while (container.firstChild) {
            container.removeChild(container.firstChild);
        }
    }

    createEquipmentCard(equipment) {
        const riskLevel = equipment.risk_level || 'LOW';
        const statusClass = this.getStatusClass(riskLevel);
        
        // Create main card element
        const card = document.createElement('div');
        card.className = `equipment-card ${statusClass}`;
        
        // Create header section
        const header = this.createCardHeader(equipment, riskLevel);
        card.appendChild(header);
        
        // Create metrics grid
        const metricsGrid = this.createMetricsGrid(equipment);
        card.appendChild(metricsGrid);
        
        // Create risk info
        const riskInfo = this.createRiskInfo(equipment, riskLevel);
        card.appendChild(riskInfo);
        
        return card;
    }

    createCardHeader(equipment, riskLevel) {
        const header = document.createElement('div');
        header.className = 'equipment-header';
        
        // Equipment info section
        const infoDiv = document.createElement('div');
        
        const nameDiv = document.createElement('div');
        nameDiv.className = 'equipment-name';
        nameDiv.textContent = equipment.equipment_id;
        
        const typeDiv = document.createElement('div');
        typeDiv.className = 'equipment-type';
        typeDiv.textContent = this.getEquipmentType(equipment.equipment_id);
        
        infoDiv.appendChild(nameDiv);
        infoDiv.appendChild(typeDiv);
        
        // Status badge
        const statusBadge = document.createElement('div');
        statusBadge.className = `status-badge status-${this.getStatusClass(riskLevel)}`;
        statusBadge.textContent = riskLevel;
        
        header.appendChild(infoDiv);
        header.appendChild(statusBadge);
        
        return header;
    }

    createMetricsGrid(equipment) {
        const grid = document.createElement('div');
        grid.className = 'metrics-grid';
        
        const metrics = [
            {
                value: (equipment.vibration_rms || 0).toFixed(1),
                label: 'Vibration'
            },
            {
                value: `${Math.round(equipment.temperature_bearing || 0)}°`,
                label: 'Temp'
            },
            {
                value: (equipment.pressure_oil || 0).toFixed(1),
                label: 'Pressure'
            },
            {
                value: `${Math.round((equipment.failure_probability || 0) * 100)}%`,
                label: 'Risk'
            }
        ];
        
        metrics.forEach(metric => {
            const item = document.createElement('div');
            item.className = 'metric-item';
            
            const value = document.createElement('div');
            value.className = 'metric-value';
            value.textContent = metric.value;
            
            const label = document.createElement('div');
            label.className = 'metric-label';
            label.textContent = metric.label;
            
            item.appendChild(value);
            item.appendChild(label);
            grid.appendChild(item);
        });
        
        return grid;
    }

    createRiskInfo(equipment, riskLevel) {
        const riskInfo = document.createElement('div');
        riskInfo.className = 'risk-info';
        
        const statusSpan = document.createElement('span');
        statusSpan.textContent = `Status: ${this.getStatusText(riskLevel)}`;
        
        const riskSpan = document.createElement('span');
        riskSpan.textContent = `Risk: ${Math.round((equipment.failure_probability || 0) * 100)}%`;
        
        riskInfo.appendChild(statusSpan);
        riskInfo.appendChild(riskSpan);
        
        return riskInfo;
    }

    updateCharts() {
        this.updateRiskChart();
        this.updateHealthChart();
    }

    updateRiskChart() {
        const ctx = document.getElementById('riskChart').getContext('2d');
        
        // Calculate risk distribution
        const riskCounts = {
            'LOW': 0,
            'MEDIUM': 0,
            'HIGH': 0,
            'CRITICAL': 0
        };
        
        this.equipmentData.forEach(eq => {
            const risk = eq.risk_level || 'LOW';
            riskCounts[risk]++;
        });
        
        if (this.riskChart) {
            this.riskChart.destroy();
        }
        
        this.riskChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical'],
                datasets: [{
                    data: [riskCounts.LOW, riskCounts.MEDIUM, riskCounts.HIGH, riskCounts.CRITICAL],
                    backgroundColor: [
                        '#27ae60',
                        '#f39c12', 
                        '#e67e22',
                        '#e74c3c'
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            padding: 20,
                            usePointStyle: true
                        }
                    }
                }
            }
        });
    }

    updateHealthChart() {
        const ctx = document.getElementById('healthChart').getContext('2d');
        
        // Generate trend data (simulated)
        const labels = [];
        const vibrationData = [];
        const temperatureData = [];
        const pressureData = [];
        
        for (let i = 23; i >= 0; i--) {
            labels.push(`${i}h ago`);
            
            // Simulate historical data with some trends
            const baseVibration = this.equipmentData[0]?.vibration_rms || 1.0;
            const baseTemp = this.equipmentData[0]?.temperature_bearing || 70;
            const basePressure = this.equipmentData[0]?.pressure_oil || 20;
            
            vibrationData.push(baseVibration + (Math.random() - 0.5) * 0.4);
            temperatureData.push(baseTemp + (Math.random() - 0.5) * 8);
            pressureData.push(basePressure + (Math.random() - 0.5) * 3);
        }
        
        if (this.healthChart) {
            this.healthChart.destroy();
        }
        
        this.healthChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Vibration (RMS)',
                        data: vibrationData,
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y'
                    },
                    {
                        label: 'Temperature (°C)',
                        data: temperatureData,
                        borderColor: '#f39c12',
                        backgroundColor: 'rgba(243, 156, 18, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y1'
                    },
                    {
                        label: 'Pressure (PSI)',
                        data: pressureData,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4,
                        yAxisID: 'y2'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                scales: {
                    x: {
                        display: true,
                        title: {
                            display: true,
                            text: 'Time'
                        }
                    },
                    y: {
                        type: 'linear',
                        display: true,
                        position: 'left',
                        title: {
                            display: true,
                            text: 'Vibration'
                        }
                    },
                    y1: {
                        type: 'linear',
                        display: false,
                        position: 'right'
                    },
                    y2: {
                        type: 'linear',
                        display: false,
                        position: 'right'
                    }
                },
                plugins: {
                    legend: {
                        position: 'bottom'
                    }
                }
            }
        });
    }

    showErrorMessage() {
        const container = document.getElementById('equipmentOverview');
        this.clearContainer(container);
        
        const errorDiv = document.createElement('div');
        errorDiv.style.cssText = `
            grid-column: 1 / -1;
            background: #fff3cd;
            color: #856404;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #ffeaa7;
            text-align: center;
        `;
        
        const errorText = document.createElement('p');
        errorText.textContent = 'Unable to load equipment data. Please check your connection and try again.';
        
        const retryButton = document.createElement('button');
        retryButton.textContent = 'Retry';
        retryButton.style.cssText = `
            margin-top: 10px;
            padding: 8px 16px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        `;
        retryButton.addEventListener('click', () => this.loadDashboardData());
        
        errorDiv.appendChild(errorText);
        errorDiv.appendChild(retryButton);
        container.appendChild(errorDiv);
    }

    setupEventListeners() {
        // Auto-refresh every 30 seconds
        setInterval(() => {
            this.loadDashboardData();
        }, 30000);
        
        // Manual refresh on visibility change
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                this.loadDashboardData();
            }
        });
    }

    // Helper methods
    getStatusClass(riskLevel) {
        const statusMap = {
            'LOW': 'healthy',
            'MEDIUM': 'warning',
            'HIGH': 'warning',
            'CRITICAL': 'critical'
        };
        return statusMap[riskLevel] || 'healthy';
    }

    getStatusText(riskLevel) {
        const textMap = {
            'LOW': 'Operational',
            'MEDIUM': 'Monitor',
            'HIGH': 'Attention Required',
            'CRITICAL': 'Immediate Action'
        };
        return textMap[riskLevel] || 'Unknown';
    }

    getEquipmentType(equipmentId) {
        const typeMap = {
            'EQ_001': 'Turbine Generator',
            'EQ_002': 'Air Compressor',
            'EQ_003': 'Hydraulic Pump',
            'EQ_004': 'Cooling System',
            'EQ_005': 'Conveyor Motor'
        };
        return typeMap[equipmentId] || 'Industrial Equipment';
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DashboardManager();
});