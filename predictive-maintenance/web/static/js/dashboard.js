// Dashboard Data Management
let equipmentData = [];
let riskChart = null;
let healthChart = null;

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();
    setInterval(loadDashboardData, 30000); // Refresh every 30 seconds
});

// API Data Loading
async function loadDashboardData() {
    try {
        const response = await fetch('/api/equipment/demo');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Response is not JSON format');
        }
        
        equipmentData = await response.json();
        displayEquipmentOverview();
        updateCharts();
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        showErrorMessage();
    }
}

// Equipment Display Functions
function displayEquipmentOverview() {
    const container = document.getElementById('equipmentOverview');
    container.innerHTML = '';

    equipmentData.forEach(equipment => {
        const card = createEquipmentCard(equipment);
        container.appendChild(card);
    });
}

function createEquipmentCard(equipment) {
    const card = document.createElement('div');
    const riskLevel = equipment.risk_level || 'LOW';
    const statusClass = getStatusClass(riskLevel);
    
    card.className = `equipment-card ${statusClass}`;
    
    card.innerHTML = `
        <div class="equipment-header">
            <div>
                <div class="equipment-name">${equipment.equipment_id}</div>
                <div class="equipment-type">${getEquipmentType(equipment.equipment_id)}</div>
            </div>
            <div class="status-badge status-${statusClass}">
                ${riskLevel}
            </div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-item">
                <div class="metric-value">${(equipment.vibration_rms || 0).toFixed(1)}</div>
                <div class="metric-label">Vibration</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${Math.round(equipment.temperature_bearing || 0)}°</div>
                <div class="metric-label">Temp</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${(equipment.pressure_oil || 0).toFixed(1)}</div>
                <div class="metric-label">Pressure</div>
            </div>
            <div class="metric-item">
                <div class="metric-value">${Math.round((equipment.failure_probability || 0) * 100)}%</div>
                <div class="metric-label">Risk</div>
            </div>
        </div>
        
        <div class="risk-info">
            <span>Status: ${getStatusText(riskLevel)}</span>
            <span>Risk: ${Math.round((equipment.failure_probability || 0) * 100)}%</span>
        </div>
    `;
    
    return card;
}

// Chart Management
function updateCharts() {
    updateRiskChart();
    updateHealthChart();
}

function updateRiskChart() {
    const ctx = document.getElementById('riskChart').getContext('2d');
    
    // Calculate risk distribution
    const riskCounts = {
        'LOW': 0,
        'MEDIUM': 0,
        'HIGH': 0,
        'CRITICAL': 0
    };
    
    equipmentData.forEach(eq => {
        const risk = eq.risk_level || 'LOW';
        riskCounts[risk]++;
    });
    
    if (riskChart) {
        riskChart.destroy();
    }
    
    riskChart = new Chart(ctx, {
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

function updateHealthChart() {
    const ctx = document.getElementById('healthChart').getContext('2d');
    
    // Generate trend data (simulated historical data)
    const labels = [];
    const vibrationData = [];
    const temperatureData = [];
    const pressureData = [];
    
    for (let i = 23; i >= 0; i--) {
        labels.push(`${i}h ago`);
        
        // Simulate historical data with realistic trends
        const baseVibration = equipmentData[0]?.vibration_rms || 1.0;
        const baseTemp = equipmentData[0]?.temperature_bearing || 70;
        const basePressure = equipmentData[0]?.pressure_oil || 20;
        
        vibrationData.push(baseVibration + (Math.random() - 0.5) * 0.4);
        temperatureData.push(baseTemp + (Math.random() - 0.5) * 8);
        pressureData.push(basePressure + (Math.random() - 0.5) * 3);
    }
    
    if (healthChart) {
        healthChart.destroy();
    }
    
    healthChart = new Chart(ctx, {
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
                    fill: false
                },
                {
                    label: 'Temperature (°C)',
                    data: temperatureData,
                    borderColor: '#f39c12',
                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                    tension: 0.4,
                    fill: false
                },
                {
                    label: 'Pressure (bar)',
                    data: pressureData,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    position: 'bottom'
                }
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
                    display: true,
                    title: {
                        display: true,
                        text: 'Sensor Values'
                    }
                }
            }
        }
    });
}

// Utility Functions
function getStatusClass(riskLevel) {
    const statusMap = {
        'LOW': 'low',
        'MEDIUM': 'medium',
        'HIGH': 'high',
        'CRITICAL': 'critical'
    };
    return statusMap[riskLevel] || 'low';
}

function getStatusText(riskLevel) {
    const statusText = {
        'LOW': 'Operational',
        'MEDIUM': 'Attention',
        'HIGH': 'Warning',
        'CRITICAL': 'Critical'
    };
    return statusText[riskLevel] || 'Operational';
}

function getEquipmentType(equipmentId) {
    // Simple mapping based on equipment ID patterns
    if (equipmentId.includes('PUMP')) return 'Centrifugal Pump';
    if (equipmentId.includes('MOTOR')) return 'Electric Motor';
    if (equipmentId.includes('COMP')) return 'Compressor';
    if (equipmentId.includes('TURB')) return 'Turbine';
    return 'Industrial Equipment';
}

// Error Handling
function showErrorMessage() {
    const container = document.getElementById('equipmentOverview');
    container.innerHTML = `
        <div class="error-message" style="
            grid-column: 1 / -1;
            background: #fdebea;
            color: #e74c3c;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
            border: 1px solid #f5c6cb;
        ">
            <h3>Unable to load equipment data</h3>
            <p>Please check your connection and try refreshing the page.</p>
            <button onclick="loadDashboardData()" style="
                margin-top: 10px;
                background: #e74c3c;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                cursor: pointer;
            ">Retry</button>
        </div>
    `;
}