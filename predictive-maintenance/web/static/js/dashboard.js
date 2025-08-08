let equipmentData = [];
let riskChart = null;
let healthChart = null;

document.addEventListener('DOMContentLoaded', function() {
    loadDashboardData();
    setInterval(loadDashboardData, 30000);
});

async function loadDashboardData() {
    try {
        const response = await fetch('/api/equipment/demo');
        
        if (!response.ok) {
            console.log('Demo API not available, generating demo data');
            generateDemoData();
            return;
        }
        
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            generateDemoData();
            return;
        }
        
        equipmentData = await response.json();
        displayEquipmentOverview();
        updateCharts();
    } catch (error) {
        console.error('Error loading dashboard data:', error);
        generateDemoData();
    }
}

function generateDemoData() {
    equipmentData = [
        {
            equipment_id: 'PUMP-001',
            vibration_rms: 1.2,
            temperature_bearing: 75,
            pressure_oil: 18.5,
            rpm: 1750,
            oil_quality: 0.85,
            power_consumption: 48.0,
            failure_probability: 0.15,
            risk_level: 'LOW'
        },
        {
            equipment_id: 'MOTOR-003',
            vibration_rms: 2.1,
            temperature_bearing: 85,
            pressure_oil: 16.0,
            rpm: 1680,
            oil_quality: 0.72,
            power_consumption: 52.0,
            failure_probability: 0.45,
            risk_level: 'MEDIUM'
        },
        {
            equipment_id: 'COMP-007',
            vibration_rms: 0.9,
            temperature_bearing: 68,
            pressure_oil: 22.0,
            rpm: 1820,
            oil_quality: 0.90,
            power_consumption: 46.0,
            failure_probability: 0.08,
            risk_level: 'LOW'
        },
        {
            equipment_id: 'TURB-012',
            vibration_rms: 3.2,
            temperature_bearing: 95,
            pressure_oil: 12.0,
            rpm: 1520,
            oil_quality: 0.35,
            power_consumption: 68.0,
            failure_probability: 0.78,
            risk_level: 'CRITICAL'
        },
        {
            equipment_id: 'MOTOR-008',
            vibration_rms: 2.5,
            temperature_bearing: 88,
            pressure_oil: 14.5,
            rpm: 1640,
            oil_quality: 0.58,
            power_consumption: 55.0,
            failure_probability: 0.62,
            risk_level: 'HIGH'
        }
    ];
    
    displayEquipmentOverview();
    updateCharts();
}

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
    const riskLevel = equipment.risk_level || getRiskLevel(equipment.failure_probability);
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
    
    card.addEventListener('click', () => {
        showEquipmentDetails(equipment);
    });
    
    return card;
}

function updateCharts() {
    updateRiskChart();
    updateHealthChart();
}

function updateRiskChart() {
    const ctx = document.getElementById('riskChart');
    if (!ctx) return;
    
    const context = ctx.getContext('2d');
    
    const riskCounts = {
        'LOW': 0,
        'MEDIUM': 0,
        'HIGH': 0,
        'CRITICAL': 0
    };
    
    equipmentData.forEach(eq => {
        const risk = eq.risk_level || getRiskLevel(eq.failure_probability);
        riskCounts[risk]++;
    });
    
    if (riskChart) {
        riskChart.destroy();
    }
    
    riskChart = new Chart(context, {
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
                        usePointStyle: true,
                        font: {
                            size: 12
                        }
                    }
                }
            }
        }
    });
}

function updateHealthChart() {
    const ctx = document.getElementById('healthChart');
    if (!ctx) return;
    
    const context = ctx.getContext('2d');
    
    const labels = [];
    const vibrationData = [];
    const temperatureData = [];
    const pressureData = [];
    
    for (let i = 23; i >= 0; i--) {
        labels.push(`${i}h ago`);
        
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
    
    healthChart = new Chart(context, {
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
                    fill: false,
                    pointRadius: 2,
                    pointHoverRadius: 4
                },
                {
                    label: 'Temperature (°C)',
                    data: temperatureData,
                    borderColor: '#f39c12',
                    backgroundColor: 'rgba(243, 156, 18, 0.1)',
                    tension: 0.4,
                    fill: false,
                    pointRadius: 2,
                    pointHoverRadius: 4
                },
                {
                    label: 'Pressure (PSI)',
                    data: pressureData,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    tension: 0.4,
                    fill: false,
                    pointRadius: 2,
                    pointHoverRadius: 4
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
                    position: 'bottom',
                    labels: {
                        font: {
                            size: 12
                        }
                    }
                }
            },
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Time',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        display: false
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Sensor Values',
                        font: {
                            size: 12
                        }
                    },
                    grid: {
                        color: 'rgba(0,0,0,0.1)'
                    }
                }
            }
        }
    });
}

function getRiskLevel(probability) {
    if (probability < 0.2) return 'LOW';
    if (probability < 0.5) return 'MEDIUM';
    if (probability < 0.8) return 'HIGH';
    return 'CRITICAL';
}

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
    if (equipmentId.includes('PUMP')) return 'Centrifugal Pump';
    if (equipmentId.includes('MOTOR')) return 'Electric Motor';
    if (equipmentId.includes('COMP')) return 'Compressor';
    if (equipmentId.includes('TURB')) return 'Turbine';
    return 'Industrial Equipment';
}

function showEquipmentDetails(equipment) {
    alert(`Equipment: ${equipment.equipment_id}\nRisk Level: ${equipment.risk_level || getRiskLevel(equipment.failure_probability)}\nFailure Probability: ${Math.round(equipment.failure_probability * 100)}%\nVibration: ${equipment.vibration_rms} RMS\nTemperature: ${equipment.temperature_bearing}°C\nPressure: ${equipment.pressure_oil} PSI`);
}

function showErrorMessage() {
    const container = document.getElementById('equipmentOverview');
    container.innerHTML = `
        <div class="error-message">
            <h3>Unable to load equipment data</h3>
            <p>Please check your connection and try refreshing the page.</p>
            <button onclick="loadDashboardData()">Retry</button>
        </div>
    `;
}

window.addEventListener('resize', function() {
    if (riskChart) {
        riskChart.resize();
    }
    if (healthChart) {
        healthChart.resize();
    }
});