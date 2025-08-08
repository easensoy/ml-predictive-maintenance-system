// Main Page Functionality
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
    loadEquipmentData();
});

// Page Initialization
function initializePage() {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', handlePredictionSubmit);
    }
    
    // Initialize any interactive elements
    initializeStatCounters();
    setupEquipmentCards();
}

// Equipment Prediction Form Handler
async function handlePredictionSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(event.target);
    const predictionData = {
        equipment_id: formData.get('equipment_id') || 'MANUAL_INPUT',
        sensor_data: {
            vibration_rms: parseFloat(formData.get('vibration')) || 0,
            temperature_bearing: parseFloat(formData.get('temperature')) || 0,
            pressure_oil: parseFloat(formData.get('pressure')) || 0,
            rpm: parseFloat(formData.get('rpm')) || 0,
            oil_quality: parseFloat(formData.get('oil_quality')) || 0,
            power_consumption: parseFloat(formData.get('power')) || 0
        }
    };
    
    try {
        showLoading(true);
        const result = await makePrediction(predictionData);
        displayPredictionResult(result);
    } catch (error) {
        showError('Failed to get prediction. Please try again.');
        console.error('Prediction error:', error);
    } finally {
        showLoading(false);
    }
}

// API Communication
async function makePrediction(data) {
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
}

async function loadEquipmentData() {
    try {
        const response = await fetch('/api/equipment/summary');
        if (response.ok) {
            const data = await response.json();
            updateEquipmentGrid(data);
        }
    } catch (error) {
        console.log('Equipment data not available, using demo mode');
        generateDemoEquipment();
    }
}

// UI Update Functions
function displayPredictionResult(result) {
    const resultContainer = document.getElementById('predictionResult') || createResultContainer();
    
    const riskLevel = getRiskLevel(result.failure_probability);
    const riskColor = getRiskColor(riskLevel);
    
    resultContainer.innerHTML = `
        <div class="prediction-result" style="
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            margin-top: 20px;
            border-left: 5px solid ${riskColor};
        ">
            <h3 style="color: #333; margin-bottom: 15px;">Prediction Results</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div style="text-align: center;">
                    <div style="font-size: 2em; font-weight: bold; color: ${riskColor};">
                        ${Math.round(result.failure_probability * 100)}%
                    </div>
                    <div style="color: #666;">Failure Risk</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.5em; font-weight: bold; color: ${riskColor};">
                        ${riskLevel}
                    </div>
                    <div style="color: #666;">Risk Level</div>
                </div>
                <div style="text-align: center;">
                    <div style="font-size: 1.2em; font-weight: bold; color: #4a69bd;">
                        ${result.confidence ? Math.round(result.confidence * 100) : 95}%
                    </div>
                    <div style="color: #666;">Confidence</div>
                </div>
            </div>
            <div style="margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                <strong>Recommendation:</strong> ${getRecommendation(riskLevel)}
            </div>
        </div>
    `;
}

function updateEquipmentGrid(equipmentData) {
    const gridContainer = document.querySelector('.equipment-grid');
    if (!gridContainer) return;
    
    gridContainer.innerHTML = '';
    
    equipmentData.forEach(equipment => {
        const card = createEquipmentOverviewCard(equipment);
        gridContainer.appendChild(card);
    });
}

function createEquipmentOverviewCard(equipment) {
    const card = document.createElement('div');
    const riskLevel = getRiskLevel(equipment.failure_probability);
    const riskColor = getRiskColor(riskLevel);
    
    card.className = 'equipment-card';
    card.innerHTML = `
        <h3 style="color: #333; margin-bottom: 10px;">${equipment.equipment_id}</h3>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="color: #666;">${getEquipmentType(equipment.equipment_id)}</span>
            <span style="
                background: ${riskColor}; 
                color: white; 
                padding: 4px 8px; 
                border-radius: 12px; 
                font-size: 0.8em;
                font-weight: bold;
            ">${riskLevel}</span>
        </div>
        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; font-size: 0.9em;">
            <div>Vibration: ${(equipment.vibration_rms || 0).toFixed(1)}</div>
            <div>Temp: ${Math.round(equipment.temperature_bearing || 0)}°C</div>
            <div>Pressure: ${(equipment.pressure_oil || 0).toFixed(1)} bar</div>
            <div>Risk: ${Math.round(equipment.failure_probability * 100)}%</div>
        </div>
    `;
    
    card.addEventListener('click', () => {
        populateFormWithEquipment(equipment);
    });
    
    return card;
}

// Demo Data Generation
function generateDemoEquipment() {
    const demoEquipment = [
        {
            equipment_id: 'PUMP-001',
            vibration_rms: 1.2,
            temperature_bearing: 75,
            pressure_oil: 22,
            failure_probability: 0.15
        },
        {
            equipment_id: 'MOTOR-003',
            vibration_rms: 2.1,
            temperature_bearing: 85,
            pressure_oil: 18,
            failure_probability: 0.45
        },
        {
            equipment_id: 'COMP-007',
            vibration_rms: 0.9,
            temperature_bearing: 68,
            pressure_oil: 25,
            failure_probability: 0.08
        },
        {
            equipment_id: 'TURB-012',
            vibration_rms: 3.2,
            temperature_bearing: 95,
            pressure_oil: 15,
            failure_probability: 0.78
        }
    ];
    
    updateEquipmentGrid(demoEquipment);
}

// Utility Functions
function getRiskLevel(probability) {
    if (probability < 0.2) return 'LOW';
    if (probability < 0.5) return 'MEDIUM';
    if (probability < 0.8) return 'HIGH';
    return 'CRITICAL';
}

function getRiskColor(riskLevel) {
    const colors = {
        'LOW': '#27ae60',
        'MEDIUM': '#f39c12',
        'HIGH': '#e67e22',
        'CRITICAL': '#e74c3c'
    };
    return colors[riskLevel] || '#27ae60';
}

function getRecommendation(riskLevel) {
    const recommendations = {
        'LOW': 'Equipment is operating normally. Continue standard maintenance schedule.',
        'MEDIUM': 'Monitor equipment closely. Consider scheduling preventive maintenance.',
        'HIGH': 'Schedule maintenance soon. Increase monitoring frequency.',
        'CRITICAL': 'Immediate attention required. Schedule emergency maintenance.'
    };
    return recommendations[riskLevel] || 'Monitor equipment status regularly.';
}

function getEquipmentType(equipmentId) {
    if (equipmentId.includes('PUMP')) return 'Centrifugal Pump';
    if (equipmentId.includes('MOTOR')) return 'Electric Motor';
    if (equipmentId.includes('COMP')) return 'Compressor';
    if (equipmentId.includes('TURB')) return 'Turbine';
    return 'Industrial Equipment';
}

function populateFormWithEquipment(equipment) {
    const form = document.getElementById('predictionForm');
    if (!form) return;
    
    form.querySelector('[name="equipment_id"]').value = equipment.equipment_id || '';
    form.querySelector('[name="vibration"]').value = equipment.vibration_rms || '';
    form.querySelector('[name="temperature"]').value = equipment.temperature_bearing || '';
    form.querySelector('[name="pressure"]').value = equipment.pressure_oil || '';
    form.querySelector('[name="rpm"]').value = equipment.rpm || '';
    form.querySelector('[name="oil_quality"]').value = equipment.oil_quality || '';
    form.querySelector('[name="power"]').value = equipment.power_consumption || '';
}

// Loading and Error States
function showLoading(show) {
    const loadingElement = document.querySelector('.loading');
    if (loadingElement) {
        loadingElement.style.display = show ? 'block' : 'none';
    }
}

function showError(message) {
    const errorContainer = document.getElementById('errorContainer') || createErrorContainer();
    errorContainer.innerHTML = `
        <div style="
            background: #fdebea;
            color: #e74c3c;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #f5c6cb;
            margin-top: 15px;
        ">
            ${message}
        </div>
    `;
    
    setTimeout(() => {
        errorContainer.innerHTML = '';
    }, 5000);
}

function createResultContainer() {
    const container = document.createElement('div');
    container.id = 'predictionResult';
    document.querySelector('.equipment-section').appendChild(container);
    return container;
}

function createErrorContainer() {
    const container = document.createElement('div');
    container.id = 'errorContainer';
    document.querySelector('.equipment-section').appendChild(container);
    return container;
}

// Statistics Animation
function initializeStatCounters() {
    const statNumbers = document.querySelectorAll('.stat-number');
    statNumbers.forEach(stat => {
        const targetValue = stat.textContent;
        animateCounter(stat, targetValue);
    });
}

function animateCounter(element, target) {
    const numericTarget = parseInt(target.replace(/[^\d]/g, ''));
    const suffix = target.replace(/[\d]/g, '');
    let current = 0;
    const increment = numericTarget / 30;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= numericTarget) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.floor(current) + suffix;
        }
    }, 50);
}

function setupEquipmentCards() {
    document.addEventListener('click', function(event) {
        if (event.target.closest('.equipment-card')) {
            const card = event.target.closest('.equipment-card');
            card.style.transform = 'scale(0.98)';
            setTimeout(() => {
                card.style.transform = '';
            }, 150);
        }
    });
}