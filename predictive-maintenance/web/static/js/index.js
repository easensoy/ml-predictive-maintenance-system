document.addEventListener('DOMContentLoaded', function() {
    initializePage();
    loadEquipmentData();
});

function initializePage() {
    const form = document.getElementById('predictionForm');
    if (form) {
        form.addEventListener('submit', handlePredictionSubmit);
    }
    
    initializeStatCounters();
    setupEquipmentCards();
}

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
        } else {
            generateDemoEquipment();
        }
    } catch (error) {
        console.log('Equipment data not available, using demo mode');
        generateDemoEquipment();
    }
}

function displayPredictionResult(result) {
    const resultContainer = document.getElementById('predictionResult') || createResultContainer();
    
    clearElement(resultContainer);
    
    const riskLevel = getRiskLevel(result.failure_probability);
    const riskColor = getRiskColor(riskLevel);
    
    const resultDiv = createElement('div', 'prediction-result');
    resultDiv.style.cssText = `
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        margin-top: 20px;
        border-left: 5px solid ${riskColor};
    `;
    
    const title = createElement('h3');
    title.textContent = 'Prediction Results';
    title.style.cssText = 'color: #333; margin-bottom: 15px;';
    resultDiv.appendChild(title);
    
    const metricsGrid = createElement('div');
    metricsGrid.style.cssText = 'display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;';
    
    const riskMetric = createMetricCard(Math.round(result.failure_probability * 100) + '%', 'Failure Risk', riskColor);
    const levelMetric = createMetricCard(riskLevel, 'Risk Level', riskColor);
    const confidenceMetric = createMetricCard((result.confidence ? Math.round(result.confidence * 100) : 95) + '%', 'Confidence', '#4a69bd');
    
    metricsGrid.appendChild(riskMetric);
    metricsGrid.appendChild(levelMetric);
    metricsGrid.appendChild(confidenceMetric);
    resultDiv.appendChild(metricsGrid);
    
    const recommendation = createElement('div');
    recommendation.style.cssText = 'margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px;';
    
    const recTitle = createElement('strong');
    recTitle.textContent = 'Recommendation: ';
    
    const recText = createElement('span');
    recText.textContent = getRecommendation(riskLevel);
    
    recommendation.appendChild(recTitle);
    recommendation.appendChild(recText);
    resultDiv.appendChild(recommendation);
    
    resultContainer.appendChild(resultDiv);
}

function createMetricCard(value, label, color) {
    const card = createElement('div');
    card.style.textAlign = 'center';
    
    const valueDiv = createElement('div');
    valueDiv.textContent = value;
    valueDiv.style.cssText = `font-size: 2em; font-weight: bold; color: ${color};`;
    
    const labelDiv = createElement('div');
    labelDiv.textContent = label;
    labelDiv.style.color = '#666';
    
    card.appendChild(valueDiv);
    card.appendChild(labelDiv);
    
    return card;
}

function updateEquipmentGrid(equipmentData) {
    const gridContainer = document.querySelector('.equipment-grid');
    if (!gridContainer) return;
    
    clearElement(gridContainer);
    
    equipmentData.forEach(equipment => {
        const card = createEquipmentOverviewCard(equipment);
        gridContainer.appendChild(card);
    });
}

function createEquipmentOverviewCard(equipment) {
    const card = createElement('div', 'equipment-card');
    
    const riskLevel = getRiskLevel(equipment.failure_probability);
    const statusClass = getStatusClass(riskLevel);
    
    const header = createElement('div', 'equipment-header');
    
    const infoDiv = createElement('div');
    const nameDiv = createElement('div', 'equipment-name');
    nameDiv.textContent = equipment.equipment_id;
    const typeDiv = createElement('div', 'equipment-type');
    typeDiv.textContent = getEquipmentType(equipment.equipment_id);
    
    infoDiv.appendChild(nameDiv);
    infoDiv.appendChild(typeDiv);
    
    const statusBadge = createElement('div', `status-badge status-${statusClass}`);
    statusBadge.textContent = riskLevel;
    
    header.appendChild(infoDiv);
    header.appendChild(statusBadge);
    card.appendChild(header);
    
    const metricsDiv = createElement('div', 'equipment-metrics');
    
    const metrics = [
        { label: 'Vibration:', value: (equipment.vibration_rms || 0).toFixed(1) + ' RMS' },
        { label: 'Temperature:', value: Math.round(equipment.temperature_bearing || 0) + '°C' },
        { label: 'Pressure:', value: (equipment.pressure_oil || 0).toFixed(1) + ' PSI' },
        { label: 'RPM:', value: equipment.rpm || 0 },
        { label: 'Oil Quality:', value: Math.round((equipment.oil_quality || 0) * 100) + '%' },
        { label: 'Power:', value: (equipment.power_consumption || 0).toFixed(1) + ' kW' }
    ];
    
    metrics.forEach(metric => {
        const metricRow = createMetricRow(metric.label, metric.value);
        metricsDiv.appendChild(metricRow);
    });
    
    card.appendChild(metricsDiv);
    
    const probabilityDiv = createProbabilityDisplay(equipment.failure_probability);
    card.appendChild(probabilityDiv);
    
    const recommendationDiv = createRecommendationDisplay(riskLevel);
    card.appendChild(recommendationDiv);
    
    card.addEventListener('click', () => {
        populateFormWithEquipment(equipment);
    });
    
    return card;
}

function createMetricRow(label, value) {
    const row = createElement('div', 'metric-row');
    
    const labelSpan = createElement('span', 'metric-label');
    labelSpan.textContent = label;
    
    const valueSpan = createElement('span', 'metric-value');
    valueSpan.textContent = value;
    
    row.appendChild(labelSpan);
    row.appendChild(valueSpan);
    
    return row;
}

function createProbabilityDisplay(probability) {
    const container = createElement('div', 'failure-probability');
    
    const label = createElement('div', 'probability-label');
    label.textContent = 'Failure Probability';
    
    const value = createElement('div', 'probability-value');
    value.textContent = Math.round(probability * 100) + '%';
    
    const barContainer = createElement('div', 'probability-bar');
    const barFill = createElement('div', 'probability-fill');
    barFill.style.width = (probability * 100) + '%';
    
    barContainer.appendChild(barFill);
    container.appendChild(label);
    container.appendChild(value);
    container.appendChild(barContainer);
    
    return container;
}

function createRecommendationDisplay(riskLevel) {
    const container = createElement('div', 'recommendation');
    
    const title = createElement('div', 'recommendation-title');
    title.textContent = 'Recommendation:';
    
    const text = createElement('div', 'recommendation-text');
    text.textContent = getRecommendation(riskLevel);
    
    container.appendChild(title);
    container.appendChild(text);
    
    return container;
}

function generateDemoEquipment() {
    const demoEquipment = [
        {
            equipment_id: 'EQ_001',
            vibration_rms: 1.2,
            temperature_bearing: 75,
            pressure_oil: 18.5,
            rpm: 1750,
            oil_quality: 0.85,
            power_consumption: 48.0,
            failure_probability: 0.05
        },
        {
            equipment_id: 'EQ_002',
            vibration_rms: 2.8,
            temperature_bearing: 92,
            pressure_oil: 12.0,
            rpm: 1650,
            oil_quality: 0.45,
            power_consumption: 65.0,
            failure_probability: 0.95
        },
        {
            equipment_id: 'EQ_003',
            vibration_rms: 0.8,
            temperature_bearing: 68,
            pressure_oil: 22.0,
            rpm: 1820,
            oil_quality: 0.90,
            power_consumption: 46.0,
            failure_probability: 0.02
        }
    ];
    
    updateEquipmentGrid(demoEquipment);
}

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

function getStatusClass(riskLevel) {
    return riskLevel.toLowerCase();
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
    
    const fields = {
        'equipment_id': equipment.equipment_id || '',
        'vibration': equipment.vibration_rms || '',
        'temperature': equipment.temperature_bearing || '',
        'pressure': equipment.pressure_oil || '',
        'rpm': equipment.rpm || '',
        'oil_quality': equipment.oil_quality || '',
        'power': equipment.power_consumption || ''
    };
    
    Object.keys(fields).forEach(field => {
        const input = form.querySelector(`[name="${field}"]`);
        if (input) {
            input.value = fields[field];
        }
    });
}

function showLoading(show) {
    const loadingElement = document.querySelector('.loading');
    if (loadingElement) {
        loadingElement.style.display = show ? 'block' : 'none';
    }
    
    const button = document.querySelector('.predict-btn');
    if (button) {
        button.disabled = show;
        button.textContent = show ? 'Analyzing...' : '🔮 Predict Failure Risk';
    }
}

function showError(message) {
    const errorContainer = document.getElementById('errorContainer') || createErrorContainer();
    clearElement(errorContainer);
    
    const errorDiv = createElement('div');
    errorDiv.style.cssText = `
        background: #fdebea;
        color: #e74c3c;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
        margin-top: 15px;
        text-align: center;
    `;
    errorDiv.textContent = message;
    
    errorContainer.appendChild(errorDiv);
    
    setTimeout(() => {
        clearElement(errorContainer);
    }, 5000);
}

function createResultContainer() {
    const container = createElement('div');
    container.id = 'predictionResult';
    document.querySelector('.equipment-section').appendChild(container);
    return container;
}

function createErrorContainer() {
    const container = createElement('div');
    container.id = 'errorContainer';
    document.querySelector('.equipment-section').appendChild(container);
    return container;
}

function initializeStatCounters() {
    const statNumbers = document.querySelectorAll('.stat-number');
    statNumbers.forEach(stat => {
        const targetValue = stat.textContent;
        animateCounter(stat, targetValue);
    });
}

function animateCounter(element, target) {
    const numericPart = target.match(/[\d.]+/);
    if (!numericPart) return;
    
    const numericTarget = parseFloat(numericPart[0]);
    const suffix = target.replace(numericPart[0], '');
    let current = 0;
    const increment = numericTarget / 30;
    
    const timer = setInterval(() => {
        current += increment;
        if (current >= numericTarget) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            const displayValue = suffix.includes('%') ? Math.floor(current) : current.toFixed(1);
            element.textContent = displayValue + suffix;
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

function createElement(tag, className = '') {
    const element = document.createElement(tag);
    if (className) {
        element.className = className;
    }
    return element;
}

function clearElement(element) {
    while (element.firstChild) {
        element.removeChild(element.firstChild);
    }
}