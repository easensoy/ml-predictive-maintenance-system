const DEMO_EQUIPMENT = [
    { equipment_id: 'PUMP_001', vibration_rms: 1.2, temperature_bearing: 75, pressure_oil: 18.5, rpm: 1750, oil_quality: 0.85, power_consumption: 48.0, failure_probability: 0.05 },
    { equipment_id: 'MOTOR_002', vibration_rms: 2.8, temperature_bearing: 92, pressure_oil: 12.0, rpm: 1650, oil_quality: 0.45, power_consumption: 65.0, failure_probability: 0.95 },
    { equipment_id: 'COMP_003', vibration_rms: 0.8, temperature_bearing: 68, pressure_oil: 22.0, rpm: 1820, oil_quality: 0.90, power_consumption: 46.0, failure_probability: 0.02 }
];

const RISK_CONFIG = {
    levels: { LOW: [0, 0.2], MEDIUM: [0.2, 0.5], HIGH: [0.5, 0.8], CRITICAL: [0.8, 1] },
    colors: { LOW: '#27ae60', MEDIUM: '#f39c12', HIGH: '#e67e22', CRITICAL: '#e74c3c' },
    texts: { LOW: 'Operational', MEDIUM: 'Attention', HIGH: 'Warning', CRITICAL: 'Critical' },
    recommendations: {
        LOW: 'Equipment is operating normally. Continue standard maintenance schedule.',
        MEDIUM: 'Monitor equipment closely. Consider scheduling preventive maintenance.',
        HIGH: 'Schedule maintenance soon. Increase monitoring frequency.',
        CRITICAL: 'Immediate attention required. Schedule emergency maintenance.'
    }
};

const $ = (selector, parent = document) => parent.querySelector(selector);
const $$ = (selector, parent = document) => [...parent.querySelectorAll(selector)];
const create = (tag, className, content) => {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (content) el.textContent = content;
    return el;
};

document.addEventListener('DOMContentLoaded', init);

function init() {
    $('#predictionForm').addEventListener('submit', handlePredict);
    loadEquipment();
    animateStats();
    setupCardClicks();
}

async function handlePredict(e) {
    e.preventDefault();
    const data = new FormData(e.target);
    const prediction = {
        equipment_id: data.get('equipment_id') || 'MANUAL_INPUT',
        sensor_data: Object.fromEntries(['vibration', 'temperature', 'pressure', 'rpm', 'oil_quality', 'power']
            .map(key => [key === 'vibration' ? 'vibration_rms' : 
                         key === 'temperature' ? 'temperature_bearing' : 
                         key === 'pressure' ? 'pressure_oil' : 
                         key === 'power' ? 'power_consumption' : key, 
                         parseFloat(data.get(key)) || 0]))
    };

    try {
        toggleLoading(true);
        const result = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(prediction)
        }).then(r => r.ok ? r.json() : Promise.reject(r.statusText));
        
        showResult(result);
    } catch (error) {
        showError('Prediction failed. Please try again.');
    } finally {
        toggleLoading(false);
    }
}

function loadEquipment() {
    fetch('/api/equipment/summary')
        .then(r => r.ok ? r.json() : Promise.reject())
        .then(updateGrid)
        .catch(() => updateGrid(DEMO_EQUIPMENT));
}

function updateGrid(equipment) {
    const grid = $('.equipment-grid');
    grid.replaceChildren(...equipment.map(createCard));
}

function createCard(eq) {
    const risk = getRisk(eq.failure_probability);
    const card = create('div', 'equipment-card');
    
    const header = create('div', 'equipment-header');
    const info = create('div');
    info.append(create('div', 'equipment-name', eq.equipment_id), create('div', 'equipment-type', getType(eq.equipment_id)));
    header.append(info, create('div', `status-badge status-${risk.toLowerCase()}`, risk));
    
    const metrics = create('div', 'equipment-metrics');
    [
        ['Vibration:', `${eq.vibration_rms?.toFixed(1)} RMS`],
        ['Temperature:', `${Math.round(eq.temperature_bearing)}°C`],
        ['Pressure:', `${eq.pressure_oil?.toFixed(1)} PSI`],
        ['RPM:', eq.rpm],
        ['Oil Quality:', `${Math.round(eq.oil_quality * 100)}%`],
        ['Power:', `${eq.power_consumption?.toFixed(1)} kW`]
    ].forEach(([label, value]) => {
        const row = create('div', 'metric-row');
        row.append(create('span', 'metric-label', label), create('span', 'metric-value', value));
        metrics.appendChild(row);
    });
    
    const prob = create('div', 'failure-probability');
    prob.append(
        create('div', 'probability-label', 'Failure Probability'),
        create('div', 'probability-value', `${Math.round(eq.failure_probability * 100)}%`)
    );
    const bar = create('div', 'probability-bar');
    const fill = create('div', 'probability-fill');
    fill.style.width = `${eq.failure_probability * 100}%`;
    bar.appendChild(fill);
    prob.appendChild(bar);
    
    const rec = create('div', 'recommendation');
    rec.append(create('div', 'recommendation-title', 'Recommendation:'), create('div', 'recommendation-text', RISK_CONFIG.recommendations[risk]));
    
    card.append(header, metrics, prob, rec);
    card.onclick = () => fillForm(eq);
    return card;
}

function showResult(result) {
    const container = $('#predictionResult') || (() => {
        const c = create('div');
        c.id = 'predictionResult';
        $('.equipment-section').appendChild(c);
        return c;
    })();
    
    const risk = getRisk(result.failure_probability);
    const color = RISK_CONFIG.colors[risk];
    
    container.replaceChildren((() => {
        const div = create('div', 'prediction-result');
        div.style.cssText = `background: white; padding: 25px; border-radius: 15px; box-shadow: 0 8px 32px rgba(0,0,0,0.1); margin-top: 20px; border-left: 5px solid ${color};`;
        
        const title = create('h3', '', 'Prediction Results');
        title.style.cssText = 'color: #333; margin-bottom: 15px;';
        
        const grid = create('div');
        grid.style.cssText = 'display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;';
        
        [
            [`${Math.round(result.failure_probability * 100)}%`, 'Failure Risk', color],
            [risk, 'Risk Level', color],
            [`${result.confidence ? Math.round(result.confidence * 100) : 95}%`, 'Confidence', '#4a69bd']
        ].forEach(([value, label, col]) => {
            const metric = create('div');
            metric.style.textAlign = 'center';
            metric.append(
                (() => { const v = create('div', '', value); v.style.cssText = `font-size: 2em; font-weight: bold; color: ${col};`; return v; })(),
                (() => { const l = create('div', '', label); l.style.color = '#666'; return l; })()
            );
            grid.appendChild(metric);
        });
        
        const rec = create('div');
        rec.style.cssText = 'margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px;';
        rec.append(create('strong', '', 'Recommendation: '), create('span', '', RISK_CONFIG.recommendations[risk]));
        
        div.append(title, grid, rec);
        return div;
    })());
}

function getRisk(prob) {
    return Object.keys(RISK_CONFIG.levels).find(level => {
        const [min, max] = RISK_CONFIG.levels[level];
        return prob >= min && prob < max;
    }) || 'CRITICAL';
}

function getType(id) {
    const types = { PUMP: 'Centrifugal Pump', MOTOR: 'Electric Motor', COMP: 'Compressor', TURB: 'Turbine' };
    return Object.keys(types).find(key => id.includes(key)) ? types[Object.keys(types).find(key => id.includes(key))] : 'Industrial Equipment';
}

function fillForm(eq) {
    const form = $('#predictionForm');
    const fields = { equipment_id: eq.equipment_id, vibration: eq.vibration_rms, temperature: eq.temperature_bearing, pressure: eq.pressure_oil, rpm: eq.rpm, oil_quality: eq.oil_quality, power: eq.power_consumption };
    Object.entries(fields).forEach(([name, value]) => {
        const input = form.querySelector(`[name="${name}"]`);
        if (input) input.value = value || '';
    });
}

function toggleLoading(show) {
    $('.loading').style.display = show ? 'block' : 'none';
    const btn = $('.predict-btn');
    btn.disabled = show;
    btn.textContent = show ? 'Analyzing...' : '🔮 Predict Failure Risk';
}

function showError(msg) {
    const container = (() => {
        let c = $('#errorContainer');
        if (!c) {
            c = create('div');
            c.id = 'errorContainer';
            $('.equipment-section').appendChild(c);
        }
        return c;
    })();
    
    const error = create('div', '', msg);
    error.style.cssText = 'background: #fdebea; color: #e74c3c; padding: 15px; border-radius: 8px; border: 1px solid #f5c6cb; margin-top: 15px; text-align: center;';
    container.replaceChildren(error);
    setTimeout(() => container.replaceChildren(), 5000);
}

function animateStats() {
    $$('.stat-number').forEach(el => {
        const target = el.textContent;
        const num = parseFloat(target.match(/[\d.]+/)?.[0] || 0);
        const suffix = target.replace(/[\d.]/g, '');
        let current = 0;
        const increment = num / 30;
        
        const timer = setInterval(() => {
            current += increment;
            if (current >= num) {
                el.textContent = target;
                clearInterval(timer);
            } else {
                el.textContent = (suffix.includes('%') ? Math.floor(current) : current.toFixed(1)) + suffix;
            }
        }, 50);
    });
}

function setupCardClicks() {
    document.onclick = e => {
        const card = e.target.closest('.equipment-card');
        if (card) {
            card.style.transform = 'scale(0.98)';
            setTimeout(() => card.style.transform = '', 150);
        }
    };
}