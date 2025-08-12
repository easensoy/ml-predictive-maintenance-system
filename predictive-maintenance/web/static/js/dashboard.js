class Dashboard {
    constructor() {
        this.data = [];
        this.charts = {};
        this.wsClient = null;
        this.init();
    }

    async init() {
        this.initWebSocket();
        await this.loadData();
    }

    initWebSocket() {
        this.wsClient = new WebSocketClient();
        
        this.wsClient.on('liveData', (data) => {
            if (data && !data.error) {
                this.data = Array.isArray(data) ? data : [data];
                this.render();
            }
        });

        this.wsClient.on('connected', () => {
            this.wsClient.startPeriodicUpdates(5000);
        });

        this.wsClient.on('systemAlert', (alert) => {
            this.showAlert(alert);
        });
    }

    showAlert(alert) {
        const alertDiv = document.createElement('div');
        alertDiv.className = 'fixed top-16 right-4 bg-red-500 text-white p-4 rounded-lg shadow-lg z-50';
        alertDiv.textContent = alert.message || 'System Alert';
        document.body.appendChild(alertDiv);
        
        setTimeout(() => alertDiv.remove(), 5000);
    }

    async loadData() {
        try {
            const res = await fetch('/api/equipment/demo');
            this.data = await res.json();
            this.render();
        } catch (e) {
            this.renderError();
        }
    }

    render() {
        this.renderCards();
        this.renderCharts();
    }

    renderCards() {
        const container = document.getElementById('equipmentOverview');
        container.replaceChildren(...this.data.map(eq => this.createCard(eq)));
    }

    createCard(eq) {
        const risk = eq.risk_level || 'LOW';
        const status = this.getStatus(risk);
        
        return this.createElement('div', {
            className: `equipment-card ${status}`,
            children: [
                this.createElement('div', {
                    className: 'equipment-header',
                    children: [
                        this.createElement('div', {
                            children: [
                                this.createElement('div', { className: 'equipment-name', textContent: eq.equipment_id }),
                                this.createElement('div', { className: 'equipment-type', textContent: this.getType(eq.equipment_id) })
                            ]
                        }),
                        this.createElement('div', { className: `status-badge status-${status}`, textContent: risk })
                    ]
                }),
                this.createElement('div', {
                    className: 'metrics-grid',
                    children: [
                        { value: (eq.vibration_rms || 0).toFixed(1), label: 'Vibration' },
                        { value: `${Math.round(eq.temperature_bearing || 0)}°`, label: 'Temp' },
                        { value: (eq.pressure_oil || 0).toFixed(1), label: 'Pressure' },
                        { value: `${Math.round((eq.failure_probability || 0) * 100)}%`, label: 'Risk' }
                    ].map(m => this.createElement('div', {
                        className: 'metric-item',
                        children: [
                            this.createElement('div', { className: 'metric-value', textContent: m.value }),
                            this.createElement('div', { className: 'metric-label', textContent: m.label })
                        ]
                    }))
                }),
                this.createElement('div', {
                    className: 'risk-info',
                    children: [
                        this.createElement('span', { textContent: `Status: ${this.getStatusText(risk)}` }),
                        this.createElement('span', { textContent: `Risk: ${Math.round((eq.failure_probability || 0) * 100)}%` })
                    ]
                })
            ]
        });
    }

    createElement(tag, { className, textContent, children, ...props } = {}) {
        const el = document.createElement(tag);
        if (className) el.className = className;
        if (textContent) el.textContent = textContent;
        if (children) el.append(...(Array.isArray(children) ? children : [children]));
        Object.assign(el, props);
        return el;
    }

    renderCharts() {
        this.renderRiskChart();
        this.renderHealthChart();
    }

    renderRiskChart() {
        const counts = this.data.reduce((acc, eq) => {
            const risk = eq.risk_level || 'LOW';
            acc[risk] = (acc[risk] || 0) + 1;
            return acc;
        }, {});

        this.destroyChart('risk');
        this.charts.risk = new Chart(document.getElementById('riskChart'), {
            type: 'doughnut',
            data: {
                labels: ['Low Risk', 'Medium Risk', 'High Risk', 'Critical'],
                datasets: [{
                    data: [counts.LOW || 0, counts.MEDIUM || 0, counts.HIGH || 0, counts.CRITICAL || 0],
                    backgroundColor: ['#27ae60', '#f39c12', '#e67e22', '#e74c3c'],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { padding: 20, usePointStyle: true } } }
            }
        });
    }

    renderHealthChart() {
        const hours = Array.from({ length: 24 }, (_, i) => `${23 - i}h ago`);
        const base = this.data[0] || {};
        const datasets = [
            { label: 'Vibration (RMS)', data: this.generateTrend(base.vibration_rms || 1.0, 0.4), borderColor: '#e74c3c', yAxisID: 'y' },
            { label: 'Temperature (°C)', data: this.generateTrend(base.temperature_bearing || 70, 8), borderColor: '#f39c12', yAxisID: 'y1' },
            { label: 'Pressure (PSI)', data: this.generateTrend(base.pressure_oil || 20, 3), borderColor: '#3498db', yAxisID: 'y2' }
        ].map(d => ({ ...d, backgroundColor: d.borderColor.replace(')', ', 0.1)').replace('#', 'rgba('), tension: 0.4 }));

        this.destroyChart('health');
        this.charts.health = new Chart(document.getElementById('healthChart'), {
            type: 'line',
            data: { labels: hours, datasets },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: { mode: 'index', intersect: false },
                scales: {
                    x: { display: true, title: { display: true, text: 'Time' } },
                    y: { type: 'linear', display: true, position: 'left', title: { display: true, text: 'Vibration' } },
                    y1: { type: 'linear', display: false, position: 'right' },
                    y2: { type: 'linear', display: false, position: 'right' }
                },
                plugins: { legend: { position: 'bottom' } }
            }
        });
    }

    generateTrend(base, variance) {
        return Array.from({ length: 24 }, () => base + (Math.random() - 0.5) * variance);
    }

    destroyChart(name) {
        if (this.charts[name]) {
            this.charts[name].destroy();
            delete this.charts[name];
        }
    }

    renderError() {
        const container = document.getElementById('equipmentOverview');
        container.replaceChildren(
            this.createElement('div', {
                style: 'grid-column: 1 / -1; background: #fff3cd; color: #856404; padding: 20px; border-radius: 8px; border: 1px solid #ffeaa7; text-align: center;',
                children: [
                    this.createElement('p', { textContent: 'Unable to load equipment data. Please check your connection and try again.' }),
                    this.createElement('button', {
                        textContent: 'Retry',
                        style: 'margin-top: 10px; padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;',
                        onclick: () => this.loadData()
                    })
                ]
            })
        );
    }

    getStatus(risk) {
        return { LOW: 'healthy', MEDIUM: 'warning', HIGH: 'warning', CRITICAL: 'critical' }[risk] || 'healthy';
    }

    getStatusText(risk) {
        return { LOW: 'Operational', MEDIUM: 'Monitor', HIGH: 'Attention Required', CRITICAL: 'Immediate Action' }[risk] || 'Unknown';
    }

    getType(id) {
        return {
            EQ_001: 'Turbine Generator',
            EQ_002: 'Air Compressor', 
            EQ_003: 'Hydraulic Pump',
            EQ_004: 'Cooling System',
            EQ_005: 'Conveyor Motor'
        }[id] || 'Industrial Equipment';
    }
}

document.addEventListener('DOMContentLoaded', () => new Dashboard());