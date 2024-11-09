// PGD Attack Charts
function createPGDCharts() {
    const epsilons = [0, 0.01, 0.1, 0.3];
    const models = ['LeNet-5', 'MLP', 'LIFT/GPT-3'];
    
    // Data for each attack type
    const gaussianData = {
        'LeNet-5': [99.22, 99.25, 99.20, 98.01],
        'MLP': [98.09, 98.05, 97.70, 87.69],
        'LIFT/GPT-3': [98.15, 98.28, 88.38, 54.80]
    };
    
    const signedConstData = {
        'LeNet-5': [99.22, 99.26, 99.06, 79.80],
        'MLP': [98.09, 98.08, 97.39, 74.20],
        'LIFT/GPT-3': [98.15, 88.05, 68.80, 29.68]
    };
    
    const leNetAttackData = {
        'LeNet-5': [99.22, 97.27, 26.80, 0.00],
        'MLP': [98.09, 97.77, 93.99, 36.62],
        'LIFT/GPT-3': [98.15, 44.88, 33.66, 20.31]
    };
    
    const mlpAttackData = {
        'LeNet-5': [99.22, 99.15, 96.98, 41.51],
        'MLP': [98.09, 96.89, 23.12, 0.00],
        'LIFT/GPT-3': [98.15, 44.46, 23.62, 20.29]
    };

    const chartConfigs = [
        {
            id: 'pgdChart1',
            title: 'Random Noise (Gaussian)',
            data: gaussianData
        },
        {
            id: 'pgdChart2',
            title: 'Random Noise (Signed Const.)',
            data: signedConstData
        },
        {
            id: 'pgdChart3',
            title: 'PGD Attack on LeNet-5',
            data: leNetAttackData
        },
        {
            id: 'pgdChart4',
            title: 'PGD Attack on MLP',
            data: mlpAttackData
        }
    ];

    chartConfigs.forEach(config => {
        const ctx = document.getElementById(config.id).getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: epsilons.map(e => `ε = ${e}`),
                datasets: models.map(model => ({
                    label: model,
                    data: config.data[model],
                    borderColor: model === 'LeNet-5' ? '#ff6384' : 
                               model === 'MLP' ? '#36a2eb' : '#4bc0c0',
                    tension: 0.1,
                    fill: false
                }))
            },
            options: {
                responsive: true,
                plugins: {
                    title: {
                        display: true,
                        text: config.title,
                        font: {
                            size: 16
                        }
                    },
                    legend: {
                        position: 'bottom'
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Accuracy (%)'
                        }
                    }
                }
            }
        });
    });
}

// Call the function when the document is ready
document.addEventListener('DOMContentLoaded', createPGDCharts);