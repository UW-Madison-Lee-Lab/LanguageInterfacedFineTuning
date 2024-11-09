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
                datasets: models.map((model, index) => ({
                    label: model,
                    data: config.data[model],
                    borderColor: model === 'LeNet-5' ? 'rgb(158,202,225,.8)' : 
                               model === 'MLP' ? 'rgb(66,146,198,.8)' : 'rgb(6,56,120,.8)',
                    tension: 0.1,
                    fill: false,
                    borderDash: model === 'LeNet-5' ? [2, 2] : 
                              model === 'MLP' ? [5, 5] : [],
                    borderWidth: model === 'LeNet-5' ? 2 : 
                               model === 'MLP' ? 2 : 3
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
                        position: 'bottom',
                        labels: {
                            generateLabels: function(chart) {
                                const datasets = chart.data.datasets;
                                return datasets.map((dataset, i) => ({
                                    text: dataset.label,
                                    fillStyle: 'transparent',
                                    strokeStyle: dataset.borderColor,
                                    lineWidth: dataset.borderWidth,
                                    lineDash: dataset.borderDash,
                                    hidden: !chart.isDatasetVisible(i),
                                    index: i
                                }));
                            }
                        }
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

function createDataAugmentationComparisonChart() {
    const epsilons = {
        gaussian: [0, 0.01, 0.1],
        signed: [0, 0.01, 0.1]
    };
    
    const models = [
        'ε=0 (no data augmentation)',
        'ε=0.05',
        'ε=0.1'
    ];
    
    const gaussianData = {
        'ε=0 (no data augmentation)': [96.88, 95.27, 56.14],
        'ε=0.05': [93.80, 94.39, 93.40],
        'ε=0.1': [93.78, 94.31, 94.98]
    };
    
    const signedData = {
        'ε=0 (no data augmentation)': [96.88, 55.83, 27.73],
        'ε=0.05': [93.80, 93.46, 61.24],
        'ε=0.1': [93.78, 94.12, 75.25]
    };
    
    const chartConfigs = [
        {
            id: 'dataAugmentationChart1',
            title: 'Gaussian Noise',
            data: gaussianData,
            epsilons: epsilons.gaussian
        },
        {
            id: 'dataAugmentationChart2',
            title: 'Signed Constant Noise',
            data: signedData,
            epsilons: epsilons.signed
        }
    ];

    chartConfigs.forEach(config => {
        const ctx = document.getElementById(config.id).getContext('2d');
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: config.epsilons.map(e => `ε = ${e}`),
                datasets: models.map((model, index) => ({
                    label: model,
                    data: config.data[model],
                    borderColor: index === 0 ? 'rgb(158,202,225,.8)' :
                                index === 1 ? 'rgb(66,146,198,.8)' : 'rgb(6,56,120,.8)',
                    tension: 0.1,
                    fill: false,
                    borderDash: index === 0 ? [2, 2] :
                               index === 1 ? [5, 5] : [],
                    borderWidth: index === 0 ? 2 :
                                index === 1 ? 2 : 3
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
                        position: 'bottom',
                        labels: {
                            generateLabels: function(chart) {
                                const datasets = chart.data.datasets;
                                return datasets.map((dataset, i) => ({
                                    text: dataset.label,
                                    fillStyle: 'transparent',
                                    strokeStyle: dataset.borderColor,
                                    lineWidth: dataset.borderWidth,
                                    lineDash: dataset.borderDash,
                                    hidden: !chart.isDatasetVisible(i),
                                    index: i
                                }));
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        min: 20,
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

// Update the DOMContentLoaded event listener
document.addEventListener('DOMContentLoaded', () => {
    createPGDCharts();
    createDataAugmentationComparisonChart();
});