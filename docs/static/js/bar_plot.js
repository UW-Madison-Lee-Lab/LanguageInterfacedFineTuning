const ctx = document.getElementById('liicl').getContext('2d');
const liicl = new Chart(ctx, {
    type: 'bar',
    data: {
    labels: ['Customers', 'TAE', 'Vehicle', 'LED', 'Hamster', 'Breast'],
    datasets: [
        {
        label: 'LIICL/Subset',
        data: [60.61, 37.64, 28.82, 8.00, 57.78, 62.07],
        backgroundColor: 'rgba(6,56,120,0.8)',
        borderColor: 'rgba(6,56,120,1)',
        borderWidth: 1
        },
        {
        label: 'LIFT/Subset',
        data: [63.26, 33.33, 23.73, 11.33, 53.33, 70.69],
        backgroundColor: 'rgba(34,113,181,0.8)',
        borderColor: 'rgba(34,113,181,1)',
        borderWidth: 1
        },
        {
        label: 'LIFT/Full-data',
        data: [84.85, 65.59, 70.20, 69.33, 53.33, 71.26],
        backgroundColor: 'rgba(158,202,225,0.8)',
        borderColor: 'rgba(158,202,225,1)',
        borderWidth: 1
        }
    ]
    },
    options: {
    responsive: true,
    plugins: {
        tooltip: {
        mode: 'index',
        intersect: false,
        },
        legend: {
        position: 'top',
        },
        title: {
        display: true,
        text: 'Comparison of Accuracies Between ICL and Fine-Tuning with LIFT on OpenML Datasets'
        }
    },
    scales: {
        x: {
        stacked: false,
        title: {
            display: true,
            text: 'Datasets'
        }
        },
        y: {
        beginAtZero: true,
        title: {
            display: true,
            text: 'Accuracy (%)'
        }
        }
    }
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const ctx = document.getElementById('modelComparisonChart').getContext('2d');
    
    const data = {
        labels: ['Majority Classifier', 'LIFT/GPT-3', 'LIFT/GPT-J', 'LIFT/Rand-GPT-J', 'LIFT/Gibberish', 'LIFT/CodeGen', 'LIFT/CodeParrot'],
        datasets: [{
            data: [37.73, 86.97, 85.47, 31.61, 80.65, 54.12, 50.65],
            backgroundColor: [
                'rgba(6,56,120,0.8)',      // Darkest blue (specified)
                'rgba(34,113,181,0.8)',    // Darker blue (specified) 
                'rgba(66,146,198,0.8)',    // Dark blue
                'rgba(107,174,214,0.8)',   // Medium blue
                'rgba(158,202,225,0.8)',   // Light blue (specified)
                'rgba(198,219,239,0.8)',   // Lighter blue
                'rgba(222,235,247,0.8)'    // Lightest blue
            ],
            borderColor: [
                'rgb(6,56,120)',      // Darkest blue
                'rgb(34,113,181)',    // Darker blue
                'rgb(66,146,198)',    // Dark blue  
                'rgb(107,174,214)',   // Medium blue
                'rgb(158,202,225)',   // Light blue
                'rgb(198,219,239)',   // Lighter blue
                'rgb(222,235,247)'    // Lightest blue
            ],
            borderWidth: 1
        }]
    };

    new Chart(ctx, {
        type: 'bar',
        data: data,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                },
                title: {
                    display: true,
                    text: 'Model Performance Comparison',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    }
                }
            }
        }
    });
});


// ... existing code ...

document.addEventListener('DOMContentLoaded', function() {
    const ctxNew = document.getElementById('context').getContext('2d');
    
    const contextData = {
        labels: ['CMC (23)', 'TAE (48)', 'Vehicle (54)'],
        datasets: [
            {
                label: 'Majority Classifier',
                data: [42.71, 35.48, 25.88],
                backgroundColor: 'rgba(6,56,120,0.8)',
                borderColor: 'rgba(6,56,120,1)',
                borderWidth: 1
            },
            {
                label: 'W/o Names',
                data: [57.74, 65.59, 70.20],
                backgroundColor: 'rgba(34,113,181,0.8)',
                borderColor: 'rgba(34,113,181,1)',
                borderWidth: 1
            },
            {
                label: 'Shuffled-Names',
                data: [56.27, 60.22, 70.20],
                backgroundColor: 'rgba(107,174,214,0.8)',
                borderColor: 'rgba(107,174,214,1)',
                borderWidth: 1
            },
            {
                label: 'Correct-Names',
                data: [57.40, 69.89, 75.29],
                backgroundColor: 'rgba(198,219,239,0.8)',
                borderColor: 'rgba(198,219,239,1)',
                borderWidth: 1
            },
        ]
    };

    new Chart(ctxNew, {
        type: 'bar',
        data: contextData,
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top'
                },
                title: {
                    display: true,
                    text: 'Majority Classifier vs LIFT Performance Comparison',
                    font: {
                        size: 16
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    }
                }
            }
        }
    });
});


