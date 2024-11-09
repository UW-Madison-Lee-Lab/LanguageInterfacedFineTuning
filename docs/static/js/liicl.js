const ctx = document.getElementById('comparisonChart').getContext('2d');
const comparisonChart = new Chart(ctx, {
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