// Only using tabular data
const data = [
    { dataset: 'Customers', p_c: '8/2', MCC: 68.18, LogReg: [87.12, 0.54], DT: [85.98, 0.53], RBF_SVM: [86.36, 0.00], XG: [85.23, 0.00], LIFT_GPT_J: [85.23, 1.61], LIFT_GPT_3: [84.85, 1.42] },
    { dataset: 'Pollution', p_c: '15/2', MCC: 50.00, LogReg: [58.33, 11.79], DT: [77.78, 3.93], RBF_SVM: [58.33, 6.81], XG: [63.89, 7.86], LIFT_GPT_J: [63.89, 3.93], LIFT_GPT_3: [63.89, 7.86] },
    { dataset: 'Spambase', p_c: '57/2', MCC: 60.59, LogReg: [93.27, 0.00], DT: [90.70, 0.14], RBF_SVM: [93.70, 0.00], XG: [95.87, 0.00], LIFT_GPT_J: [94.03, 0.54], LIFT_GPT_3: [94.90, 0.36] },
    { dataset: 'Hill-Valley', p_c: '100/2', MCC: 49.79, LogReg: [77.78, 0.00], DT: [56.38, 0.89], RBF_SVM: [68.72, 0.00], XG: [59.26, 0.00], LIFT_GPT_J: [100.00, 0.20], LIFT_GPT_3: [99.73, 0.19] },
    { dataset: 'IRIS', p_c: '4/3', MCC: 33.33, LogReg: [96.67, 0.00], DT: [97.77, 3.85], RBF_SVM: [100.00, 0.00], XG: [100.00, 0.00], LIFT_GPT_J: [96.67, 0.00], LIFT_GPT_3: [97.00, 0.00] },
    { dataset: 'TAE', p_c: '5/3', MCC: 35.48, LogReg: [45.16, 4.56], DT: [65.59, 5.49], RBF_SVM: [53.76, 6.63], XG: [66.67, 8.05], LIFT_GPT_J: [61.29, 6.97], LIFT_GPT_3: [65.59, 6.63] },
    { dataset: 'CMC', p_c: '9/3', MCC: 42.71, LogReg: [49.49, 0.83], DT: [56.72, 0.32], RBF_SVM: [56.50, 0.97], XG: [52.43, 0.42], LIFT_GPT_J: [49.83, 0.28], LIFT_GPT_3: [57.74, 0.89] },
    { dataset: 'Wine', p_c: '13/3', MCC: 38.89, LogReg: [100.00, 0.00], DT: [93.52, 2.62], RBF_SVM: [100.00, 0.00], XG: [97.22, 0.00], LIFT_GPT_J: [93.52, 1.31], LIFT_GPT_3: [92.59, 1.31] },
    { dataset: 'Vehicle', p_c: '18/4', MCC: 25.88, LogReg: [80.39, 1.00], DT: [63.92, 2.37], RBF_SVM: [81.18, 0.48], XG: [73.14, 0.28], LIFT_GPT_J: [64.31, 2.37], LIFT_GPT_3: [70.20, 2.73] },
    { dataset: 'LED', p_c: '7/10', MCC: 11.00, LogReg: [68.67, 0.94], DT: [66.33, 2.87], RBF_SVM: [68.00, 0.82], XG: [66.00, 0.82], LIFT_GPT_J: [65.33, 0.47], LIFT_GPT_3: [69.33, 2.05] },
    { dataset: 'OPT', p_c: '64/10', MCC: 10.14, LogReg: [96.53, 0.22], DT: [89.80, 1.09], RBF_SVM: [97.95, 0.00], XG: [97.48, 0.17], LIFT_GPT_J: [98.22, 0.11], LIFT_GPT_3: [98.99, 0.30] },
    { dataset: 'Mfeat', p_c: '216/10', MCC: 10.00, LogReg: [97.67, 0.12], DT: [87.67, 1.05], RBF_SVM: [98.83, 0.24], XG: [96.75, 0.00], LIFT_GPT_J: [94.17, 1.75], LIFT_GPT_3: [93.08, 0.24] },
    { dataset: 'Margin', p_c: '64/100', MCC: 0.94, LogReg: [81.35, 0.15], DT: [43.86, 1.21], RBF_SVM: [81.98, 0.30], XG: [70.21, 0.29], LIFT_GPT_J: [50.23, 1.33], LIFT_GPT_3: [59.37, 0.92] },
    { dataset: 'Texture', p_c: '64/100', MCC: 0.94, LogReg: [81.67, 0.97], DT: [46.88, 1.93], RBF_SVM: [83.44, 0.89], XG: [70.73, 1.41], LIFT_GPT_J: [50.32, 2.18], LIFT_GPT_3: [67.50, 1.42] }
];

function getColor(value) {
    // Returns a blue color scale from light to dark based on value
    const minValue = 0;
    const maxValue = 100;
    const normalizedValue = (value - minValue) / (maxValue - minValue);
    
    // Using the Blues color scheme
    const colors = [
        [247, 251, 255],
        [222, 235, 247],
        [198, 219, 239],
        [158, 202, 225],
        [107, 174, 214],
        [66, 146, 198],
        [33, 113, 181],
        [8, 69, 148]
    ];
    
    const index = Math.floor(normalizedValue * (colors.length - 1));
    const color = colors[Math.min(index, colors.length - 1)];
    return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function createTooltip(cell, dataset, method, value, std) {
    const tooltip = document.createElement('div');
    tooltip.className = 'tooltip';
    tooltip.innerHTML = `
        Dataset: ${dataset}<br>
        Method: ${method}<br>
        Accuracy: ${value.toFixed(2)}% ±${std.toFixed(2)}
    `;
    cell.appendChild(tooltip);

    cell.addEventListener('mouseover', () => {
        tooltip.style.display = 'block';
    });

    cell.addEventListener('mouseout', () => {
        tooltip.style.display = 'none';
    });
}

function createHeatmap() {
    const container = document.getElementById('heatmapContainer');
    const table = document.createElement('table');
    table.className = 'heatmap';

    // Create header row
    const methods = ['Logistic Regression', 'Decision Tree', 'RBF-SVM', 'XGBoost', 'LIFT/GPT-J', 'LIFT/GPT-3'];
    const headerRow = document.createElement('tr');
    headerRow.innerHTML = '<th>Dataset</th>' + 
        methods.map(method => `<th>${method}</th>`).join('');
    table.appendChild(headerRow);

    // Create data rows
    data.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${row.dataset}</td>`;
        
        methods.forEach(method => {
            // Map display names back to keys for data lookup
            let lookupKey = method;
            if (method === 'Decision Tree') lookupKey = 'DT';
            if (method === 'XGBoost') lookupKey = 'XG';
            if (method === 'Majority Classifier') lookupKey = 'MCC';
            if (method === 'Logistic Regression') lookupKey = 'LogReg';
            
            const methodKey = lookupKey.replace('-', '_').replace('/', '_');
            const [value, std] = row[methodKey];
            const td = document.createElement('td');
            td.style.backgroundColor = getColor(value);
            td.style.color = 'black'; // Add white text for darker cells
            td.textContent = value.toFixed(1);
            createTooltip(td, row.dataset, method, value, std);
            tr.appendChild(td);
        });

        table.appendChild(tr);
    });

    container.appendChild(table);
}

// Create heatmap when page loads
document.addEventListener('DOMContentLoaded', createHeatmap);

// Image classification data
const imageData = {
  labels: ['MNIST', 'Permuted MNIST', 'Fashion MNIST', 'Permuted F-MNIST'],
  models: ['LogReg', 'DT', 'RBF-SVM', 'XG', 'LIFT/GPT-J', 'LIFT/GPT-3'],
  values: [
    [91.95, 87.42, 97.70, 97.69, 97.01, 98.15],
    [92.58, 87.87, 98.06, 97.62, 95.80, 96.25],
    [85.59, 80.52, 90.59, 90.19, 85.10, 90.18],
    [84.95, 79.91, 88.04, 89.93, 82.25, 88.92]
  ]
};

// Create image heatmap when page loads
document.addEventListener('DOMContentLoaded', () => {
    const container = document.getElementById('imageHeatmapContainer');
    if (container) {
        const table = document.createElement('table');
        table.className = 'heatmap';

        // Create header row
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = '<th>Dataset</th>' + 
            imageData.models.map(method => `<th>${method}</th>`).join('');
        table.appendChild(headerRow);

        // Create data rows
        imageData.labels.forEach((label, i) => {
            const tr = document.createElement('tr');
            tr.innerHTML = `<td>${label}</td>`;
            
            imageData.values[i].forEach((value, j) => {
                const td = document.createElement('td');
                td.style.backgroundColor = getColor(value);
                td.style.color = 'black';
                td.textContent = value.toFixed(1);
                createTooltip(td, label, imageData.models[j], value, 0);
                tr.appendChild(td);
            });

            table.appendChild(tr);
        });

        container.appendChild(table);
    }
});
