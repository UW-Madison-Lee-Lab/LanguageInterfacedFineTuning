#   matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas

# plt.rc('font', family='serif', serif='Times')
# plt.rc('font', variant='small-caps')
# plt.rc('text', usetex=True)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)
# plt.rc('axes', labelsize=10)

#   matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas
import matplotlib as mpl
# matplotlib.style.use('ggplot')
# plt.rc('font', family='serif', serif='Times')
# plt.rc('font', variant='small-caps')
plt.rc('text', usetex=True)


matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['#377eb8',  '#4daf4a','#ff7f00',
                  '#f781bf', '#a65628', '#984ea3',
                  '#999999', '#e41a1c', '#dede00']) 
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rc('lines', linewidth=3)
plt.rc('font', family='serif', serif='times new roman')
plt.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', labelsize=20)



# orders = ['KNN', 'Decision Tree', 'MLP', 'LogReg', 'LIGF/GPT-3', 'RF', 'XGBoost']

algs = ['logreg', 'knn', 'tree', 'nn', 'rf', 'xgboost', 'gptj','gpt3']
alg_order = [3, 0, 1, 2, 5, 6, 7, 4]

models={'logreg': ('#8fe388', 'LogReg'), 'gptj': ('red', 'LIFT/GPT-J'), 'gpt3': ('#ffba08', 'LIFT/GPT-3'), 'knn': ('#3185fc', 'KNN'), 'nn': ('#5d2e8c', 'MLP'), 'tree': ('#1b998b', 'DT'), 'rf':('#ff9b85', 'RF') ,'xgboost': ('#ff7b9c', 'XGBoost')}


# m = ['s', 'o', '^', 'x', '+', 'o', 'x']
m = ['s', '<','4','.','+','x','o','*']


data = {}
add = {}
data= {1: [
    [9.949999999999999956e-01, 9.925000000000000488e-01, 1.000000000000000000e+00, 9.949999999999999956e-01, 9.899999999999999911e-01],
    [9.825000000000000400e-01, 9.825000000000000400e-01, 9.975000000000000533e-01, 9.324999999999999956e-01, 9.350000000000000533e-01], 
    [9.675000000000000266e-01, 9.799999999999999822e-01, 9.825000000000000400e-01, 8.725000000000000533e-01, 9.499999999999999556e-01],
    [9.725000000000000311e-01, 9.575000000000000178e-01, 9.875000000000000444e-01, 7.750000000000000222e-01, 9.050000000000000266e-01],
    [9.449999999999999512e-01, 9.174999999999999822e-01, 9.475000000000000089e-01, 8.000000000000000444e-01, 8.425000000000000266e-01],
    [9.425000000000000044e-01, 9.300000000000000488e-01, 9.525000000000000133e-01, 8.024999999999999911e-01, 8.774999999999999467e-01], 
], 2: [[0.985 , 0.93  , 0.9975, 0.975 , 0.9725],
       [0.975 , 0.935 , 0.9825, 0.9325, 0.9475],
       [0.9625, 0.9425, 0.96  , 0.8925, 0.935 ],
       [0.94  , 0.8825, 0.935 , 0.8175, 0.8975],
       [0.9475, 0.8775, 0.905 , 0.77  , 0.865 ],
       [0.93  , 0.8675, 0.935 , 0.78  , 0.85  ],
       ], 
       3: [[0.9325, 0.815 , 0.9875, 0.99  , 0.955 ],
       [0.935 , 0.78  , 0.9725, 0.92  , 0.9175],
       [0.87  , 0.77  , 0.9375, 0.88  , 0.865 ],
       [0.9025, 0.75  , 0.8675, 0.7575, 0.795 ],
       [0.8675, 0.77  , 0.86  , 0.755 , 0.82  ],
       [0.895 , 0.725 , 0.8825, 0.7875, 0.785 ],
       ]}

add = {1: [[0.965 ,0.905 ,0.92 ,0.7925 ,0.77 ,0.7425],
    [0.98,0.96,0.9575,0.915,0.9175,0.9425],
    [0.9725,0.9600,0.9600,-1,0.9025,0.8800]], 
    2: [[0.97 ,0.8975,0.895,0.8,0.7625,0.775],
       [0.9925,0.99,0.9775,0.9525,0.9275,0.93],
       [0.9375,0.9200,0.9100,-1,0.8125,0.8000]],
    3: [[0.9325,0.925,0.915,0.8,0.775,0.7875],
       [0.89,0.875,0.875,0.79,0.82,0.825],
       [0.8300,0.7975,0.7850,-1,0.6825,0.6575]]}



def load_data(j, data, add):
    k = j + 1
    data = np.asarray(data[k]).T * 100
    data = np.delete(data, 3, 1)
    add = np.asarray(add[k]) * 100
    add = np.delete(add, 3, 1)
    data = np.concatenate([data, add], axis=0)
    return data

titles = ['Clean Data', 'Corrupted Label (5\%)', 'Corrupted Label (20\%)']

a = [0, 1, 2, 3, 4]
x = [10, 40, 80, 300, 490]  # the label locations

fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12,4))
for j in range(3):
    ax = axes[j]
    title = titles[j]
    d = load_data(j, data, add)
    d = d[alg_order, :]
    # reorder 
    # data
    for i in range(8):
        y = d[i, :] #can change to max, mean, along axis=0
        color, label = models[algs[i]]
        ax.plot(a, y, marker=m[i], label=label, color=color,linewidth=2,alpha=0.75)

    ax.set_ylim([60,100])
    ax.set_xticks(a)
    ax.set_xticklabels(x)
    # ax.set_xlabel('Epoch', size=20)
    # ax.set_xlim([0,300])
    ax.set_title(title, size=20)
    if j == 1:
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_xlabel('No. of Epochs Used for Training the (Neural Network) Generator', size=20)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, bbox_to_anchor=(1, 1.15), ncol=8, prop={'size': 12})  # (x, y)
    elif j == 0:
        ax.set_ylabel('Accuracy',size=20)

    
# ax.grid(True)
fig.tight_layout()
width =11
height = 3
fig.set_size_inches(width, height)

plt.show()
fig.savefig(f'clf_vary_difficulty.pdf', bbox_inches='tight')