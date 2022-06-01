#   matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import numpy as np
import pandas
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
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)
plt.rc('axes', labelsize=20)

full_names = ['LIFT/GPT-J', 'Two-stage LIFT/GPT-J', '']
c = ['blue', 'green', 'black', 'olive']
model_color=['#1b998b', '#3185fc', '#ffba08', '#997b66']
# ,'knn':'#3185fc','nn':'#5d2e8c','svm':'#997b66','rf':'#ff9b85','xgboost':'#ff7b9c'}
# m = ['s', 'o', '^', 'x', '+', 'o', 'x']
m = ['.', '.']

# a = [0, 1, 2, 3, 4, 5, 6, 7]
# x = [1, 2, 5, 10, 20, 50, 100, 200]  # the label locations


data_regression = {0:
[[0.10506001495333138, 0.09804173432224533, 0.10199889848794548],
[0.17704899031834903, 0.1925020922010446, 0.16748499387976368],
 [0.2956480522595278, 0.3215939974325852, 0.2975796483670988],
 [0.537864626471848, 0.5576800893904527, 0.6592448791478737],
 [0.890254662473231, 0.8331797037031325, 0.9069316664463922],
 [1.5537964382081677, 1.4666238308163515, 1.4165365253942714],
 [1.5932395016196454, 1.4964751212583527, 1.4964751212583527],
 [1.943976178810879, 1.849568689496792, 2.12757065959928]],
1:
[[0.11917805304045445, 0.12381201041483246, 0.12490990932462678],
[0.18289619945763452, 0.18546313184778968, 0.17242861283740668],
[0.2512741546580323, 0.2631494783534408, 0.2746706317881707],
[0.6120155145784806, 0.5435133290614442, 0.5590784347476355],
[0.8061939607944861, 0.8180120903250612, 0.7853001642709188],
[1.2836420309823393, 1.3065900592163369, 1.2852619706684878],
[1.310923931023374, 1.370633451762684, 1.2702043584289908],
[1.7866004385514351, 1.6819436193514825, 1.864006579076018]]
}

def extract_data(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
    d = []
    for l in lines:
        p = l.strip().split(' ')[1:]
        d.append([float(x) for x in p])
    return d




nrows, ncols = 2, 3

fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12,0.5 + 3 * nrows))
odcs = [25.0, 11.0, 0, 50.0, 50.0, 50.0]
dids = [2, 40496, 0, 3, 4, 6]
titles = {2: 'Blobs', 3: 'Circles', 4: 'Moon', 6: 'Two Circles'}
for r in range(nrows):
    for c in range(ncols):
        ax = axes[r, c]
        j = r * ncols + c
        did = dids[j]
        # regression 
        if r == 0 and c == 2: # reg
            title = r'$y = 0.2x_1 + 0.4x_2$'
            x = [1, 2, '', 10, '', 50, '', 200]
            baseline = np.asarray([1.3731939195975151, 1.5361246509509126, 1.3076916367583749]).mean().item()
            data = data_regression
            ax.set_ylabel('RMSE',size=30) # RMSE
            ax.set_ylim([0, 2.2])
        else:
            x = [10, '', 50, '', 200, '', 1000]
            ax.set_ylabel('Accuracy',size=30) # RMSE
            ax.set_ylim([0, 100])
            if did < 10:
                title = titles[did]
            else:
                title = f'OpenML-{did}'
            ft_name = f'ift/inter_ft_clf_{did}_ft.txt'
            ift_name = f'ift/inter_ft_clf_{did}_ift.txt'
            bl_name = f'ift/inter_ft_clf_{did}_bl.txt'
            data = {0: extract_data(ft_name), 1: extract_data(ift_name)}
            baseline = np.asanyarray(extract_data(bl_name)).mean().item()
            random_score = odcs[j]
                        
        a = np.arange(len(x))
        for i in range(2):
            if r == 0 and c == 2: 
                d = data[i][::-1]
            else:
                d = data[i]
            y = np.asarray(d) #can change to max, mean, along axis=0
            mean = y.mean(axis=1)
            error = y.std(axis=1)
            # ax.plot(a, mean, marker=m[i], label=full_names[i], color=c[i])
            ax.errorbar(a, mean, yerr=error, label=full_names[i], color=model_color[i])
        ax.plot(a, [random_score]* len(a), label='ODC', color=model_color[3], marker='.')
        ax.plot(a, [baseline]* len(a), label='Pretext only', color=model_color[2])
        if r== nrows-1 and c == 1:
            ax.set_xlabel('Number of training samples',size=30)
        
        ax.set_title(title,size=30)
        ax.set_xticks(a)
        ax.set_xticklabels(x)
        ax.grid(False)


handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, prop={'size': 20})
font = font_manager.FontProperties(variant='small-caps')

fig.tight_layout()
width =4.487
height = 1.618

# width = 6
# height = 4
# fig.set_size_inches(width, height)
plt.show()
fig.savefig(f'clf_ift_full.pdf', bbox_inches='tight')
