import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d
import matplotlib as mpl
from PIL import Image


plt.rc('font', family='serif', serif='times new roman')
plt.rc('text', usetex=True)
mpl.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.rc('xtick', labelsize=26)
plt.rc('ytick', labelsize=26)
plt.rc('ytick', labelsize=26)
plt.rc('axes', labelsize=28)
plt.rc('axes', linewidth=1)
mpl.rcParams['patch.linewidth']=5 #width of the boundary of legend

fig = plt.figure(figsize=plt.figaspect(0.5))
width = 40
height = 13 #14
fig.set_size_inches(width, height) #exact size of the figure
fig.subplots_adjust(left=.05, bottom=.0, right=.97, top=1, hspace = 0.05, wspace = 0.2) #margin of the figure


# alg_names = ['True function', 'Log. Reg.', 'KNN', 'Decision Tree', 'MLP', 'Random Forest', 'XGBoost', 'GPT3']
alg_names = ['True Function', 'LogReg', 'KNN', 'DT', 'MLP', 'RBF-SVM', 'RF', 'XG','LIFT/GPT-J','LIFT/GPT-3']
num_cols = len(alg_names)
# algs = ['function', 'logreg', 'knn', 'decision_tree', 'nn', 'rf', 'xgboost', 'gpt']
algs = ['function', 'logreg', 'knn', 'tree', 'nn', 'svm','rf','xgboost','gptj','gpt3']

eps = [10, 80, 490]
num_rows = len(eps)
# labels = ['(a)', '(b)', '(c)', '(d)', '(e)']
labels = ['','','','','']

# color_1 = '#f7c592'
# color_2 = '#9bc1dd'
color_1 = '#a39faa'
color_2 = '#fbece1'
# color_1 = '#F9D5A7'
# color_2 = '#F9813A'
# color_1 = '#D6E5FA'
# color_2 = '#FFF9F9'
# color_1 = '#79B4B7'
# color_2 = '#FEFBF3'
# color_1 = '#005AB5'
# color_2 = '#DC3220'
# color_1 = '#ed665f'
# color_2 = '#00bfc4'
# color_1 = '#d5e8d4'
# color_2 = '#f8cecc'
# color_1 = '#01b4bc'
# color_2 = '#f6d51f'
# color_1 = '#5fa55a'
# color_2 = '#f6d51f'
# color_1 = '#01b4bc'
# color_2 = '#fa5457'
# color_1 = '#385b73'
# color_2 = '#f2c58e'


noise = 0

def get_suffix(name, noise=0):
    if name == 'function':
        return ''
    # if name in ['rf', 'xgboost'] or noise == 0.05:
    #     suffix = f'_corrupted_{noise}'
    # elif noise == 0.2:
    #     suffix = '_corrupted'
    # else:
    #     suffix = ''
    # return suffix
    else:
        return '_corrupted_0'

for i in range(num_rows):
    # GPT3
    for j in range(num_cols):
        ax = fig.add_subplot(num_rows, num_cols, i * num_cols + j +1)
        # suffix = get_suffix(algs[j], noise)
        suffix = ''
        if color_1 is None:
            fname = f"{algs[j]}_ep{eps[i]}{suffix}.png"
        else:
            # fname = f"new_colors/{algs[j]}_ep{eps[i]}{suffix}_{color_1}_{color_2}.png"
            
            fname = f"new_colors/{algs[j]}_ep{eps[i]}_corrupted_{noise}_{color_1}_{color_2}.png"
            print("fname",fname)
            
        img = Image.open(f"./plots/{fname}")
        # im1 = im.crop((left, top, right, bottom))
        w, h = img.size
        # l=82;r=65; t=60; b=54
        l=int(0.1*w);r=int(0.1*w); t=int(0.1*h); b=int(0.1*h)
        img = img.crop((l, t, w-r, h-b))
        ax.imshow(img)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0: # first row
            ax.set_title(alg_names[j], fontsize = 40, y = 1, pad=20)
        if j == 0: # first col
            ax.set_ylabel(labels[i], rotation = 0, labelpad=35, fontsize=40)

plt.tight_layout()
if color_1 is None:
    plt.savefig(f'clf_decision_boundary_{noise}.pdf')
else:
    plt.savefig(f'new_colors_clf_decision_boundary_corrupted_{noise}_{color_1}_{color_2}.pdf')