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
plt.rc('axes', labelsize=28)
plt.rc('axes', linewidth=1)
mpl.rcParams['patch.linewidth']=5 #width of the boundary of legend

resolution = 100

# fig = plt.figure(figsize=plt.figaspect(0.5))
fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True, sharey=False,figsize=(12,9))

save_dir='./'
df = pd.read_csv("./decision_boundary/nn_gen_training_data.csv")
# ax = grid_df.plot.scatter(x = 0, y = 1, c = 'white', marker=".") # background
df.plot.scatter(x = '0', y = '1', c = 'y', cmap = 'tab10', colorbar = False, ax = ax,s=120)
ax.set_xlabel(r"$X_0$")
ax.set_ylabel(r"$X_1$")
# ax.set_title('Binary Data Points Used To Train a Neural Network Data Generator',fontdict={'fontsize': 26, 'fontweight': 'medium'})
# x0 = df["0"].values.reshape((resolution, resolution))
# x1 = df["1"].values.reshape((resolution, resolution))
# z = df['y'].values.reshape(x0.shape)
# ax.contourf(x0, x1,z ,cmap ='ocean')

fig.savefig(os.path.join(save_dir, 'nn_training_samples.pdf'),)