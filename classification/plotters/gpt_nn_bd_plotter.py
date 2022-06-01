import pandas as pd
import argparse
import os
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Unsupervised Word Translation')
parser.add_argument("-l", "--color_1", default='#ffffff', type=str)
parser.add_argument("-o", "--color_2", default='#000000', type=str)
parser.add_argument("-b", "--bd_color", default='#000000', type=str)

args = parser.parse_args()

def plot_decision_boundry_custom_color(name, df, resolution=100, save_dir='results/plots/new_colors/',color_1='#000000',color_2='#ffffff',bd_color='#000000'):
    # ax = grid_df.plot.scatter(x = 0, y = 1, c = 'white', marker=".") # background
    # ax = df.plot.scatter(x = 0, y = 1, c = 'y', cmap = 'ocean', colorbar = False, marker=".")
    ax = plt.figure(figsize=(10,10))
    ax.set_facecolor('white')
    ax1 = plt.gca()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    # ax.set_xlabel('x0')
    # ax.set_ylabel('x1')
    # ax.set_title(name)
    x0 = df["0"].values.reshape((resolution, resolution))
    x1 = df["1"].values.reshape((resolution, resolution))
    z = df['y'].values.reshape(x0.shape)
    # plt.contourf(x_grid_1, x_grid_2, z,cmap ="ocean")
    cs = plt.contourf(x0, x1, z, levels=[0, 0.7],
    colors=[bd_color, '#ffffff'], extend='both')
    cs.cmap.set_over(color_1)
    cs.cmap.set_under(color_2)
    cs.changed()
    fig = ax.get_figure()
    # plt.title(name)
    plt.savefig(os.path.join(save_dir, name + '.png'),dpi=300)


noise = 0.2

epochs = [10, 80, 490]
# epochs = [10]
models = ['gpt3','gptj']
# models = ['gpt3']
for ep in epochs:
    for md in models:
        # try:
        print(md,ep)
        grid_df = pd.read_csv(f"{md}_grid_ep_{ep}_corrupted_{noise}.csv")
        grid_df = grid_df.iloc[: , 1:]
        plot_decision_boundry_custom_color(md + f'_ep{ep}_corrupted_{noise}_{args.color_1}_{args.color_2}', grid_df, resolution=300,
                            color_1=args.color_1,color_2=args.color_2,bd_color=args.bd_color,
                            save_dir='results/plots/new_colors/')
        # except:
        #     from IPython import embed; embed()
