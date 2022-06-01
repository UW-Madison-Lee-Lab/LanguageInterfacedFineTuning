import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
from scipy.interpolate import griddata
import os
# matplotlib.style.use('seaborn')
# plt.rc('font', family='serif', serif='Times')
plt.rc('font', variant='small-caps')
plt.rc('text', usetex=False)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', labelsize=10)

def plot_decision_boundry_custom_color(name, df, resolution=100, save_dir='results/plots/new_colors/',color_1='#000000',color_2='#ffffff',bd_color='#000000',df_scatter=None,scatter=False):
    ax = plt.figure(figsize=(10,10))
    ax.set_facecolor('white')
    ax1 = plt.gca()
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)
    x0 = df[0].values.reshape((resolution, resolution))
    x1 = df[1].values.reshape((resolution, resolution))
    z = df['y'].values.reshape(x0.shape)
    cs = plt.contourf(x0, x1, z, levels=[0, 0.7],
    colors=[bd_color, '#ffffff'], extend='both')
    if scatter is not None and scatter is True:
        plt.scatter
    cs.cmap.set_over(color_1)
    cs.cmap.set_under(color_2)
    cs.changed()
    fig = ax.get_figure()
    # plt.title(name)
    plt.savefig(os.path.join(save_dir, name + '.png'),dpi=300)

def plot_decision_boundry_2d(name, df, resolution=100, save_dir='results/plots'):
    print("resolution",resolution)
    ax = df.plot.scatter(x = 0, y = 1, c = 'y', cmap = 'ocean', colorbar = False, marker=".")
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_title(name)
    x0 = df[0].values.reshape((resolution, resolution))
    x1 = df[1].values.reshape((resolution, resolution))
    z = df['y'].values.reshape(x0.shape)
    ax.contourf(x0, x1,z ,cmap ='ocean')
    fig = ax.get_figure()
    fig.savefig(os.path.join(save_dir, name + '.png'))


def plot_decision_boundry_2d_old(name, df, grid_df):
    ax = grid_df.plot.scatter(x = 0, y = 1, c = 'white') # background
#     train_df.plot.scatter(x = 0, y = 1, c = 'y', cmap = 'tab10', colorbar = False, ax = ax)
    df.plot.scatter(x = 0, y = 1, c = 'y', cmap = 'tab10', colorbar = False, ax = ax)
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.set_title(name)


def plot(file_name, x, y, labels, x_low=None, x_high=None, y_low=None, y_high=None):
    def is_valid(i, y_pred):
        if y_pred[i] is None:
            return False
        if x_low is not None and x[i] < x_low:
            return False
        if x_high is not None and x[i] > x_high:
            return False
        if y_low is not None and y_pred[i] < y_low:
            return False
        if y_high is not None and y_pred[i] > y_high:
            return False
        return True
    
    inds = []
    for i in range(len(x)):
        if is_valid(i, y[1]):
            inds.append(i) # y_pred from gpt
    
    plt.figure(figsize=(12,6))
    x_valid = x[inds]
    y = np.asarray(y)
    y_valid = y[:, inds]
    colors = ['g', 'b', 'r', 'k', 'y', 'c', 'm']
#     plt.plot(x_valid, y_valid[0], c='g', label='True function')
#     plt.plot(x_valid, y_valid[1], c='b',label='GPT-J')
    for k in range(len(y)):
        plt.plot(x_valid, y_valid[k], c=colors[k], label=labels[k])
    

    plt.legend()
    plt.title("Robustnesss")
    plt.xlabel('x')
    plt.ylabel('y')

    if file_name is None:
        plt.savefig('./plot.png')
    else:
        try:
            plt.savefig(file_name)
        except:
            plt.savefig('./plot.png')

def plot_function(x_train, y_train, X_out=None, y_out=None):
    plt.figure(figsize=(12,6))
    plt.scatter(x_train, y_train, c='g', label='True function')
    if X_out is not None:
        plt.scatter(X_out, y_out, c='r', label='outliers')
    plt.legend()
    plt.title("1D visualization")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def eval(y_test_outputs, y_grid_outputs, test_df,plot=False,X_grid=None, y_grid=None,file_name=None):
    valid_test_x = [test_df[0][i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
    valid_test_y = [test_df["y"][i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]
    valid_test_y_outputs = [y_test_outputs[i] for i in range(len(y_test_outputs)) if y_test_outputs[i] != None]

    print("Valid #outputs/Total #outputs:%d/%d" % (len(valid_test_x),len(y_test_outputs)))
    l2_err = L2error(np.array(valid_test_y_outputs), np.array(valid_test_y))
    print('L2-error {:.4f}'.format(l2_err))

    if plot and X_grid is not None:
#         y_grid_outputs = list(map(query,grid_prompts))
        valid_plot_x = np.array([X_grid[i,0] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])
        valid_plot_y = [y_grid[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None]
        valid_plot_y_outputs = np.array([y_grid_outputs[i] for i in range(len(y_grid_outputs)) if y_grid_outputs[i] != None])
        #     for i in range(len(y_grid_outputs)):
#         if y_grid_outputs[i] is not None:
#             x.append(X_grid[i,0])
#             y_true.append(y_grid[i])
#             y_pred.append(y_grid_outputs[i])

        plt.figure(figsize=(12,6))
        plt.scatter(valid_plot_x,valid_plot_y_outputs,c=['b']*len(valid_plot_x),label='GPTJ Predicted Labels')
        plt.plot(valid_plot_x,valid_plot_y,c='g',label='Observed Labels')

        plt.legend()
        plt.title("1D_visualization")
        plt.xlabel('x')
        plt.ylabel('y')
        if file_name is None:
            plt.savefig('./plot.png')
        else:
            try:
                plt.savefig(file_name)
            except:
                plt.savefig('./plot.png')

    return y_test_outputs,len(valid_test_x),l2_err
