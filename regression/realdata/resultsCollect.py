from curses.ascii import RS
import os, json, openai, time
import numpy as np
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel

func_list = ["linear", "quadratic", "exponential","cosine","l1norm","piecewise"]
metrics = ['loss', 'loss_woo', 'num_o', 'poly_loss', 'krr_loss', 'knn_loss', 'nn_loss', 'xg_loss', 'rf_loss', 'gp_loss']
setting_cols = ['function', 'noise_level', 'p', 'integer', '(lb,ub)', 'n_train', 'n_valid', 'n_test']

sort_func = {'linear':1, 'quadratic':2, 'exponential':3, 'cosine':4, 'l1norm':5, 'piecewise':6}

# kernel = DotProduct() + WhiteKernel()
kernel = ConstantKernel(1.0) * RBF(length_scale=10)
reg_model = {
    'poly': PolynomialFeatures(3),
    'knn': [KNeighborsRegressor(n_neighbors=2),KNeighborsRegressor(n_neighbors=5),KNeighborsRegressor(n_neighbors=8)], 
    'linreg': LinearRegression(),
    'krr': [KernelRidge(kernel="rbf", gamma=0.01), KernelRidge(kernel="rbf", gamma=0.1), KernelRidge(kernel="rbf", gamma=1)],
    'nn': [MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes = (50,50,50), learning_rate_init=0.01), MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes = (50,50,50), learning_rate_init=0.001), MLPRegressor(random_state=1, max_iter=500, hidden_layer_sizes = (50,50,50), learning_rate_init=0.0001)],
    'xg': [GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=5, learning_rate=0.1, loss='squared_error'), GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=5, learning_rate=0.01, loss='squared_error'), GradientBoostingRegressor(n_estimators=500, max_depth=4, min_samples_split=5, learning_rate=0.001, loss='squared_error')],
    'rf': [RandomForestRegressor(n_estimators=500, max_depth=4, random_state=0), RandomForestRegressor(n_estimators=500, max_depth=6, random_state=0)],
    'gp': [GaussianProcessRegressor(alpha=0.16000000000000003, copy_X_train=True,
                         kernel=1.41**2 * RBF(length_scale=0.1),
                         n_restarts_optimizer=5, normalize_y=False,
                         optimizer='fmin_l_bfgs_b',random_state=0),
                         GaussianProcessRegressor(alpha=0.16000000000000003, copy_X_train=True,
                         kernel=1.41**2 * RBF(length_scale=0.1),
                         n_restarts_optimizer=10, normalize_y=False,
                         optimizer='fmin_l_bfgs_b',random_state=0)]
}
baseline_list = ['poly', 'krr', 'knn', 'nn', 'xg', 'rf', 'gp']

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

def regressionLoss(y_pred, y_true, metric = 'RAE', outlier_filter = False, outlier_thres = 5):
    metric = metric.lower()
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if outlier_filter:
        std = regressionLoss(y_pred, y_true, 'RMSE', False)
        outlier_flag = (np.abs(y_pred - y_true) > std*outlier_thres)
        num_outlier = np.sum(outlier_flag)
        return regressionLoss(y_pred[~outlier_flag], y_true[~outlier_flag], metric, False), num_outlier
    y_mean = y_true.mean()
    if metric == 'rae':
        return sum(abs(y_pred - y_true)) / sum(abs(y_mean - y_true))
    elif metric == 'rse':
        return sum((y_pred - y_true)**2) / sum((y_mean - y_true)**2)
    elif metric == 'rmse':
        return ((y_pred - y_true)**2).mean()**.5
    elif metric == 'r square':
        sst = sum((y_true - y_mean) ** 2)
        ssr = sum((y_true - y_pred) ** 2)
        return 1 - ssr/sst


def save_csv(file_list, file_path, metric = 'RAE', grid_loss = False):
    """
        "train_x","train_y",validation_x,validation_y,test_x,test_y,gpt3_test_y,grid_x,grid_y,gpt3_grid_y
        lr_test_y,lr_grid_y,poly_test_y,poly_grid_y,knn_test_y,knn_grid_y
    """
    counter = 1
    all_acc_syn= []
    all_acc_real = []

    n_train = 200
    n_valid= 50
    n_test = 100

    for file in file_list:
        if file.endswith("all.json"):
            print("----------------------------%d------------------------------"%counter)
            counter+=1
            print("file path: ",file)
            with open(file,'r') as fp:
                data_json = json.load(fp)

            syn = False
            is_grid_dataset = True if 'grid_x' in data_json else False
            headtail = os.path.split(file)
            file_name_split = headtail[1].split('_')
            if not syn: 
                dataname = file_name_split[0]
                if len(file_name_split) == 3:
                    context = False
                    n_train = file_name_split[1]
                else:
                    context = True
                    n_train = file_name_split[2]
            else:
                function = file_name_split[0]
                n = int(file_name_split[2])
                p = int(file_name_split[4])
                integer = False if file_name_split[6] == '0' else True
                lb_ub = file_name_split[7].replace(".0","").replace(",",", ")
                noise_level = float(file_name_split[9])
            if syn:
                all_acc_syn.append([noise_level,function,p,integer,lb_ub,n_train,n_valid,n_test])
            else:
                all_acc_real.append([dataname, n_train, context])
            try:
                train_df = pd.read_csv('data/%s/%s_train_%s_num.csv' % (dataname, dataname, n_train))
            except: 
                train_df = pd.read_csv('data/%s/%s_train_%s.csv' % (dataname, dataname, n_train))
            X_train = train_df[train_df.columns[:-1]].values
            y_train = train_df[train_df.columns[-1]].values
            
            
            try:
                test_df = pd.read_csv('data/%s/%s_test_num.csv' % (dataname, dataname))
            except:
                test_df = pd.read_csv('data/%s/%s_test.csv' % (dataname, dataname))
            X_test = test_df[test_df.columns[:-1]].values
            y_test = test_df[test_df.columns[-1]].values

            if is_grid_dataset:
                y_grid = np.array(data_json['grid_y'])
                X_grid = np.array(data_json['grid_x'])
            
                train_min, train_max = file_name_split[7].split(',')
                train_min = float(train_min[1:])
                train_max = float(train_max[:-1])

                loss_idx = X_grid.T[0] >= train_min
                for col_idx in range(X_grid.shape[1]):
                    loss_idx = loss_idx & (X_grid.T[col_idx] >= train_min) & (X_grid.T[col_idx] <= train_max)
                # without extrapolation
                X_grid_woe = X_grid[loss_idx]
                y_grid_woe = y_grid[loss_idx]

            poly = reg_model['poly']
            X_poly_train = poly.fit_transform(X_train)
            poly_reg = linear_model.LinearRegression()
            poly_reg.fit(X_poly_train, y_train)
            poly_test_y = poly_reg.predict(poly.transform(X_test))
            data_json['poly_test_y'] = poly_test_y.tolist()

            poly_loss = regressionLoss(poly_test_y, y_test, metric)

            print("PR %s: %.2f" % (metric , poly_loss))
            data_json['poly_loss'] = poly_loss
            if syn:
                all_acc_syn[-1].append(poly_loss)
            else:
                all_acc_real[-1].append(poly_loss)

            if is_grid_dataset:
                poly_grid_y = poly_reg.predict(poly.transform(X_grid))
                data_json['poly_grid_y'] = poly_grid_y.tolist()
                poly_loss_grid = regressionLoss(poly_grid_y, y_grid, metric)

                poly_grid_y_woe = poly_reg.predict(poly.transform(X_grid_woe))
                poly_loss_grid_woe = regressionLoss(poly_grid_y_woe, y_grid_woe, metric)

                if grid_loss: 
                    data_json['poly_loss_grid'] = poly_loss_grid
                    data_json['poly_loss_grid_woe'] = poly_loss_grid_woe
                    all_acc_syn[-1].extend([poly_loss_grid, poly_loss_grid_woe])
                
            for baseline_idx in range(1, len(baseline_list)):
                # hyperparameter selection
                baseline_loss = []
                method = baseline_list[baseline_idx]
                reg = []
                for baseline_reg in reg_model[method]:
                    baseline_reg.fit(X_train, y_train)
                    baseline_test_y = baseline_reg.predict(X_test)
                    data_json['%s_test_y' % method] = baseline_test_y.tolist()
                    baseline_loss.append(regressionLoss(baseline_test_y, y_test, metric))
                    reg.append(baseline_reg)

                best_idx = np.array(baseline_loss).argmin()
                baseline_loss = baseline_loss[best_idx]
                baseline_reg = reg[best_idx]
                print("%s %s: %.2f" % (method.upper(), metric , baseline_loss))
                data_json['%s_loss' % method] = baseline_loss
                if syn:
                    all_acc_syn[-1].append(baseline_loss)
                else:
                    all_acc_real[-1].append(baseline_loss)

                if is_grid_dataset:
                    baseline_grid_y = baseline_reg.predict(X_grid)
                    data_json['%s_grid_y' % method] = baseline_grid_y.tolist()
                    baseline_loss_grid = regressionLoss(baseline_grid_y, y_grid, metric)

                    baseline_grid_y_woe = baseline_reg.predict(X_grid_woe)
                    baseline_loss_grid_woe = regressionLoss(baseline_grid_y_woe, y_grid_woe, metric)

                    if grid_loss: 
                        data_json['%s_loss_grid' % method] = baseline_loss_grid
                        data_json['%s_loss_grid_woe' % method] = baseline_loss_grid_woe
                        all_acc_syn[-1].extend([baseline_loss_grid, baseline_loss_grid_woe])

            # calculate gpt3 loss
            gpt3_test_y = np.array(data_json['gpt3_test_y'])
            loss = regressionLoss(gpt3_test_y, y_test, metric)

            # try:
            loss_woo, num_o = regressionLoss(gpt3_test_y, y_test, metric, True)

            if syn:
                all_acc_syn[-1].extend([len(y_test), loss, loss_woo, num_o])
            else:
                all_acc_real[-1].extend([len(y_test), loss, loss_woo, num_o])
                
            print('%s     : %.4f' % (metric, loss))
            print('%s(woo): %.4f   #outlier: %2d}' % (metric, loss_woo, num_o))
            data_json['loss_woo'] = loss_woo
            data_json['num_o'] = num_o
            data_json['loss'] = loss

            # calculate gptj loss
#             gptj_test_y = np.array(data_json['gptj_test_y'])
#             loss_gptj, _ = regressionLoss(gptj_test_y, y_test, metric, True)
#             data_json['loss_gptj'] = loss_gptj
#             print("GPTJ %s: %.4f" % (metric, loss_gptj))
            
#             if syn:
#                 all_acc_syn[-1].append(loss_gptj)
#             else:
#                 all_acc_real[-1].append(loss_gptj)

            if grid_loss:
                gpt3_grid_y = np.array(data_json['gpt3_grid_y'])
                invalid_idx = gpt3_grid_y == None
                valid_y_grid = y_grid[~invalid_idx]
                valid_gpt3_grid_y = gpt3_grid_y[~invalid_idx]
                data_json['pc_valid_grid'] = 1-invalid_idx.mean()

                loss_grid = regressionLoss(valid_gpt3_grid_y, valid_y_grid, metric, True)

                gpt3_grid_y_woe = gpt3_grid_y[loss_idx]
                invalid_idx = gpt3_grid_y_woe == None
                valid_y_grid_woe = y_grid_woe[~invalid_idx]
                valid_gpt3_grid_y_woe = gpt3_grid_y_woe[~invalid_idx]
                loss_grid_woe = regressionLoss(valid_gpt3_grid_y_woe, valid_y_grid_woe, metric, True)

                all_acc_syn[-1].extend([len(valid_y_grid), loss_grid, len(valid_y_grid_woe),loss_grid_woe])
                
                data_json['loss_grid'] = loss_grid
                data_json['pc_valid_grid_woe'] = 1-invalid_idx.mean()
                data_json['loss_grid_woe'] = loss_grid_woe

#                 gptj_grid_y = np.array(data_json['gptj_grid_y'])
#                 loss_grid_gptj = regressionLoss(gptj_grid_y, y_grid, metric, True)
#                 data_json['loss_grid_gptj'] = loss_grid_gptj

            with open(file,'w') as fp:
                json.dump(data_json, fp, cls=NpEncoder)
    head_tail = os.path.split(file_path)
    if len(all_acc_syn) > 0:
        all_acc_syn = pd.DataFrame(all_acc_syn)
        
        if grid_loss:
            columns = []
            for b in baseline_list:
                columns += ['%s_loss' % b, '%s_loss_grid' % b, '%s_loss_woe' %b] 
            all_acc_syn.columns = ['noise_level', 'function', 'p', 'integer', '(lb,ub)', 'n_train', 'n_valid', 'n_test'] + columns + ['num_valids', 'loss', 'loss_woo', 'num_o', 'num_valid_grid','loss_grid', 'num_valid_grid_woe', 'loss_grid_woe', 'loss_grid_gptj']
        else:
            columns = []
            for b in baseline_list:
                columns += ['%s_loss' % b] 
            all_acc_syn.columns = ['noise_level', 'function', 'p', 'integer', '(lb,ub)', 'n_train', 'n_valid', 'n_test'] + columns + ['num_valids', 'loss', 'loss_woo', 'num_o']
            
        all_acc_syn.to_csv(os.path.join(head_tail[0], 'syn_'+head_tail[1]))
    if len(all_acc_real) > 0:
        columns = []
        for b in baseline_list:
            columns += ['%s_loss' % b] 
        all_acc_real = pd.DataFrame(all_acc_real)
        all_acc_real.columns = ['dataset', 'pc_train', 'context'] + columns + ['num_valids','loss','loss_woo','num_o']
        all_acc_real.to_csv(os.path.join(head_tail[0], 'real_'+head_tail[1]))