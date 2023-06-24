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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()

func_list = ["linear", "quadratic", "exponential","cosine","l1norm","piecewise"]
setting_cols = ['function', 'noise_level', 'p', 'integer', '(lb,ub)', 'n']
metrics = ['poly_loss', 'krr_loss', 'knn_loss', 'nn_loss', 'xg_loss', 'rf_loss', 'gp_loss', 'gptj_loss', 'gpt3_loss']
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

def regressionLoss(y_pred, y_true, metric = 'RAE', outlier_filter = False, outlier_thres = 3):
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

def collect_data_json(file_list, metric = 'RAE', grid_loss = False, gpt3 = True, gptj = True, baseline = True, grid_compute = True):
    """
        "train_x","train_y",validation_x,validation_y,test_x,test_y,gpt3_test_y,grid_x,grid_y,gpt3_grid_y
        lr_test_y,lr_grid_y,poly_test_y,poly_grid_y,knn_test_y,knn_grid_y
    """
    counter = 1

    for file in file_list:
        if file.endswith("all.json"):
            print("----------------------------%d------------------------------"%counter)
            counter+=1
            print("file path: ",file)
            with open(file,'r') as fp:
                data_json = json.load(fp)

            syn = True if not 'openml' in file else False
            is_grid_dataset = True if ('grid_x' in data_json) and grid_compute else False
            file_name_split = os.path.split(file)[1].split('_')

            X_train = np.array(data_json['train_x'])
            y_train = np.array(data_json['train_y'])

            X_valid = np.array(data_json['validation_x'])
            y_valid = np.array(data_json['validation_y'])

            X_test= np.array(data_json['test_x'])
            y_test = np.array(data_json['test_y'])

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

            if baseline:
                poly = reg_model['poly']
                X_poly_train = poly.fit_transform(X_train)
                poly_reg = linear_model.LinearRegression()
                poly_reg.fit(X_poly_train, y_train)
                poly_test_y = poly_reg.predict(poly.transform(X_test))
                data_json['poly_test_y'] = poly_test_y.tolist()

                poly_loss = regressionLoss(poly_test_y, y_test, metric)

                print("PR %s: %.2f" % (metric , poly_loss))
                data_json['poly_loss'] = poly_loss

                if is_grid_dataset:
                    poly_grid_y = poly_reg.predict(poly.transform(X_grid))
                    data_json['poly_grid_y'] = poly_grid_y.tolist()
                    poly_loss_grid = regressionLoss(poly_grid_y, y_grid, metric)

                    poly_grid_y_woe = poly_reg.predict(poly.transform(X_grid_woe))
                    poly_loss_grid_woe = regressionLoss(poly_grid_y_woe, y_grid_woe, metric)

                    if grid_loss: 
                        data_json['poly_loss_grid'] = poly_loss_grid
                        data_json['poly_loss_grid_woe'] = poly_loss_grid_woe
                    
                for baseline_idx in range(1, len(baseline_list)):
                    # hyperparameter selection
                    baseline_loss = []
                    method = baseline_list[baseline_idx]
                    reg = []
                    for baseline_reg in reg_model[method]:
                        baseline_reg.fit(X_train, y_train)
                        baseline_valid_y = baseline_reg.predict(X_valid)
                        baseline_loss.append(regressionLoss(baseline_valid_y, y_valid, metric))
                        reg.append(baseline_reg)

                    best_idx = np.array(baseline_loss).argmin()
                    baseline_reg = reg[best_idx]
                    baseline_test_y = baseline_reg.predict(X_test)
                    data_json['%s_test_y' % method] = baseline_test_y.tolist()
                    baseline_loss = regressionLoss(baseline_test_y, y_test, metric)
                    print("%s %s: %.2f" % (method.upper(), metric , baseline_loss))
                    data_json['%s_loss' % method] = baseline_loss

                    if is_grid_dataset:
                        baseline_grid_y = baseline_reg.predict(X_grid)
                        data_json['%s_grid_y' % method] = baseline_grid_y.tolist()
                        baseline_loss_grid = regressionLoss(baseline_grid_y, y_grid, metric)

                        baseline_grid_y_woe = baseline_reg.predict(X_grid_woe)
                        baseline_loss_grid_woe = regressionLoss(baseline_grid_y_woe, y_grid_woe, metric)

                        if grid_loss: 
                            data_json['%s_loss_grid' % method] = baseline_loss_grid
                            data_json['%s_loss_grid_woe' % method] = baseline_loss_grid_woe

            # calculate gpt3 loss
            if gpt3:
                gpt3_test_y = np.array(data_json['gpt3_test_y'])

                invalid_idx = gpt3_test_y == None
                valid_test_y = y_test[~invalid_idx]
                valid_gpt3_test_y = gpt3_test_y[~invalid_idx]

                print("Valid #outputs/Total #outputs:%d/%d" % (len(valid_test_y),len(y_test)))
                loss = regressionLoss(valid_gpt3_test_y, valid_test_y, metric)

                # try:
                loss_woo, num_o = regressionLoss(valid_gpt3_test_y, valid_test_y, metric, True)
                    
                print('%s     : %.4f' % (metric, loss))
                print('%s(woo): %.4f   #outlier: %2d}' % (metric, loss_woo, num_o))
                data_json['loss_woo'] = loss_woo
                data_json['num_o'] = num_o
                data_json['loss'] = loss
                
                if grid_loss:
                    gpt3_grid_y = np.array(data_json['gpt3_grid_y'])
                    invalid_idx = gpt3_grid_y == None
                    valid_y_grid = y_grid[~invalid_idx]
                    valid_gpt3_grid_y = gpt3_grid_y[~invalid_idx]
                    data_json['pc_valid_grid'] = 1-invalid_idx.mean()

                    loss_grid,_ = regressionLoss(valid_gpt3_grid_y, valid_y_grid, metric, True)

                    gpt3_grid_y_woe = gpt3_grid_y[loss_idx]
                    invalid_idx = gpt3_grid_y_woe == None
                    valid_y_grid_woe = y_grid_woe[~invalid_idx]
                    valid_gpt3_grid_y_woe = gpt3_grid_y_woe[~invalid_idx]
                    loss_grid_woe,_ = regressionLoss(valid_gpt3_grid_y_woe, valid_y_grid_woe, metric, True)

                    data_json['loss_grid'] = loss_grid
                    data_json['pc_valid_grid_woe'] = 1-invalid_idx.mean()
                    data_json['loss_grid_woe'] = loss_grid_woe
                    
            if gptj:
                # calculate gptj loss
                gptj_test_y = np.array(data_json['gptj_test_y'])
                gptj_test_y[gptj_test_y == None] = y_train.mean()
                loss_gptj, _ = regressionLoss(gptj_test_y, y_test, metric, True)
                data_json['loss_gptj'] = loss_gptj
                print("GPTJ %s: %.4f" % (metric, loss_gptj))
                
                if grid_loss:
                    gptj_grid_y = np.array(data_json['gptj_grid_y'])
                    loss_grid_gptj,_ = regressionLoss(gptj_grid_y, y_grid, metric, True)
                    data_json['loss_grid_gptj'] = loss_grid_gptj
                    
                    gptj_grid_y_woe = gptj_grid_y[loss_idx]
                    loss_grid_woe_gptj, _ = regressionLoss(gptj_grid_y_woe, y_grid_woe, metric, True)
                    data_json['loss_grid_gptj_woe'] = loss_grid_woe_gptj

            with open(file,'w') as fp:
                json.dump(data_json,fp, cls=NpEncoder)

def save_csv(file_list, file_path, metric = 'RAE', grid_loss = False, gpt3 = True, gptj = True, baselines = True, gpt3_larger = False):
    counter = 0
    all_acc_syn = []
    method_list = []
    if baselines: method_list += baseline_list
    if gpt3: method_list.append('gpt3')
    if gptj: method_list.append('gptj')

    for file in file_list:
         if file.endswith('all.json'):
            counter += 1
            print("----------------------------%d------------------------------"%counter)
            print("file path: ", file)
            with open(file, 'r') as fp:
                data_json = json.load(fp)
            
            is_grid_dataset = True if 'grid_x' in data_json else False
            headtail = os.path.split(file)
            file_name_split = headtail[1].split('_')

            function = file_name_split[0]
            n = int(file_name_split[2])
            p = int(file_name_split[4])
            integer = False if file_name_split[6] == '0' else True
            lb_ub = file_name_split[7].replace(".0","").replace(",",", ")
            noise_level = float(file_name_split[9])

            all_acc_syn.append([noise_level,function,p,integer,lb_ub,n])

            for method in method_list:
                if grid_loss:
                    try:
                        method_loss = data_json['%s_loss_grid_woe' % method]
                    except KeyError:
                        y_grid = np.array(data_json['grid_y'])
                        method_grid_y = data_json['%s_grid_y' % method]
                        method_loss, _ = regressionLoss(method_grid_y, y_grid, metric, True)
                else:
                    y_train = np.array(data_json['train_y'])
                    y_test = np.array(data_json['test_y'])
                    method_test_y = np.array(data_json['%s_test_y' % method])
                    method_test_y[method_test_y == None] = y_train.mean()
                    method_loss, _ = regressionLoss(method_test_y, y_test, metric, True)

                all_acc_syn[-1].append(method_loss)
            
            if gpt3_larger:
                for method in ['babbage', 'curie', 'davinci']:
                    y_train = np.array(data_json['train_y'])
                    y_test = np.array(data_json['test_y'])
                    method_test_y = np.array(data_json['gpt3_test_y_%s' % method])
                    method_test_y[method_test_y == None] = y_train.mean()
                    method_loss, _ = regressionLoss(method_test_y, y_test, metric, True)

                    all_acc_syn[-1].append(method_loss)
    columns = ['noise_level', 'function', 'p', 'integer', '(lb,ub)', 'n']
    head_tail = os.path.split(file_path)
    for method in method_list:
        columns.append('%s_loss' % method)
    if gpt3_larger: columns.extend(['babbage_loss', 'curie_loss', 'davinci_loss'])
    all_acc_syn = pd.DataFrame(all_acc_syn)
    all_acc_syn.columns = columns
    all_acc_syn.to_csv(os.path.join(head_tail[0], 'syn_'+head_tail[1]))

def summary_syn(folder_name, num_sims = 5, p = 50, noise_level = 0.1, integer = False, positive = False, metrics = metrics):
    mean_df = pd.read_csv(os.path.join(folder_name,'syn_mean_across_%dsims.csv' % num_sims))
    std_df = pd.read_csv(os.path.join(folder_name, 'syn_std_across_%dsims.csv' % num_sims))
    report_metric = ['function'] + metrics
    if integer and positive:
        rg = '(0, 300)'
    elif (not integer) and positive:
        rg = '(0, 10)'
    elif integer and (not positive):
        rg = '(-150, 150)'
    else:
        rg = '(-10, 10)'
        
    return (mean_df[(mean_df.p == p) & 
            (mean_df.noise_level == noise_level) & 
            (mean_df.integer == integer) & 
            (mean_df['(lb,ub)'] == rg)].sort_values(by = 'function', key = lambda x: x.map(sort_func))[report_metric].reset_index(drop = True), 
            std_df[(std_df.p == p) & 
            (std_df.noise_level == noise_level) & 
            (std_df.integer == integer) & 
            (std_df['(lb,ub)'] == rg)].sort_values(by = 'function', key = lambda x: x.map(sort_func))[report_metric].reset_index(drop = True)
           )

def print_overleaf_syn(folder_name, num_sims = 3, p = 50, noise_level = 0.1, integer = False, positive = False, metric = 'RAE', metrics = metrics):
    mean_sum, std_sum = summary_syn(folder_name, num_sims, p = p, noise_level = noise_level, integer = integer, positive = positive, metrics = metrics)
    tex = r"""
\begin{table}[]
\begin{center}
\begin{small}
\begin{sc}
\tiny{
\begin{tabularx}{0.9\textwidth}{lccccccccc}
\toprule
Method & \qr{} & \krr{} & \knn{}  & \ann{} & \gbt{} & \randomf{} & \gp{} & \lift{}/\gptj{} & \lift{}/\gptt{} \\
  &  %s  & %s  & %s  &  %s &  %s  &  %s &  %s & %s & %s \\ \midrule
    """ % (metric, metric, metric, metric, metric, metric, metric, metric, metric)
    rows = ['\linear{}', '\quadr{}', '\expo{}', '\cosi{}', '\lone{}', '\pw{}']
    for i in range(6):
        row = rows[i]
        for col in metrics:
            row += '& $%.2f \pm %.1f$ ' % (mean_sum.loc[i][col], std_sum.loc[i][col])
        row += '\\\ \n'
        tex += row
    tex += r"""  
\bottomrule
\end{tabularx} }
\end{sc}
\end{small}
\end{center}
\caption{Comparison of various methods in approximating different functions when 
"""
    if integer and positive:
        region = '\sZ^+'
    elif (not integer) and positive:
        region = '\sR^+'
    elif integer and (not positive):
        region = '\sZ'
    else:
        region = '\sR'
    tex +=  r'$p=%d, \sigma=%.1f, (\rvx,\ry) \in %s$. }' % (p, noise_level, region)
    tex += r"""
\end{table}
    """
    print(tex)

def get_mean_std_syn(folder_name, num_sims = 3, metrics = metrics):
    dfs = []
    for i in range(1,num_sims+1):
        dfs.append(pd.read_csv(os.path.join(folder_name, 'data_%d' % i, '%s%d.csv' % ('syn_all_models_acc', i))).sort_values(by = setting_cols).reset_index(drop = True))
    mean_df, std_df = dfs[0].copy(), dfs[0].copy()

    for metric in metrics:
        bucket = []
        for i in range(num_sims):
            bucket.append(dfs[i][metric])
        bucket = pd.concat(bucket, axis = 1)
        mean_df[metric] = bucket.mean(axis = 1)
        std_df[metric] = bucket.std(axis = 1)

    mean_df.to_csv(os.path.join(folder_name,'syn_mean_across_%dsims.csv' % num_sims))
    std_df.to_csv(os.path.join(folder_name, 'syn_std_across_%dsims.csv' % num_sims))

def get_mean_std_real(folder_name, num_sims = 3, settings = None):
    dfs = []
    for i in range(1,num_sims+1):
        dfs.append(pd.read_csv(os.path.join(folder_name, 'data_%d' % i, '%s%d.csv' % ('real_all_models_acc', i))).sort_values(by = 'did').reset_index(drop = True))
        row_idxs = []
        for idx, row in dfs[i-1].iterrows():
            if (settings == None) or row['did'].tolist() in settings:
                row_idxs.append(idx)
        dfs[i-1] = dfs[i-1].loc[row_idxs].reset_index(drop = True)
        dfs[i-1]['num_invalids'] = (100 - dfs[i-1]['num_valids'])/100
        if (settings != None) and (len(dfs[i-1]) < len(settings)):
            for st in settings:
                if not st in dfs[i-1]['did'].values.tolist():
                    raise RuntimeError('Did %s is missing from data_%d folder!' % (st, i))

    mean_df, std_df = dfs[0].copy(), dfs[0].copy()

    for metric in metrics:
        bucket = []
        for i in range(num_sims):
            bucket.append(dfs[i][metric])
        bucket = pd.concat(bucket, axis = 1)
        mean_df[metric] = bucket.mean(axis = 1)
        std_df[metric] = bucket.std(axis = 1)

    mean_df.to_csv(os.path.join(folder_name,'real_mean_across_%dsims.csv' % num_sims))
    std_df.to_csv(os.path.join(folder_name, 'real_std_across_%dsims.csv' % num_sims))

def collect_data(folder_name, num_sims = 5, metric = 'RAE', settings_syn = None, settings_real = None):
    """
    Run baseline methods and collect the experiment results.

    Parameters
    ----------------
    folder_name: the folder you want to summarize
    num_sims: how many simulations in this folder
    metric: default 'RAE'. Options: 'RAE', 'R Square', 'RMSE'
    settings_syn: a list if you only want to summarize a fixed set of settings of synthetic datsets.
                    Example: [
                                ['cosine', 0.1, 2, False, '(-10, 10)', 200, 50, 100],
                                ['exponential', 0.1, 2, False, '(-10, 10)', 200, 50, 100],
                            ]
    settings_real: a list of dids if you only want to collect the results for a fixed set of dids.
    """
    for i in range(1,num_sims+1):
        data_path = 'data_%d' % i
        if not os.path.isdir(folder_name):
            os.mkdir(folder_name)
        file_path = os.path.join(folder_name, data_path, 'all_models_acc%d.csv' % i)
        save_csv(os.path.join(folder_name,data_path), file_path, metric = metric)
    try:
        get_mean_std_syn(folder_name, num_sims, settings_syn)
    except:
        print('No results for synthetic datasets is found!')
    try:
        get_mean_std_real(folder_name, num_sims, settings_real)
    except:
        print('No results for real datasets is found!')

def operate_all_FineTunes_on_openai(op='list', openai_key = '[REPLACE IT WITH YOUR OPENAI KEY]'):
    openai.api_key = openai_key
    OpenAIObject = openai.FineTune.list()
    finetune_list = OpenAIObject['data']
    for i, file in enumerate(finetune_list):
        flag = True
        while(flag):
            try:
            #if 1:
                if op == 'cancel':
                    print(file['status'])
                    if not file['status'] in ['failed','cancelled','succeeded']:
                        print(op,' ',i,' ',file['id'])
                        print(openai.FineTune.cancel(file['id']))
                        
                if op == 'list':
                    pass
                    if file['status'] not in ['failed','cancelled','succeeded']:
                        print(op,' ',i,' ',file['id'])
                        print(openai.FineTune.retrieve(file['id']))
                        
                flag = False
            except:
                print("An exception occurred")
                flag = True
                time.sleep(60)

def operate_all_Files_on_openai(op='list', openai_key = '[REPLACE IT WITH YOUR OPENAI KEY]'):
    openai.api_key = openai_key
    OpenAIObject = openai.File.list()
    file_list = OpenAIObject['data']
    
    for i, file in enumerate(file_list):
        flag = True
        while(flag):
            try:
                if op == 'delete':
                    openai.File.delete(file['id'])
                print(op,' ',i,' ',file['id'])
                flag = False
            except:
                print("An exception occurred")
                flag = True
                time.sleep(60)

def region(rg, integer):
    if rg == '(0, 300)' and integer:
        region = '\sZ^+'
    elif rg == '(0, 300)' and not integer:
        region = '\sR^+'
    elif rg == '(-150, 150)' and integer:
        region = '\sZ'
    else:
        region = '\sR'
    return region

def print_overleaf_basic(folder_name, num_sims = 3, manipulate = 'noise level', p = 1, metric = 'RAE', metrics = metrics, report_func = ['linear', 'piecewise']):
    mean_df = pd.read_csv(os.path.join(folder_name,'syn_mean_across_%dsims.csv' % num_sims))
    std_df = pd.read_csv(os.path.join(folder_name, 'syn_std_across_%dsims.csv' % num_sims))
    
    if manipulate == 'noise level':
        report_metric = ['function', 'noise_level', 'num_invalids', 'loss', 'poly_loss', 'krr_loss', 'knn_loss', 'nn_loss', 'xg_loss', 'rf_loss', 'gp_loss']
    
        report_idx = ((mean_df.noise_level <0.2) | (mean_df.noise_level >0.2)) & (mean_df.p == p) & (mean_df.integer == False) & (mean_df['(lb,ub)'] == '(-10, 10)')
        mean_sum = mean_df[report_idx].sort_values(by = 'function', key = lambda x: x.map(sort_func))[report_metric].reset_index(drop = True)[:6]
        std_sum = std_df[report_idx].sort_values(by = 'function', key = lambda x: x.map(sort_func))[report_metric].reset_index(drop = True)[:6]
        tex = r"""
\begin{table}[]
\begin{center}
\begin{small}
\begin{sc}
\tiny{
\begin{tabularx}{0.9\textwidth}{llccccccc}
\toprule
Method &  & \multicolumn{2}{c}{\gpt{}}   & \qr{} & \krr{} & \knn{}  & \ann{} & \xg{} & \rf{} \\
  & $\sigma$ &  Invalid outputs  & %s  & %s  &  %s &  %s  &  %s &  %s & %s \\ \midrule
        """ %  (metric, metric, metric, metric, metric, metric, metric)
        rows = ['\linear{}', '\quadr{}']
        for i in range(6):
            func_idx = i // 3
            if i == 3: tex += '\midrule'
            first_func_idx = (i % 3) == 0

            if first_func_idx:
                row = '\multirow{3}{*}{%s} & %.1f' % (rows[func_idx], mean_sum.loc[i]['noise_level'])
            else:
                row = '& %.1f' %  mean_sum.loc[i]['noise_level']

            for col in mean_sum.columns[2:]:
                row += '& $%.2f \pm %.2f$ ' % (mean_sum.loc[i][col], std_sum.loc[i][col])
            row += '\\\ \n'
            tex += row
        tex += r"""  
\bottomrule
\end{tabularx} }
\end{sc}
\end{small}
\end{center}
\caption{Comparison of various methods in approximating different functions under different noise level when 
    """

        tex +=  r'$p=%d, (\rvx,\ry) \in %s$. }' % (p, '\sR')
        tex += r"""
\end{table}
        """
    elif manipulate == 'input type':
        report_idx = np.zeros(mean_df.shape[0])
        for integer in [True, False]:
            for positive in [True, False]:
                if integer and positive:
                    rg = '(0, 300)'
                elif (not integer) and positive:
                    rg = '(0, 300)'
                elif integer and (not positive):
                    rg = '(-150, 150)'
                else:
                    rg = '(-150, 150)'
                report_idx += (mean_df.noise_level == 0.1) & (mean_df['integer'] == integer) & (mean_df['(lb,ub)'] == rg) & (mean_df.p == p) & ((mean_df['function'] == report_func[0]) | (mean_df['function'] == report_func[1]))
        report_idx = report_idx >= 1
        report_metrics = ['function','integer', '(lb,ub)'] + metrics
        mean_sum = mean_df[report_idx].sort_values(by = 'function', key = lambda x: x.map(sort_func))[report_metrics].reset_index(drop = True)[:8]
        std_sum = std_df[report_idx].sort_values(by = 'function', key = lambda x: x.map(sort_func))[report_metrics].reset_index(drop = True)[:8]
        tex = r"""
\begin{table}[]
\begin{center}
\begin{small}
\begin{sc}
\tiny{
\begin{tabularx}{0.9\textwidth}{llccccccc}
\toprule
Method & \gptt{} & \gptj{} & \qr{} & \krr{} & \knn{}  & \ann{} & \xg{} & rf{} \\
  & $\gX$ & %s  & %s  & %s  &  %s &  %s  &  %s &  %s & %s \\ \midrule
        """ %  (metric, metric, metric, metric, metric, metric, metric, metric)
        rows = ['\expo{}', '\cosi{}']
        for i in range(8):
            func_idx = i // 4
            if i == 4: tex += '\midrule'
            first_func_idx = (i % 4) == 0

            if first_func_idx:
                row = '\multirow{4}{*}{%s} & $%s$' % (rows[func_idx], region(mean_sum.loc[i]['(lb,ub)'], mean_sum.loc[i]['integer']))
            else:
                row = '& $%s$' %  region(mean_sum.loc[i]['(lb,ub)'], mean_sum.loc[i]['integer'])

            for col in metrics:
                row += '& $%.2f \pm %.1f$ ' % (mean_sum.loc[i][col], std_sum.loc[i][col])
            row += '\\\ \n'
            tex += row
        tex += r"""  
\bottomrule
\end{tabularx} }
\end{sc}
\end{small}
\end{center}
\caption{Comparison of various methods in approximating different functions with different numerical type when 
    """

        tex +=  r'$p=%d, (\rvx,\ry) \in %s$. }' % (p, '\sR')
        tex += r"""
\end{table}
        """
    print(tex)

