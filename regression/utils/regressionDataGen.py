import pandas as pd
import numpy as np
import os, random, itertools, openml
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class dataGenerate(object):
    """
    A class of functions for generating jsonl dataset. 
    """
    def __init__(self,data_dir):
        self.data_dir = data_dir

    def data2text(self, row, integer = False, label = True):
        prompt = "When we have " 
        for i in range(len(row)-label):
            if integer:
                prompt += "x%d=%d, " % (i+1, row[i])
            else:
                prompt += "x%d=%.4f, " % (i+1, row[i]) 
        prompt += "what should be the y value?"
        if not label:
            return "%s###" % prompt
        else:
            if integer:
                completion = "%d" % row['y']
            else:
                completion = "%.3f" % row['y']
            return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
    
    def df2jsonl(self, df, filename, integer = False):
        jsonl = '\n'.join(df.apply(func = partial(self.data2text, integer = integer), axis = 1).tolist())
        with open(os.path.join(self.data_dir, filename), 'w') as f:
            f.write(jsonl)
        print("Save a file at:",os.path.join(self.data_dir, filename))
        return os.path.join(self.data_dir, filename)
            
    def array2prompts(self, X, integer = False):
        return list(map(partial(self.data2text, integer = integer, label = False), X))

    def X_generate(self, n, p, integer = False, lb = -10, ub = 10, donut = False):
        if donut: 
            X = np.zeros(p).reshape(1, p)
            while X.shape[0] < n:
                X = np.random.uniform(lb, ub, n*2*p).reshape(n*2,p)
                q1 = lb + (ub-lb) / 3
                q2 = ub - (ub-lb) / 3
                idx = X.T[0] >= q1
                for i in range(p):
                    idx = idx & (X.T[i] >= q1) & (X.T[i] <= q2)
                X = X[~idx]
            X = X[:n]
        else:
            X = np.random.uniform(lb, ub, n*p).reshape(n,p)
            if integer: X = X.round()
        return X
    
    def gridX_generate(self, interval, p, integer, resolution = 100):
        lb, ub = interval
        X_grid = np.linspace(lb * np.ones(p), ub * np.ones(p), resolution).T
        X_grid = np.array(list(itertools.product(*X_grid)))
        grid_prompts = self.array2prompts(X_grid, integer)
        return X_grid, grid_prompts
        
    def data_split(self, X, y, n_train, n_valid):
        n = X.shape[0]
        idx = np.arange(n)
        random.shuffle(idx)
        train_idx, valid_idx = idx[:int(n_train)], idx[int(n_train):]
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        
        return X_train, X_valid, y_train, y_valid
    
    def piecewise1D(self, x):
        if  x < -3.0:
            y = x + 6
        elif -3.0 <= x < 3.0:
            y = 0
        else:
            y = x - 6
        return ((y + 4)/8 - 0.5) * 18
        
    def piecewisehD(self, x):
        return np.vectorize(self.piecewise1D)(x).mean()
    
    def linear(self, X, beta):
        return np.dot(X, beta) / len(beta)
    
    def quadratic(self, X):
        return np.apply_along_axis(lambda x: ((x**2 - 50)*9/50).mean(), 1, X)
    
    def exponential(self, X):
        return np.apply_along_axis(lambda x: ((np.exp(0.2*x) - np.exp(-2)) / (np.exp(2) - np.exp(-2)) - 0.5).mean() * 18, 1, X)
    
    def cosine(self, X):
        return np.apply_along_axis(lambda x: np.cos(0.5*np.pi*x).mean()*9, 1, X)
    
    def l1norm(self, X):
        return np.apply_along_axis(lambda x: ((np.abs(x)/10 - 0.5) * 18).mean(), 1, X)
    
    def piecewise(self, X):
        return np.apply_along_axis(self.piecewisehD, 1, X)

    def generate(self, func, n_train, n_valid, n_test, p, integer = False, 
                 noise_level = 0, test_int = None, outliers = None,
               lb = -10, ub = 10, grid_int = None, resolution = 100, 
               beta = None, donut = False
              ):
        np.random.seed(123)
        if beta is None: beta = np.ones(p) * 0.9
        if test_int is None: test_int = (lb, ub)
        if grid_int is None: grid_int = (lb - (ub-lb)/2, ub + (ub-lb)/2)
        
        # Generate x and y   
        n = n_train + n_valid
        X = self.X_generate(n, p, integer, lb, ub, donut = donut)
        X_test = self.X_generate(n_test, p, integer, test_int[0], test_int[1])
        
        if p in [1, 2]:
            X_grid, grid_prompts = self.gridX_generate(grid_int, p, integer, resolution)
            output_file = open(os.path.join(self.data_dir,'%s_n_%d_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_grid_prompts.txt')%(func,n,p,integer,lb,ub,noise_level), 'w')
            for ele in grid_prompts:
                output_file.write(ele + '\n')
            output_file.close()
        
        if func == 'linear':
            y_true = self.linear(X, beta)
            y_test = self.linear(X_test, beta)
            if p in [1,2]: y_grid = self.linear(X_grid, beta)
            
        elif func == 'quadratic':
            y_true = self.quadratic(X)
            y_test = self.quadratic(X_test)
            if p in [1,2]: y_grid = self.quadratic(X_grid)
            
        elif func == 'exponential':
            y_true = self.exponential(X)
            y_test = self.exponential(X_test)
            if p in [1,2]: y_grid = self.exponential(X_grid)
            
        elif func == 'cosine':
            y_true = self.cosine(X)
            y_test = self.cosine(X_test)
            if p in [1,2]: y_grid = self.cosine(X_grid)
            
        elif func == 'l1norm':
            y_true = self.l1norm(X)
            y_test = self.l1norm(X_test)
            if p in [1,2]: y_grid = self.l1norm(X_grid)
            
        elif func == 'piecewise':
            y_true = self.piecewise(X)
            y_test = self.piecewise(X_test)
            if p in [1,2]: y_grid = self.piecewise(X_grid)
            
        else:
            raise NotImplementedErrors("invalid function name")
        
        y = (y_true + np.random.normal(0,noise_level*np.std(y_true),n)).reshape(n,1)

        if integer: y = y.round()
        
        # split into train, valid, test dataset
        X_train, X_valid, y_train, y_valid = self.data_split(X, y, n_train, n_valid)

        if outliers:
            X_out, y_out = outliers
            X_train = np.concatenate([X_train, X_out], axis = 0)
            y_train = np.concatenate([y_train, y_out], axis = 0)
  
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test
        
        train_file = self.df2jsonl(train_df, '%s_n_%d_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_train.jsonl'%(func,n,p,integer,lb,ub,noise_level), integer)
        valid_file = self.df2jsonl(valid_df, '%s_n_%d_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_valid.jsonl'%(func,n,p,integer,lb,ub,noise_level), integer)
        test_prompts = self.array2prompts(X_test, integer = integer)
        valid_prompts = self.array2prompts(X_valid, integer = integer)
            
        if p in [1,2]:
            return train_df, valid_df, test_df, test_prompts, valid_prompts, grid_prompts, X_grid, y_grid, train_file, valid_file
        else:
            return train_df, valid_df, test_df, test_prompts, valid_prompts, None, None, None, train_file, valid_file

class dataOpenMLLoad(object):
    """
    A class of functions for loading OpenML datasets. 
    """
    def __init__(self,data_dir):
        self.data_dir = data_dir    

    def load_openml(self, did=-1, normalize=False, ignore_cat=False):
        # if not os.path.isdir(os.path.join(self.data_dir, 'openml_datasets')):
        #     os.mkdir(os.path.join(self.data_dir, 'openml_datasets'))
        fpath = os.path.join('openml_datasets', f'{did}_{normalize}.npy')
        if not os.path.isdir('openml_datasets'): os.mkdir('openml_datasets')
        if os.path.isfile(fpath):
            data = np.load(fpath, allow_pickle=True)
            X, y = data.item()['X'], data.item()['y']
            attribute_names = data.item()['attr']
            label =  data.item()['label']
        else:	
            # dataset
            ds = openml.datasets.get_dataset(did)
            # values
            X, y, categorical_indicator, attribute_names = ds.get_data(target=ds.default_target_attribute)
            Xy = pd.concat([X,y], axis=1, ignore_index=True) # X & y concatenated together
            if ignore_cat:
                # non-cat
                non_categorial_indices = np.where(np.array(categorical_indicator) == False)[0] # find where categorical columns are  
                Xy = Xy.iloc[:, [*non_categorial_indices, -1]] # Slice columns -- ignore categorical X columns and add y column (-1)
                attribute_names = [attribute_names[i] for i in non_categorial_indices]        
            Xy.replace('?', np.NaN, inplace=True) # replace ? with NaNs    
            Xy = Xy[Xy.iloc[:, -1].notna()] # remove all the rows whose labels are NaN
            y_after_NaN_removal = Xy.iloc[:, -1]
            Xy.dropna(axis=1, inplace=True) # drop all the columns with missing entries
            Xy.dropna(inplace=True) # drop all the rows with missing entries
            assert((Xy.iloc[:, -1] == y_after_NaN_removal).all())
            X, y = Xy.iloc[:, :-1], Xy.iloc[:, -1]
            # if X.shape[0] == 0 or X.shape[1] == 0: # check if X is empty or not
            #     print("Empty dataset")
            # else:
            #     if normalize:
            #         scaler = StandardScaler()
            #         X = scaler.fit_transform(X)
            #     else:
            #         X = X.to_numpy(dtype=np.float32)
            #     y = y.cat.codes.values
            #import pdb; pdb.set_trace()
            assert(X.shape[1] == len(attribute_names))
            label = ds.default_target_attribute
            np.save(fpath, {'X': X, 'y': y, 'attr': attribute_names, 'label': label})
        return X, y, attribute_names, label

    def data2text_openml(self, row, integer = False, label = True, 
				  context = False, feature_names = None, target_names = None, init = '', end = ''):
        if context:
            prompt = init
            for i in range(len(row)-label):
                v = row[i]
                if isinstance(v, np.float):
                    prompt += "%s=%.4f, " % (feature_names[i], v)
                elif isinstance(v, np.integer):
                    prompt += "%s=%d, " % (feature_names[i], v)
                else:
                    prompt += "%s=%s, " % (feature_names[i], v)
            prompt += end

            if not label:
                return prompt
            else:
                completion = "%s" % row['y'] #target_names[int(row['y'])]
                return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
        else:
            prompt = "When we have " 
            for i in range(len(row)-label):
                v = row[i]
                if isinstance(v, np.float):
                    prompt += "x%d=%.4f, " % (i+1, v) 
                elif isinstance(v, np.integer):
                    prompt += "x%d=%d, " % (i+1, v)
                else:
                    prompt += "x%d=%s, " % (i+1, v)
            prompt += "what should be the y value?"

            if not label:
                return f"{prompt}###"
            else:
                completion = "%s" % str(row['y']) #row['y']
                return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)

    def df2jsonl_openml(self, df, filename, integer = False, 
                context = False, feature_names = None, target_names = None, init = '', end = ''):
        jsonl = '\n'.join(df.apply(func = partial(self.data2text_openml, 
                                                integer = integer, 
                                                context = context, 
                                                feature_names = feature_names, 
                                                target_names = target_names, 
                                                init = init, 
                                                end = end), axis = 1).tolist())

        with open(filename, 'w') as f:
            f.write(jsonl)
        return filename

    def array2prompts_openml(self, X, integer = False,
				 context = False, feature_names = None, target_names = None, init = '', end = ''):
        return list(map(partial(self.data2text_openml, 
                                integer = integer, 
                                label = False,
                                context = context, 
                                feature_names = feature_names, 
                                target_names = target_names, 
                                init = init, 
                                end = end
                            ), X))
    
    def data_split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        # n = X.shape[0]
        # idx = np.arange(n)
        # random.shuffle(idx)
        # train_idx, valid_idx, test_idx = idx[:int(.6*n)], idx[int(.6*n):int(.8*n)], idx[int(.8*n):]
        # X_train, X_valid, X_test = X[train_idx], X[valid_idx], X[test_idx]
        # y_train, y_valid, y_test = y[train_idx], y[valid_idx], y[test_idx]
        return X_train, X_valid, X_test, y_train, y_valid, y_test

    def preprocess_data(self, data,  normalized=False, corruption_level=0, outliers=None):
        X, y = data['data'], data['target']
        if normalized:
            X = self.scaler.fit_transform(X)
            
        X_train, X_valid, X_test, y_train, y_valid, y_test = self.data_split(X, y)
        if outliers is not None:
            X_out, y_out = outliers
            X_train = np.concatenate([X_train, X_out], axis = 0)
            y_train = np.concatenate([y_train, y_out], axis = 0)
        if corruption_level > 0:
            # corrupt here
            n = len(y_train)
            m = int(n * corruption_level)
            inds = random.sample(range(1, n), m)
            for i in inds:
                y_train[i] = 1 - y_train[i] #binary

        train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], val_df['y'], test_df['y'] = y_train, y_valid, y_test   

        return train_df, val_df, test_df

    def prepare_prompts(self, did, dfs, context=False, init=None, end=None, feature_names=None, target_names=None):
        X_test = dfs['test'].values[:, :-1]
        X_valid = dfs['val'].values[:, :-1]
        jsonl_files = {}
        # if not os.path.isdir(os.path.join(self.data_dir, 'openml_datasets')):
        #     os.mkdir(os.path.join(self.data_dir, 'openml_datasets'))
        for mode in ['train', 'val']:
            fname = os.path.join(self.data_dir, f'openml_%d_context_%s_%s.jsonl' % (did, context, mode))
            jsonl_files[mode] = self.df2jsonl_openml(dfs[mode], fname,
                        context = context, 
                        feature_names = feature_names, 
                        target_names = target_names, 
                        init = init, 
                        end = end)
        test_prompts = self.array2prompts_openml(X_test,
            context = context, 
            feature_names = feature_names,
            target_names = target_names, 
            init = init, 
            end = end)
        valid_prompts = self.array2prompts_openml(X_valid,
            context = context, 
            feature_names = feature_names,
            target_names = target_names, 
            init = init, 
            end = end)
        return jsonl_files, test_prompts, valid_prompts
    
    def load_openml_datasets(self, did, use_name=False):
        X, y, att_names, target_name = self.load_openml(did, normalize = False)
        train_df, val_df, test_df = self.preprocess_data({'data': X, 'target': y}, normalized=False)
        dfs = {'train': train_df, 'val': val_df, 'test': test_df}
        if use_name:
            init = 'When we have '
            end = 'what should be the %s?###' % target_name
            jsonl_files, test_prompts, valid_prompts = self.prepare_prompts(did, dfs, context=True, init=init, end=end, feature_names=att_names, target_names=target_name)
        else:
            jsonl_files, test_prompts, valid_prompts = self.prepare_prompts(did, dfs)
        return train_df, val_df, test_df, test_prompts, valid_prompts, None, None, None, jsonl_files['train'], jsonl_files['val']

def generate_data(data_dir, func, n_train, n_valid, n_test, p, integer = False, 
                 noise_level = 0, test_int = None, outliers = None,
               lb = -10, ub = 10, grid_int = None, resolution = 100, 
               beta = None, donut = False
              ):
    """
    Parameters
    ------------
    func: the function name, which could be ‘linear’, ‘quadratic’, ‘exponential’, ‘cosine’, ‘l1norm’, and ‘piecewise’
    n_train: number of training samples
    n_valid: number of validation samples
    n_test: number of testing samples
    p: the number of features
    integer: whether you want to round all the values
    noise_level: the standard deviation of Gaussian distribution
    test_int: the range/interval of the testing data distribution (for each dimension), default to be as the same as the training interval
    outliers: (X, y), where X here should be a 2D numpy.array with shape m*p, where the y here should be a 2D numpy.array with shape m*1
    lb: lower bound of the training dsitribution
    ub: upper bound of the training distribution
    grid_int: the interval of producing grid dataset
    resolution: the number of grid data points for each dimension 
    beta: the coefficient of linear function
    """
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    dataGen = dataGenerate(data_dir)
    return dataGen.generate(
        func, n_train, n_valid, n_test, p, integer, noise_level, test_int, outliers,
               lb, ub, grid_int, resolution, beta, donut
    )

def load_openml_data(data_dir, did, use_name = False):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    dataLoad = dataOpenMLLoad(data_dir)
    return dataLoad.load_openml_datasets(did, use_name = use_name)