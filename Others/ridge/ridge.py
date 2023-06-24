import pandas as pd
import numpy as np
import os, random, itertools, sys, time, json, openai
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
sys.path.insert(1, '../../regression/utils')
from GPT3FineTuner import GPT3FineTuner
from GPTJFineTuner import GPTJFineTuner


class dataGenerate(object):
    """
    A class of functions for generating jsonl dataset. 
    """
    def __init__(self,data_dir):
        self.data_dir = data_dir

    def data2text(self, row, integer = False, mode = 'standard', label = True):
        prompt = "When we have " 
        for i in range(len(row)-label):
            if integer:
                prompt += "x%d=%d, " % (i+1, row[i])
            else:
                prompt += "x%d=%.4f, " % (i+1, row[i]) 
        if mode == 'standard':
            prompt += "what should be the y value?"
        elif mode == 'contradict':
            prompt += 'what is x1-x2?'
        elif mode == 'correct':
            prompt += 'what is x1+x2?'
        else:
            NotImplementedError('Mode %s is not supported! Please select between standard, contradict, and correct.' % mode)

        if not label:
            return "%s###" % prompt
        else:
            if integer:
                completion = "%d" % row['y']
            else:
                completion = "%.3f" % row['y']
            return "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
    
    def df2jsonl(self, df, filename, integer = False, mode = 'standard'):
        jsonl = '\n'.join(df.apply(func = partial(self.data2text, integer = integer, mode = mode), axis = 1).tolist())
        with open(os.path.join(self.data_dir, filename), 'w') as f:
            f.write(jsonl)
        print("Save a file at:",os.path.join(self.data_dir, filename))
        return os.path.join(self.data_dir, filename)
            
    def array2prompts(self, X, integer = False, mode = 'standard'):
        return list(map(partial(self.data2text, integer = integer, label = False, mode = mode), X))

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
    
    def linear(self, X, beta):
        return np.dot(X, beta) / len(beta)

    def generate(self, func, mode, n_train, n_valid, n_test, p, integer = False, 
                 noise_level = 0, test_int = None, 
               lb = -10, ub = 10,  beta = None, lbd = 0, grid_int = None, resolution = 200,
              ):
        """
        mode: "standard", "contridict", "correct"
        """
        np.random.seed(123)
        if beta is None: beta = np.ones(p) 
        if test_int is None: test_int = (lb, ub)
        if grid_int is None: grid_int = (lb, ub)
        
        # Generate x and y   
        n = n_train + n_valid
        X = self.X_generate(n, p, integer, lb, ub)
        X_test = self.X_generate(n_test, p, integer, test_int[0], test_int[1])

        if p == 1:
            X_grid, grid_prompts = self.gridX_generate(grid_int, p, integer, resolution)
            output_file = open(os.path.join(self.data_dir,'%s_n_%d_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_grid_prompts.txt')%(func,n,p,integer,lb,ub,noise_level), 'w')
            for ele in grid_prompts:
                output_file.write(ele + '\n')
            output_file.close()
        
        y_true = self.linear(X, beta)
        y_test = self.linear(X_test, beta)
        if p == 1: y_grid = self.linear(X_grid, beta)
        
        y = (y_true + np.random.normal(0,noise_level*np.std(y_true),n)).reshape(n,1)

        if integer: y = y.round()
        
        # split into train, valid, test dataset
        X_train, X_valid, y_train, y_valid = self.data_split(X, y, n_train, n_valid)
        X_train = np.concatenate([X_train, lbd * np.diag(np.ones(p))])
        y_train = np.concatenate([y_train, np.zeros(p).reshape(p,1)])
        train_df, valid_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_valid), pd.DataFrame(X_test)
        train_df['y'], valid_df['y'], test_df['y'] = y_train, y_valid, y_test
        
        train_file = self.df2jsonl(train_df, '%s_lbd_%d_n_%d_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_train.jsonl'%(func, lbd, n,p,integer,lb,ub,noise_level), integer, mode)
        valid_file = self.df2jsonl(valid_df, '%s_lbd_%d_n_%d_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_valid.jsonl'%(func, lbd, n,p,integer,lb,ub,noise_level), integer, mode)
        valid_prompts = self.array2prompts(X_valid, integer = integer, mode = mode)
        test_prompts = self.array2prompts(X_test, integer = integer, mode = mode)
            
        if p == 1:
            return train_df, valid_df, test_df, test_prompts, valid_prompts, train_file, valid_file, grid_prompts, X_grid, y_grid
        else:
            return train_df, valid_df, test_df, test_prompts, valid_prompts, train_file, valid_file, None, None, None

def generate_data(data_dir, lbd, n_train, n_valid, n_test, p = 1,
                 noise_level = 0, test_int = None, 
               lb = -10, ub = 10,  beta = None
              ):
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    dataGen = dataGenerate(data_dir)
    return dataGen.generate('linear', 'standard', n_train, n_valid, n_test, p, False, noise_level, test_int, lb, ub, beta, lbd = lbd)

def run_setting_gpt3(data_dir, lbd_list = [0, 10, 50, 100, 1000], n_train = 200, n_valid = 50, n_test = 100, p_list = [1,10,50,100],
    num_epochs = 10, batch_size = 5, lr_list = [0.05, 0.1, 0.2], openai_key = [replace it with your openai key]):
    openai.api_key = openai_key
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    config = {'model_type':'ada',"num_epochs":num_epochs,"batch_size":batch_size, 'lr':lr_list}

    counter = 1
    

    # run exps
    for lbd in lbd_list:
        for p in p_list:
            print("------------------Runing group %d---------------------"%counter)
            counter += 1
            train_df, valid_df, test_df, test_prompts, valid_prompts,train_file,valid_file, grid_prompts, X_grid, y_grid = generate_data(data_dir=data_dir,\
            n_train=n_train,n_test=n_test,n_valid=n_valid, p = p, lbd = lbd)
            
            print("train file saved at: "+train_file)
            print("validation file saved at: "+valid_file)
            
            gpt3_fine_tuner = GPT3FineTuner(config=config,train_jsonl=train_file,valid_jsonl=valid_file, openai_key = openai_key)

            gpt3_fine_tuner.fine_tune()

            file_name = valid_file.split('valid.')[0].replace(",","").replace("(","").replace(")","")+'ft_info.json'            
            y_test_outputs,y_grid_outputs,_,_,_ = gpt3_fine_tuner.eval(test_prompts=test_prompts,n_train=n_train,test_df=test_df,training_csv_file_name = file_name, valid_df = valid_df, valid_prompts = valid_prompts, plot = True, X_grid=X_grid,grid_prompts=grid_prompts,y_grid=y_grid,train_df = train_df)

            # save fine-tuned info and results
            with open(valid_file.split('valid.')[0]+'ft_info.json', 'w') as fp:
                json.dump(openai.FineTune.retrieve(id=gpt3_fine_tuner.ft_id).copy(), fp, indent=4)
            if p == 1:
                tr_ts_vl_json = {"train_x":train_df[train_df.columns[:p]].values.tolist(),"train_y":list(train_df['y']),"validation_x":valid_df[valid_df.columns[:p]].values.tolist(),"validation_y":list(valid_df['y']),
                                    "test_x":test_df[test_df.columns[:p]].values.tolist(),"test_y":list(test_df['y']),"gpt3_test_y":y_test_outputs,"grid_x":X_grid.tolist(),"grid_y":y_grid.tolist(),'gpt3_grid_y':y_grid_outputs, 'openai_key': openai_key, 'ft_id': gpt3_fine_tuner.ft_id,'model_id':gpt3_fine_tuner.ft_info}
            else:
                tr_ts_vl_json = {"train_x":train_df[train_df.columns[:p]].values.tolist(),"train_y":list(train_df['y']),"validation_x":valid_df[valid_df.columns[:p]].values.tolist(),"validation_y":list(valid_df['y']),
                                    "test_x":test_df[test_df.columns[:p]].values.tolist(),"test_y":list(test_df['y']),"gpt3_test_y":y_test_outputs, 'openai_key': openai_key, 'ft_id': gpt3_fine_tuner.ft_id,'model_id':gpt3_fine_tuner.ft_info}
            with open(valid_file.split('valid.')[0]+'all.json','w') as fp:
                json.dump(tr_ts_vl_json,fp)

def run_setting_gptj(data_dir_list, cuda_idx = 0, epochs = [2,6,10], batch_size = 4):

    p = 2
    counter = 1
    dg = dataGenerate('data')
    # run exps
    for data_dir in data_dir_list:
        for file in os.listdir(data_dir):
            if file.endswith('all.json'):
                print("------------------Runing group %d---------------------"%counter)
                counter += 1
            
                with open('%s/%s' % (data_dir, file), 'r') as f:
                    data_json = json.load(f)
                train_df = pd.DataFrame(data_json['train_x'])
                train_df['y'] = data_json['train_y']
                valid_df = pd.DataFrame(data_json['validation_x'])
                valid_df['y'] = data_json['validation_y']
                test_df = pd.DataFrame(data_json['test_x'])
                test_df['y'] = data_json['test_y']
                
                test_prompts = dg.array2prompts(np.array(data_json['test_x']))
                valid_prompts = dg.array2prompts(data_json['validation_x'])
                
                train_file = '%s/%s' % (data_dir, file.split('all.json')[0] + 'train.jsonl')
                valid_file = '%s/%s' % (data_dir, file.split('all.json')[0] + 'valid.jsonl')
                
                n_train = int(file.split('_n_')[1].split('_')[0]) - 50
                p = int(file.split('_p_')[1].split('_')[0])
                
                if p == 1:
                    config = {'learning_rate': 1e-4, 'batch_size': batch_size, 'epochs':epochs,  'weight_decay': 0.01, 'warmup_steps': 6}
                elif p < 100:
                    config = {'learning_rate': 1e-4, 'batch_size': 1, 'epochs':epochs,  'weight_decay': 0.01, 'warmup_steps': 6}
                else:
                    continue
                
                gptj_fine_tuner = GPTJFineTuner(config=config,train_jsonl=train_file,valid_jsonl=valid_file,cuda_idx=cuda_idx)
                gptj_fine_tuner.fine_tune()
                gptj_test_y, _, _, rmse, rmse_woo = gptj_fine_tuner.eval(test_prompts = test_prompts, 
                    n_train = n_train, 
                    test_df = test_df, 
                    valid_df = valid_df, 
                    valid_prompts = valid_prompts, 
                    plot = False, 
                    X_grid = None,
                    grid_prompts = None,
                    y_grid = None,
                    train_df = train_df
                )

                try:
                    with open(valid_file.split('valid.')[0]+'all.json','r') as fp:
                        tr_ts_vl_json = json.load(fp)
                    
                    tr_ts_vl_json['gptj_test_y'] = gptj_test_y
                except:
                    tr_ts_vl_json = {"train_x":train_df[train_df.columns[:p]].values.tolist(),"train_y":list(train_df['y']),"validation_x":valid_df[valid_df.columns[:p]].values.tolist(),"validation_y":list(valid_df['y']),
                    "test_x":test_df[test_df.columns[:p]].values.tolist(),"test_y":list(test_df['y']),"gptj_test_y":gptj_test_y}
                with open(valid_file.split('valid.')[0]+'all.json','w') as fp:
                    json.dump(tr_ts_vl_json,fp)
                
