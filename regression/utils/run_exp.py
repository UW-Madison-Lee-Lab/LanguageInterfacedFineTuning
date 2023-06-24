from tqdm import tqdm
from resultsCollector import regressionLoss
from GPT3FineTuner import GPT3FineTuner
from regressionDataGen import generate_data, load_openml_data, dataGenerate
import os, json, time, openai, sys
import numpy as np
from copy import deepcopy
try:
    from GPTJFineTuner import GPTJFineTuner
except ModuleNotFoundError:
    print('GPTJ loading failed!')
import pandas as pd

lr_out = (np.array([[-6.5], [-2], [-.5], [9]]), np.array([[7], [3], [4], [100]]))
qr_out = (np.array([[-6.5], [-2], [-.5], [9]]), np.array([[7], [3], [4], [100]]))
exp_out = (np.array([[-6.5], [-2], [-.5], [9]]), np.array([[7], [3], [4], [100]]))
cos_out = (np.array([[-6.5], [-2], [-.5], [9]]), np.array([[7], [3], [4], [100]]))
l1_out = (np.array([[-6.5], [-2], [-.5], [9]]), np.array([[7], [3], [4], [100]]))
pw_out = (np.array([[-6.5], [-2], [-.5], [9]]), np.array([[7], [3], [4], [100]]))

func_list = ['linear', 'quadratic', 'exponential', 'cosine', 'l1norm', 'piecewise']
outliers_list = [lr_out, qr_out, exp_out, cos_out, l1_out, pw_out]

def run_setting_gpt3(data_dir, n_sims = 3, n_train = 200, n_valid = 50, n_test = 100, data_list = ['linear'], p_list = [1,50,120], integer_list = [False], lb_ub_list = [(-10,10)],
    noise_level_list = [0.1], beta = None, resolution = 200, num_epochs = 10, batch_size = 5, grid_int = None, donut = False, outliers = None, use_name_list = [False], lr_list = [0.05, 0.1, 0.2],
    openai_key = [replace it with your openai key],
    valid_temperature = 0.75):
    openai.api_key = openai_key
    for sim_idx in range(n_sims):
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        data_sim_dir = '%s/data_%d' % (data_dir, sim_idx+1)
        if not os.path.isdir(data_sim_dir):
            os.mkdir(data_sim_dir)

    config = {'model_type':'ada',"num_epochs":num_epochs,"batch_size":batch_size, "lr": lr_list}

    counter = 0
    for dataset in data_list:
        if isinstance(dataset, int):
            lb_ub_list_ = [(None, None)]
            integer_list_ = [None]
            p_list_ = [None]
            noise_level_list_ = [None]
            use_name_list_ = use_name_list
        else:
            lb_ub_list_, integer_list_, p_list_,  noise_level_list_ = lb_ub_list, integer_list, p_list, noise_level_list
            use_name_list_ = [False]
        for lb,ub in lb_ub_list_:
            for integer in integer_list_:
                for p in p_list_:
                    for noise_level in noise_level_list_:
                        for use_name in use_name_list_:
                            counter += 1
                            print("------------------Runing group %d---------------------"%counter)
                            print("p=%s,dataset=%s,integer=%s,range=[%s,%s]"%(p,dataset,integer,lb,ub))
                            config['lr'] = lr_list
                            for sim_idx in range(n_sims):
                                print('---Simulation %d---' % (sim_idx+1))
                                data_sim_dir = '%s/data_%d' % (data_dir, sim_idx+1)
                                # run exps
                                if isinstance(dataset, str):
                                    train_df, valid_df, test_df, test_prompts, valid_prompts, grid_prompts,X_grid,y_grid,train_file,valid_file = generate_data(data_dir=data_sim_dir,\
                                        func=dataset,n_train=n_train,n_test=n_test,n_valid=n_valid,p=p,integer=integer,noise_level=noise_level,\
                                            lb=lb, ub =ub,resolution=resolution, beta=beta, grid_int=grid_int, donut = donut, outliers = outliers)
                                else:
                                    train_df, valid_df, test_df, test_prompts, valid_prompts, grid_prompts,X_grid,y_grid,train_file,valid_file = load_openml_data(data_sim_dir, dataset, use_name)
                                
                                print("train file saved at: "+train_file)
                                print("validation file saved at: "+valid_file)
                                
                                gpt3_fine_tuner = GPT3FineTuner(config=config,train_jsonl=train_file,valid_jsonl=valid_file, openai_key=openai_key)
                                gpt3_fine_tuner.fine_tune()

                                plot_save_path = valid_file.split('valid.')[0]+".png"
                                file_name = valid_file.split('valid.')[0].replace(",","").replace("(","").replace(")","")+'ft_info.json'
                                y_test_outputs,y_grid_outputs,_,_,_ = gpt3_fine_tuner.eval(test_prompts=test_prompts,n_train=n_train,test_df=test_df,training_csv_file_name = file_name,valid_df = valid_df,valid_prompts = valid_prompts,plot=True,X_grid=X_grid,grid_prompts=grid_prompts,y_grid=y_grid,file_name=plot_save_path,train_df = train_df,valid_temperature=valid_temperature)
                                if sim_idx == 0: config['lr'] = [lr_list[gpt3_fine_tuner.best_idx]]
                                # save fine-tuned info and results
                                with open(valid_file.split('valid.')[0]+'ft_info.json', 'w') as fp:
                                    json.dump(openai.FineTune.retrieve(id=gpt3_fine_tuner.ft_id).copy(), fp, indent=4)

                                if isinstance(dataset, str) and (p == 1 or p == 2):
                                    tr_ts_vl_json = {"train_x":train_df[train_df.columns[:p]].values.tolist(),"train_y":list(train_df['y']),"validation_x":valid_df[valid_df.columns[:p]].values.tolist(),"validation_y":list(valid_df['y']),
                                                        "test_x":test_df[test_df.columns[:p]].values.tolist(),"test_y":list(test_df['y']),"gpt3_test_y":y_test_outputs,"grid_x":X_grid.tolist(),"grid_y":y_grid.tolist(),'gpt3_grid_y':y_grid_outputs, 'openai_key': openai_key, 'ft_id': gpt3_fine_tuner.ft_id,'model_id':gpt3_fine_tuner.ft_info}
                                else:
                                    tr_ts_vl_json = {"train_x":train_df[train_df.columns[:p]].values.tolist(),"train_y":list(train_df['y']),"validation_x":valid_df[valid_df.columns[:p]].values.tolist(),"validation_y":list(valid_df['y']),
                                                        "test_x":test_df[test_df.columns[:p]].values.tolist(),"test_y":list(test_df['y']),"gpt3_test_y":y_test_outputs, 'openai_key': openai_key, 'ft_id': gpt3_fine_tuner.ft_id,'model_id':gpt3_fine_tuner.ft_info}
                                with open(valid_file.split('valid.')[0]+'all.json','w') as fp:
                                    json.dump(tr_ts_vl_json,fp)
        
def run_setting_gptj(data_dir, n_sims = 3, n_train = 200, n_valid = 50, n_test = 100, data_list = ['linear'], p_list = [1,50,120], integer_list = [False], lb_ub_list = [(-10,10)],
    noise_level_list = [0.1], epochs = [2,6,10], cuda_idx = 0, metric = 'RAE', batch_size = 4):
    config = {'learning_rate': 1e-4, 'batch_size': batch_size, 'epochs':epochs,  'weight_decay': 0.01, 'warmup_steps': 6}
    dg = dataGenerate('data')
    counter = 0
    for func in data_list:
        for noise in noise_level_list:
            for p in p_list:
                for integer in integer_list:
                    for lb, ub in lb_ub_list:
                        counter += 1
                        print("------------------Runing group %d---------------------"%counter)
                        print("p=%s,dataset=%s,integer=%s,range=[%s,%s]"%(p,func,integer,lb,ub))
                        config['epochs'] = epochs
                        for sim_idx in range(n_sims):
                            print('---Simulation %d---' % (sim_idx+1))
                            with open('%s/data_%d/%s_n_%s_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_all.json' % (data_dir, sim_idx+1, func, n_train+n_valid, p, integer, lb, ub, noise), 'r') as f:
                                data_json = json.load(f)
                                
                            train_df = pd.DataFrame(data_json['train_x'])
                            train_df['y'] = data_json['train_y']

                            valid_df = pd.DataFrame(data_json['validation_x'])
                            valid_df['y'] = data_json['validation_y']

                            test_df = pd.DataFrame(data_json['test_x'])
                            test_df['y'] = data_json['test_y']

                            test_prompts = dg.array2prompts(data_json['test_x'])
                            valid_prompts = dg.array2prompts(data_json['validation_x'])
                            if p in [1,2]:
                                grid_prompts = dg.array2prompts(data_json['grid_x'])
                                X_grid = np.array(data_json['grid_x'])
                                y_grid = np.array(data_json['grid_y'])
                            else:
                                grid_prompts, X_grid, y_grid = None, None, None

                            train_file = '%s/data_%d/%s_n_%s_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_train.jsonl' % (data_dir,sim_idx+1, func, n_train+n_valid, p, integer, lb, ub, noise)
                            valid_file = '%s/data_%d/%s_n_%s_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_valid.jsonl' % (data_dir,sim_idx+1, func, n_train+n_valid, p, integer, lb, ub, noise)
                            #valid_prompts, grid_prompts,X_grid,y_grid,train_file,valid_file 
                            gptj_fine_tuner = GPTJFineTuner(config=config,train_jsonl=train_file,valid_jsonl=valid_file,cuda_idx=cuda_idx)
                            gptj_fine_tuner.fine_tune()
                            gptj_test_y, gptj_grid_y, _, rmse, rmse_woo = gptj_fine_tuner.eval(test_prompts = test_prompts, 
                                n_train = n_train, 
                                test_df = test_df, 
                                valid_df = valid_df, 
                                valid_prompts = valid_prompts, 
                                plot = True, 
                                X_grid = X_grid,
                                grid_prompts = grid_prompts,
                                y_grid = y_grid,
                                train_df = train_df
                            )

                            if sim_idx == 0: config['epochs'] = [epochs[gptj_fine_tuner.best_idx]]
                            # save fine-tuned info and results

                            if p == 1 or p == 2:
                                data_json['gptj_test_y'] = gptj_test_y
                                data_json['gptj_grid_y'] = gptj_grid_y
                                data_json['gptj_grid_loss'], _ = regressionLoss(gptj_grid_y, y_grid, metric, True)
                            else:
                                data_json['gptj_test_y'] = gptj_test_y
                            data_json['gptj_loss'], _ = regressionLoss(gptj_test_y, data_json['test_y'], metric, True)
                            with open('%s/data_%d/%s_n_%s_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_all.json' % (data_dir, sim_idx+1, func, n_train+n_valid, p, integer, lb, ub, noise), 'w') as f:
                                json.dump(data_json,f)
                            del gptj_fine_tuner

def rerun_setting_gpt3(data_dir, n_sims = 3, n_train = 200, n_valid = 50, n_test = 100, data_list = ['linear'], p_list = [1,50,120], integer_list = [False], lb_ub_list = [(-10,10)],
    noise_level_list = [0.1], num_epochs = 10, lr_list = [0.05, 0.1, 0.2], openai_key = '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',
    valid_temperature = 0.75, model = 'ada', batch_size = 4):

    config = {'model_type':model,"num_epochs":num_epochs,"batch_size":batch_size, "lr": lr_list}
    openai.api_key = openai_key
    dg = dataGenerate('data')

    counter = 0
    for func in data_list:
        for noise in noise_level_list:
            for p in p_list:
                for integer in integer_list:
                    for lb, ub in lb_ub_list:
                        counter += 1
                        print("------------------Runing group %d---------------------"%counter)
                        print("p=%s,dataset=%s,integer=%s,range=[%s,%s]"%(p,func,integer,lb,ub))
                        config['lr'] = lr_list
                        for sim_idx in range(n_sims):
                            print('---Simulation %d---' % (sim_idx+1))
                            with open('%s/data_%d/%s_n_%s_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_all.json' % (data_dir, sim_idx+1, func, n_train+n_valid, p, integer, lb, ub, noise), 'r') as f:
                                data_json = json.load(f)
                                
                            train_df = pd.DataFrame(data_json['train_x'])
                            train_df['y'] = data_json['train_y']

                            valid_df = pd.DataFrame(data_json['validation_x'])
                            valid_df['y'] = data_json['validation_y']

                            test_df = pd.DataFrame(data_json['test_x'])
                            test_df['y'] = data_json['test_y']

                            test_prompts = dg.array2prompts(data_json['test_x'])
                            valid_prompts = dg.array2prompts(data_json['validation_x'])
                            if p in [1,2]:
                                grid_prompts = dg.array2prompts(data_json['grid_x'])
                                X_grid = np.array(data_json['grid_x'])
                                y_grid = np.array(data_json['grid_y'])
                            else:
                                grid_prompts, X_grid, y_grid = None, None, None

                            train_file = '%s/data_%d/%s_n_%s_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_train.jsonl' % (data_dir, sim_idx+1, func, n_train+n_valid, p, integer, lb, ub, noise)
                            valid_file = '%s/data_%d/%s_n_%s_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_valid.jsonl' % (data_dir, sim_idx+1, func, n_train+n_valid, p, integer, lb, ub, noise)
                            #valid_prompts, grid_prompts,X_grid,y_grid,train_file,valid_file 
                            gpt3_fine_tuner = GPT3FineTuner(config=config,train_jsonl=train_file,valid_jsonl=valid_file, openai_key=openai_key)
                            gpt3_fine_tuner.fine_tune()

                            plot_save_path = valid_file.split('valid.')[0]+".png"
                            file_name = valid_file.split('valid.')[0].replace(",","").replace("(","").replace(")","")+'ft_info.json'

                            gpt3_test_y,gpt3_grid_y,_,_,_ = gpt3_fine_tuner.eval(
                                test_prompts=test_prompts,
                                n_train=n_train,
                                test_df=test_df,
                                training_csv_file_name = file_name,
                                valid_df = valid_df,
                                valid_prompts = valid_prompts,
                                plot=True,
                                X_grid=X_grid,
                                grid_prompts=grid_prompts,
                                y_grid=y_grid,
                                file_name=plot_save_path,
                                train_df = train_df,
                                valid_temperature=valid_temperature)

                            if sim_idx == 0: config['lr'] = [lr_list[gpt3_fine_tuner.best_idx]]
                            # save fine-tuned info and results


                            with open('%s/data_%d/%s_n_%s_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_all.json' % (data_dir, sim_idx+1, func, n_train+n_valid, p, integer, lb, ub, noise), 'r') as f:
                                data_json = json.load(f) # if there are other experiments running at the same time, the file might have been overwritten!
                            if p == 1 or p == 2:
                                data_json['gpt3_test_y_%s' % model] = gpt3_test_y
                                data_json['gpt3_grid_y_%s' % model] = gpt3_grid_y
                            else:
                                data_json['gpt3_test_y_%s' % model] = gpt3_test_y
                            data_json['openai_key_%s' % model] = openai_key
                            data_json['ft_id_%s' % model] = gpt3_fine_tuner.ft_id
                            data_json['model_id_%s' % model] = gpt3_fine_tuner.ft_info
                            with open('%s/data_%d/%s_n_%s_p_%d_int_%d_(%.1f,%.1f)_noise_%.2f_all.json' % (data_dir, sim_idx+1, func, n_train+n_valid, p, integer, lb, ub, noise), 'w') as f:
                                json.dump(data_json,f)
