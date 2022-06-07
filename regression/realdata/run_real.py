import sys, os, openai, json
sys.path.insert(1, '../utils')
from GPT3FineTuner import GPT3FineTuner
from GPTJFineTuner import GPTJFineTuner
import numpy as np
import pandas as pd

def run_setting_gpt3(data_dir, n_sims = 3, num_epochs = 10, batch_size = 5, 
                        data_list = ['servo', 'CCPP', 'insurance'], 
                     lr_list = [0.05, 0.1, 0.2],
                     prefix_list = ['_', '_fn_'],
                     pc_list = ['20', '40', '60', '80', 'full'],
    openai_key = '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',
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
    
    for data in data_list:
        for prefix in prefix_list:
            data_prefix = data.lower()+prefix
            for pc in pc_list:
                print("------------------Runing group %d---------------------"%counter)
                print("%s%s"%(data_prefix,pc))
                train_df = pd.read_csv("data/%s/%s_train_%s.csv" % (data, data.lower(),pc))
                valid_df = pd.read_csv("data/%s/%s_valid.csv" % (data, data.lower()))
                test_df = pd.read_csv("data/%s/%s_test.csv" % (data, data.lower()))
                cols = train_df.columns.tolist()
                cols[-1] = 'y'
                cols[:-1] = list(range(len(cols) - 1))
                train_df.columns = cols
                valid_df.columns = cols
                test_df.columns = cols

                train_file = "data/%s/%s%s_train.jsonl" % (data, data_prefix, pc)
                valid_file = "data/%s/%svalid.jsonl" % (data, data_prefix)

                test_prompts = []
                with open('data/%s/%stest.jsonl' % (data, data_prefix), 'r') as fp:
                    for line in fp:
                        json_obj = json.loads(line)
                        test_prompts.append(json_obj['prompt'])

                valid_prompts = []
                with open('data/%s/%svalid.jsonl' % (data, data_prefix), 'r') as fp:
                    for line in fp:
                        json_obj = json.loads(line)
                        valid_prompts.append(json_obj['prompt'])
                # select hyperparameter for each experiment
                config['lr'] = lr_list
                for sim_idx in range(n_sims):
                    print('---Simulation %d---' % (sim_idx+1))
                    data_sim_dir = '%s/data_%d' % (data_dir, sim_idx+1)
                    gpt3_fine_tuner = GPT3FineTuner(config=config,train_jsonl=train_file,valid_jsonl=valid_file, openai_key=openai_key)
                    gpt3_fine_tuner.fine_tune()

                    plot_save_path = valid_file.split('valid.')[0]+".png"
                    file_name = valid_file.split('valid.')[0].replace(",","").replace("(","").replace(")","")+'ft_info.json'
                    y_test_outputs,y_grid_outputs,_,_,_ = gpt3_fine_tuner.eval(test_prompts=test_prompts,n_train=len(train_df),test_df=test_df,training_csv_file_name = file_name,valid_df = valid_df,valid_prompts = valid_prompts,plot=False,file_name=plot_save_path,train_df = train_df,valid_temperature=valid_temperature, y_name = train_df.columns[-1])
                    if sim_idx == 0: config['lr'] = [lr_list[gpt3_fine_tuner.best_idx]]
                    # save fine-tuned info and results
                    with open('%s/data_%d/%s%s_ft_info.json' % (data_dir, sim_idx+1, data, pc), 'w') as fp:
                        json.dump(openai.FineTune.retrieve(id=gpt3_fine_tuner.ft_id).copy(), fp, indent=4)

                    p = -1
                    try: 
                        with open('%s/data_%d/%s%s_all.json' % (data_dir, sim_idx+1, data_prefix, pc),'r') as fp:
                            tr_ts_vl_json = json.load(fp)
                            tr_ts_vl_json['gpt3_test_y'] = y_test_outputs
                            tr_ts_vl_json['openai_key'] = openai_key
                            tr_ts_vl_json['ft_id'] = gpt3_fine_tuner.ft_id
                            tr_ts_vl_json['model_id'] = gpt3_fine_tuner.ft_info
                    except:
                        tr_ts_vl_json = {"train_x":train_df[train_df.columns[:p]].values.tolist(),"train_y":list(train_df['y']),"validation_x":valid_df[valid_df.columns[:p]].values.tolist(),"validation_y":list(valid_df['y']),
                                            "test_x":test_df[test_df.columns[:p]].values.tolist(),"test_y":list(test_df['y']),"gpt3_test_y":y_test_outputs, 'openai_key': openai_key, 'ft_id': gpt3_fine_tuner.ft_id,'model_id':gpt3_fine_tuner.ft_info}
                    with open('%s/data_%d/%s%s_all.json' % (data_dir, sim_idx+1, data_prefix, pc),'w') as fp:
                        json.dump(tr_ts_vl_json,fp)
            
def run_setting_gptj(data_dir, n_sims = 3, 
                        data_list = ['servo', 'CCPP', 'insurance'], 
                     lr_list = [0.05, 0.1, 0.2],
                     prefix_list = ['_', '_fn_'],
                     pc_list = ['20', '40', '60', '80', 'full'],
                     epochs =  [2,6,10],
                     cuda_idx = 0,
                     batch_size = 4
                    ):
    config = {'learning_rate': 1e-4, 'batch_size': batch_size, 'epochs':epochs,  'weight_decay': 0.01, 'warmup_steps': 6}
    for sim_idx in range(n_sims):
        if not os.path.isdir(data_dir):
            os.mkdir(data_dir)
        data_sim_dir = '%s/data_%d' % (data_dir, sim_idx+1)
        if not os.path.isdir(data_sim_dir):
            os.mkdir(data_sim_dir)

    counter = 0
    
    for data in data_list:
        for prefix in prefix_list:
            data_prefix = data.lower()+prefix
            for pc in pc_list:
                print("------------------Runing group %d---------------------"%counter)
                print("%s%s"%(data_prefix,pc))
                train_df = pd.read_csv("data/%s/%s_train_%s.csv" % (data, data.lower(),pc))
                valid_df = pd.read_csv("data/%s/%s_valid.csv" % (data, data.lower()))
                test_df = pd.read_csv("data/%s/%s_test.csv" % (data, data.lower()))
                cols = train_df.columns.tolist()
                cols[-1] = 'y'
                cols[:-1] = list(range(len(cols) - 1))
                train_df.columns = cols
                valid_df.columns = cols
                test_df.columns = cols

                train_file = "data/%s/%s%s_train.jsonl" % (data, data_prefix, pc)
                valid_file = "data/%s/%svalid.jsonl" % (data, data_prefix)

                test_prompts = []
                with open('data/%s/%stest.jsonl' % (data, data_prefix), 'r') as fp:
                    for line in fp:
                        json_obj = json.loads(line)
                        test_prompts.append(json_obj['prompt'])

                valid_prompts = []
                with open('data/%s/%svalid.jsonl' % (data, data_prefix), 'r') as fp:
                    for line in fp:
                        json_obj = json.loads(line)
                        valid_prompts.append(json_obj['prompt'])
                # select hyperparameter for each experiment
                config['lr'] = lr_list
                for sim_idx in range(n_sims):
                    print('---Simulation %d---' % (sim_idx+1))
                    data_sim_dir = '%s/data_%d' % (data_dir, sim_idx+1)
                    gptj_fine_tuner = GPTJFineTuner(config=config,train_jsonl=train_file,valid_jsonl=valid_file,cuda_idx=cuda_idx)
                    gptj_fine_tuner.fine_tune()

                    y_test_outputs,_,_,_,_ = gptj_fine_tuner.eval(
                                test_prompts = test_prompts, 
                                n_train = len(train_df), 
                                test_df = test_df, 
                                valid_df = valid_df, 
                                valid_prompts = valid_prompts, 
                                plot = False, 
                                train_df = train_df
                    )
                    if sim_idx == 0: config['epochs'] = [epochs[gptj_fine_tuner.best_idx]]
                    # save fine-tuned info and results
                    try:
                        with open('%s/data_%d/%s%s_all.json' % (data_dir, sim_idx+1, data_prefix, pc),'r') as fp:
                            tr_ts_vl_json = json.load(fp)
                        tr_ts_vl_json['gptj_test_y'] = y_test_outputs
                        with open('%s/data_%d/%s%s_all.json' % (data_dir, sim_idx+1, data_prefix, pc),'w') as fp:
                            json.dump(tr_ts_vl_json,fp)
                    except:
                        p = -1
                        tr_ts_vl_json = {"train_x":train_df[train_df.columns[:p]].values.tolist(),"train_y":list(train_df['y']),"validation_x":valid_df[valid_df.columns[:p]].values.tolist(),"validation_y":list(valid_df['y']),
                                            "test_x":test_df[test_df.columns[:p]].values.tolist(),"test_y":list(test_df['y']),"gpt3_test_y":y_test_outputs, 'openai_key': openai_key, 'ft_id': gpt3_fine_tuner.ft_id,'model_id':gpt3_fine_tuner.ft_info}
                        with open('%s/data_%d/%s%s_all.json' % (data_dir, sim_idx+1, data_prefix, pc),'w') as fp:
                            json.dump(tr_ts_vl_json,fp)
            
