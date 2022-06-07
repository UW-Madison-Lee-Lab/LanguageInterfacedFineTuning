import sys
sys.path.append('./')
sys.path.append('./../')
from re import A
import pandas as pd
import numpy as np
import openai
import argparse
import torch
import time, json, os

from utils.classification_data_generator import df2jsonl,array2prompts
from utils.classificationDataGen import dataGenerate
from utils.helper import log
import utils.configs as cfgs
from models.GPTJFineTuner import GPTJFineTuner
from models.baselines import clf_model_teachers
from run_exps_helper import *
from utils.train_nn_generator import DataNN
from utils.helper import add_ending_symbol
from models import lora_gptj
from plotters.plotter import plot_decision_boundry_custom_color


pid = os.getpid()
print("pid:",pid) 

parser = argparse.ArgumentParser(description='GPTJ')
parser.add_argument("-b","--batch_size", default=4, type=int)
parser.add_argument("-c", "--corrupted", default=0, type=float)
parser.add_argument("-d","--data_id", default=-1, type=int)
parser.add_argument("-e","--subset", type=str,default='none',choices=['in_context','fraction','none'])
parser.add_argument("-f","--use_feature_name", action='store_true',help='replace feature indices with feature names')
parser.add_argument("-g","--gpu_id", default=0, type=int)
parser.add_argument('-i',"--run_idx", default=-1, type=int,choices=[-1,0,1,2],help='index of the split of dataset, -1 then run on all 3 splits')
parser.add_argument("-k","--label_corruption", type=str,default='none',choices=['random','system','none'])
parser.add_argument("-l","--lr", default=0.1, type=float,help="learning rate multiplier of GPT3, or learning rate of GPTJ")
parser.add_argument("-m", "--mixup", action='store_true')
parser.add_argument('-n',"--in_context", action='store_true')
parser.add_argument("-o", "--openai_key", default='', type=str)
parser.add_argument("-p", "--epochs", default=10, type=int)
parser.add_argument("-r","--subset_fraction", default=1, type=float)
parser.add_argument("-t", "--task", default='accuracy', type=str)


args = parser.parse_args()

CUDA_ID = args.gpu_id
LABEL_NOISES = [0.05, 0.1, 0.2]
if args.in_context:
    CONFIG = {'learning_rate': args.lr, 'batch_size': args.batch_size, 'epochs':[0], 'weight_decay': 0.01, 'warmup_steps': 6}
else:
    CONFIG = {'learning_rate': args.lr, 'batch_size': args.batch_size, 'epochs':[args.epochs], 'weight_decay': 0.01, 'warmup_steps': 6}

if args.in_context or args.subset=='in_context':
    NUM_PROMPTS = cfgs.in_context_num_prompts[args.data_id]
    print("NUM_PROMPTS",NUM_PROMPTS)
    
if args.subset=='fraction':
    NUM_PROMPTS = int(cfgs.sample_size[args.data_id]*0.8*0.8*args.subset_fraction)
    print("NUM_PROMPTS",NUM_PROMPTS)

######## Data
done_keys = []
data_ids = [args.data_id] if args.data_id > -1 else cfgs.openml_data_ids.keys()
lst_ids = []
for k in data_ids:
    if k not in done_keys:
        lst_ids.append(k)
data_ids = np.sort(lst_ids)

# set up saving path
log_fpath = f"results/evals/run_gptj_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}.txt"
logf = open(log_fpath, 'a+')


if args.run_idx == -1:
    start_idx = 0
    end_idx = 3
else:
    start_idx = args.run_idx
    end_idx = args.run_idx + 1


if args.task == cfgs.ACCURACY:
    for data_id in data_ids:
        for run_idx in range(start_idx,end_idx):
            csv_fpath = f"results/csv/run_gptj_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}_did_{args.data_id}_runidx_{args.run_idx}.csv"
            # load dataset
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(int(data_id), run_idx,mixup=args.mixup)
            train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
            train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
            jsonl_files = load_jsonl(data_id, run_idx,False,args.use_feature_name) # load w/ or w/o feature names

            # Subset
            if args.subset != 'none' and args.subset_fraction != 1:
                # load dataset 
                jsonl_files['train'] = extract_subset(jsonl_files['train'],NUM_PROMPTS,run_idx)
                
            # In Context
            in_context_prefix = ''
            if args.in_context:
                val_prompts = extract_random_incontext_prompts([jsonl_files['train']],NUM_PROMPTS,jsonl_files['val'],random_state=run_idx)
                test_prompts = extract_random_incontext_prompts([jsonl_files['train']],NUM_PROMPTS,jsonl_files['test'],random_state=run_idx)
            else:
                val_prompts = extract_prompts(jsonl_files['val'],in_context_prefix)
                test_prompts = extract_prompts(jsonl_files['test'],in_context_prefix)
            

            print("Training and Validation jsonl:", jsonl_files)
            print("===============config==============")
            print(CONFIG)
            gptj_fine_tuner = GPTJFineTuner(config=CONFIG,train_jsonl=jsonl_files['train'],valid_jsonl=jsonl_files['val'],cuda_idx=CUDA_ID)
            gptj_fine_tuner.fine_tune()
            log(logf, f"==== DID {data_id} {run_idx}_th run===")            
            gptj_best_test_y, val_valid_y_num_list,val_acc_list, test_acc_list = gptj_fine_tuner.eval(
                    valid_prompts = val_prompts, 
                    valid_df = val_df,
                    test_prompts = test_prompts, 
                    test_df = test_df, 
                    train_df = train_df,
                    logf=logf
                )
            try:
                pd.DataFrame(gptj_best_test_y).to_csv(csv_fpath)
            except:
                print("!!!! can't save gptj_best_test_y")
    
    
elif args.task == cfgs.IMBALANCE:
    for data_id in data_ids:
        for run_idx in range(start_idx,end_idx):
            csv_fpath = f"results/csv/run_gptj_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}_did_{args.data_id}_runidx_{args.run_idx}.csv"
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(int(data_id), run_idx,mixup=args.mixup)
            
            flip=False
            if len(y_train) + len(y_val) + len(y_test) > 2*(sum(y_train)+sum(y_val)+sum(y_test)):
                    print('Majority was zeros, but flipped')
                    flip = True
            
            train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
            train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
            jsonl_files = load_jsonl(data_id, run_idx, False)
            
            in_context_prefix = ''
            val_prompts = extract_prompts(jsonl_files['val'],in_context_prefix)
            test_prompts = extract_prompts(jsonl_files['test'],in_context_prefix)
            print("Training and Validation jsonl:", jsonl_files)
            print("===============config==============")
            print(CONFIG)
            gptj_fine_tuner = GPTJFineTuner(config=CONFIG,train_jsonl=jsonl_files['train'],valid_jsonl=jsonl_files['val'],cuda_idx=CUDA_ID)
            gptj_fine_tuner.fine_tune()
            log(logf, f"==== DID {data_id} {run_idx}_th run===")
            gptj_best_test_y, val_valid_y_num_list,val_acc_list, test_acc_list = gptj_fine_tuner.eval(
                valid_prompts = val_prompts, 
                valid_df = val_df,
                test_prompts = test_prompts, 
                test_df = test_df, 
                train_df = train_df,
                logf=logf,
                imbalance=True,
                flip = flip
            )
            try:
                pd.DataFrame(gptj_best_test_y).to_csv(csv_fpath)
            except:
                print("!!!! can't save gptj_best_test_y")

elif args.task == cfgs.LABEL_CORRUPTION:
    random_label_noises = [0.05, 0.1, 0.2]
    for data_id in data_ids:
        for random_label_noise in random_label_noises:
            for run_idx in range(start_idx,end_idx):

                csv_fpath = f"results/csv/run_gptj_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}_did_{args.data_id}_runidx_{args.run_idx}.csv"
                print('For random label noise', random_label_noise)
                print('Log path', log_fpath)
                print('Results path', csv_fpath)

                X_train, y_train, X_val, y_val, X_test, y_test = load_data(int(data_id), run_idx,mixup=args.mixup)
                train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
                train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
                jsonl_files = load_jsonl(data_id, run_idx, False)

                # Corrupt labels
                if args.label_corruption != 'none':
                    jsonl_files['train'] = load_corrupted_data(int(data_id), run_idx, LABEL_NOISES,args.label_corruption)


                in_context_prefix = ''
                val_prompts = extract_prompts(jsonl_files['val'],in_context_prefix)
                test_prompts = extract_prompts(jsonl_files['test'],in_context_prefix)

                print("Training and Validation jsonl:", jsonl_files)
                # CUDA_ID = int(not CUDA_ID)
                print("===============config==============")
                print(CONFIG)
                gptj_fine_tuner = GPTJFineTuner(config=CONFIG,train_jsonl=jsonl_files['train'],valid_jsonl=jsonl_files['val'],cuda_idx=CUDA_ID)
                gptj_fine_tuner.fine_tune()
                log(logf, f"==== DID {data_id} noise {random_label_noise} {run_idx}_th run ====")
                gptj_best_test_y, val_valid_y_num_list,val_acc_list, test_acc_list = gptj_fine_tuner.eval(
                    valid_prompts = val_prompts, 
                    valid_df = val_df,
                    test_prompts = test_prompts, 
                    test_df = test_df, 
                    train_df = train_df,
                    logf=logf,
                    imbalance=False
                )
                try:
                    pd.DataFrame(gptj_best_test_y).to_csv(csv_fpath)
                except:
                    print("!!!! can't save gptj_best_test_y")

elif args.task == cfgs.TEACH:
    test_values = 200
    for data_id in data_ids:

        print('Log path', log_fpath)
        print('Results path', csv_fpath)

        accuracies = []
        log(logf, "did epoch lrm val_acc test_acc model_id")
        for run_idx in range(1):
            csv_fpath = f"results/csv/run_gptj_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}_did_{args.data_id}_runidx_{args.run_idx}.csv"
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(int(data_id), run_idx,mixup=args.mixup)
            train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
            train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
            jsonl_files = load_jsonl(data_id, run_idx, False)

            in_context_prefix = ''
            val_prompts = extract_prompts(jsonl_files['val'],in_context_prefix)
            test_prompts = extract_prompts(jsonl_files['test'],in_context_prefix)

            jsonl_train_sampled = 'data/teacher_datasets/dataset_{}_teachermodel_train.jsonl'.format(data_id)

            # Fine-tune model
            jsonl_files["train"] = jsonl_train_sampled
            print("Training and Validation jsonl:", jsonl_files)
            # gptj_fine_tuner = GPTJFineTuner(config=CONFIG,train_jsonl=jsonl_files['train'],valid_jsonl=jsonl_files['val'],cuda_idx=args.gpu_id)
            # CUDA_ID = int(not CUDA_ID)
            print("===============config==============")
            print(CONFIG)
            gptj_fine_tuner = GPTJFineTuner(config=CONFIG,train_jsonl=jsonl_files['train'],valid_jsonl=jsonl_files['val'],cuda_idx=CUDA_ID)
            gptj_fine_tuner.fine_tune()
            log(logf, f"==== DID {data_id} {run_idx}_th run===")
            gptj_best_test_y, val_valid_y_num_list,val_acc_list, test_acc_list = gptj_fine_tuner.eval(
                valid_prompts = val_prompts, 
                valid_df = val_df,
                test_prompts = test_prompts, 
                test_df = test_df, 
                train_df = train_df,
                logf=logf
            )

            # Get predicted y for randomly sampled rest
            filename_rest = 'data/teacher_datasets/dataset_{}_teachermodel_rest.csv'.format(data_id)
            rest_df = pd.read_csv(filename_rest, sep='|', header=None)
            use_model = gptj_fine_tuner.ft_model
            # Parse string array into actual array
            prompts = ([[float(x) for x in arr[1:-1].split(',')] for arr in rest_df.iloc[:test_values, 0]])
            # Convert each prompt into a sentence for GPT
            prompts = array2prompts(prompts)
            # Feed prompts to GPT
            y_pred_teach = gptj_fine_tuner.query(use_model, prompts, bs = 15)

            # Test on all teacher datasets
            log(logf, "data_id model_name clf_name acc_teach")
            for model_name in clf_model_teachers:
                clf_dict = clf_model_teachers[model_name]
                for clf_name,clf in clf_dict.items():
                    try:
                        filename='data/teacher_datasets/dataset_{}_teachermodel_{}_{}.csv'.format(data_id,model_name, clf_name)
                        teach_df = pd.read_csv(filename, sep='|', header=None)
                        acc_teach = get_accuracy(y_pred_teach, teach_df.iloc[:test_values, 1])
                        log(logf, f"{data_id} {model_name} {clf_name} {acc_teach}")
                        accuracies.append([model_name, clf_name, acc_teach])
                    except Exception as e: 
                        print(e)
                        from IPython import embed; embed()
      
        pd.DataFrame(accuracies).to_csv(csv_fpath, header=False, index=False)

elif args.task == cfgs.SAMPLING:
    for data_id in data_ids:
        for run_idx in range(start_idx,end_idx):
            csv_fpath = f"results/csv/run_gptj_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}_did_{args.data_id}_runidx_{args.run_idx}.csv"
            # load dataset
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(int(data_id), run_idx,mixup=args.mixup)
            for s in [0.05, 0.1, 0.2, 0.5]:
                while True:
                    n = X_train.shape[0]
                    m = max(10, int(n* s))
                    print("Number of Training Samples:",m)
                    X_train_subset = X_train[:m]
                    y_train_subset = y_train[:m]
                    if len(set(y_train_subset)) > 1:
                        break
                
                # convert 
                train_df, val_df, test_df = pd.DataFrame(X_train_subset), pd.DataFrame(X_val), pd.DataFrame(X_test)
                train_df['y'], val_df['y'], test_df['y'] = y_train_subset, y_val, y_test
                jsonl_files = load_jsonl(data_id, run_idx, False) # load w/ or w/o feature names
                jsonl_files['train'] = df2jsonl(train_df, f"openml/{data_id}_{s}_train.jsonl")

                in_context_prefix = ''
                val_prompts = extract_prompts(jsonl_files['val'],in_context_prefix)
                test_prompts = extract_prompts(jsonl_files['test'],in_context_prefix)
                

                print("Training and Validation jsonl:", jsonl_files)
                print("===============config==============")
                print(CONFIG)
                gptj_fine_tuner = GPTJFineTuner(config=CONFIG,train_jsonl=jsonl_files['train'],valid_jsonl=jsonl_files['val'],cuda_idx=CUDA_ID)
                gptj_fine_tuner.fine_tune()
                log(logf, f"==== DID {data_id} Sample Fraction {s} {run_idx}_th run===")            
                gptj_best_test_y, val_valid_y_num_list,val_acc_list, test_acc_list = gptj_fine_tuner.eval(
                        valid_prompts = val_prompts, 
                        valid_df = val_df,
                        test_prompts = test_prompts, 
                        test_df = test_df, 
                        train_df = train_df,
                        logf=logf
                    )
                try:
                    pd.DataFrame(gptj_best_test_y).to_csv(csv_fpath)
                except:
                    print("!!!! can't save gptj_best_test_y")

elif args.task == cfgs.NNET:
    data_gen = dataGenerate()
    epochs = [10, 80, 490]
    accuracies = []
    for epoch in epochs:
        model = DataNN([2, 6, 6, 2], 'tanh')
        model_path = f"data/nnet/model_at_ep_{epoch}.pth"
        checkpoint = torch.load(model_path)['model_state_dict']
        model.load_state_dict(checkpoint)

        def nnet2(X):
            out = model(torch.from_numpy(X).float())
            y = torch.argmax(out, dim=1)
            return y.numpy()

        train_df, valid_df, test_df, test_prompts, grid_df, grid_prompts = data_gen.neural_net(nnet2, n=2000, name=str(epoch), ranges=(-6, 6), noise=0, resolution=300, corrupted=args.corrupted)


        train_data = train_df.values
        test_data = test_df.values
        x_train = train_data[:, :-1]
        y_train = train_data[:, -1]
        x_test = test_data[:, :-1]
        y_test = test_data[:, -1]
        x_grid = grid_df.values

        t_prompts = test_prompts + grid_prompts
        t_prompts = add_ending_symbol(t_prompts) 
        config = {'learning_rate': 1e-4, 'batch_size': args.batch_size, 'epochs':10, 'weight_decay': 0.01, 'warmup_steps': 6}


        log(logf, f"==== NN Generator at epoch {epochs} ===")            
        ft_model = lora_gptj.LoRaQGPTJ(adapter=True, device=torch.device('cuda:%d' % args.gpu_id) if torch.cuda.is_available() else 'cpu')
        ft_model.finetune(data_gen.train_jsonl, data_gen.val_jsonl,config,saving_checkpoint = False)
        y_preds = query(ft_model, t_prompts, bs = 400)
        n_test = len(test_prompts)
        y_pred, y_grid = y_preds[:n_test], y_preds[n_test:]
        acc = (np.array(y_pred) == np.array(y_test)).mean()
        print('Accuracy of gpt', acc)

        df = pd.DataFrame(data={0: x_grid[:, 0], 1: x_grid[:, 1], 'y': np.asarray(y_grid)})
        df.to_csv(f'gptj_grid_ep_{epoch}_corrupted_{args.corrupted}.csv')

        plot_path = 'results/plots/'
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)
        plot_decision_boundry_custom_color('gptj' + f'_ep{epoch}_corrupted_{args.corrupted}', df, resolution=300,
                                color_1='#a39faa',color_2='#fbece1',bd_color='#a39faa',
                                save_dir=plot_path)
