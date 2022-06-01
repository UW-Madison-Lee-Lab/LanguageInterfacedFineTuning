from re import A
import pandas as pd
import numpy as np
import openai
import argparse
import time, json, os
import torch

# from main_baselines import load_baseline_data
from sklearn.metrics import precision_score,recall_score, accuracy_score, f1_score

import utils.configs as cfgs
from utils.classificationDataGen import dataGenerate
from utils.classification_data_generator import df2jsonl
from utils.helper import log
from run_exps_helper import *
from models.baselines import clf_model_teachers
from utils.train_nn_generator import DataNN
from utils.helper import add_ending_symbol
from utils.plotter import plot_decision_boundry_custom_color


######  MAIN ##############
parser = argparse.ArgumentParser(description='GPT3')
parser.add_argument("-b","--batch_size", default=4, type=int)
parser.add_argument("-d", "--data_id", default=-1, type=int)
parser.add_argument("-e","--subset", type=str,default='none',choices=['in_context','fraction','none'])
parser.add_argument("-f","--use_feature_name", action='store_true',help='replace feature indices with feature names')
parser.add_argument("-g", "--gpu_id", default=1, type=int)
parser.add_argument("-i", "--run_idx", default=0, type=int)
parser.add_argument("-k","--label_corruption", type=str,default='none',choices=['random','system','none'])
parser.add_argument("-l", "--lrm", default=0.1, type=float)
parser.add_argument('-n',"--in_context", action='store_true')
parser.add_argument("-m", "--mixup", action='store_true')
parser.add_argument("-o", "--openai_key", default='', type=str)
parser.add_argument("-p", "--epochs", default=15, type=int)
parser.add_argument("-r","--subset_fraction", default=1, type=float)
parser.add_argument("-t", "--task", default='accuracy', type=str)
parser.add_argument("-y", "--model_type", default='ada', type=str,choices=['ada','babbage', 'curie', 'davinci'])


args = parser.parse_args()

# openai key
openai.api_key = args.key
print("openai.api_key",openai.api_key)
openai_config = {'model_type':args.model_type, "num_epochs":args.epochs, "batch_size":8, 'learning_rate_multiplier': args.lrm}


CUDA_ID = args.gpu_id
LABEL_NOISES = [0.05, 0.1, 0.2]
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
log_fpath = f"results/evals/run_gpt3_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}.txt"
logf = open(log_fpath, 'a+')


if args.run_index == -1:
    start_idx = 0
    end_idx = 3
else:
    start_idx = args.run_index
    end_idx = args.run_index + 1


if args.task == cfgs.ACCURACY:
    log(logf, "data_id run_idx epochs lrm acc_val acc") 
    for data_id in data_ids:
        for run_idx in range(start_idx,end_idx):
            csv_fpath = f"results/csv/run_gpt3_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}_did_{args.data_id}_runidx_{args.run_index}.csv"
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

            y_pred_val, y_pred, gpt3_fine_tuner = run_gpt3(data_id, jsonl_files['train'], jsonl_files['val'], val_prompts,test_prompts, args.in_context,openai_config, positive_class=int(y_test[0]))
            acc_val = get_accuracy(y_pred_val, y_val)
            acc = get_accuracy(y_pred, y_test)
            log(logf, f"{data_id} {run_idx} {args.epochs} {args.lrm} {acc_val} {acc}") 

            try:
                pd.DataFrame(y_pred).to_csv(csv_fpath)
            except:
                print("!!!! can't save y_pred")

elif args.task == cfgs.IMBALANCE:
    log(logf, "data_id run_idx epochs lrm acc_val f1_val precision_val recall_val acc f1 precision recall") 
    for data_id in data_ids:
        for run_idx in range(start_idx,end_idx):
            csv_fpath = f"results/csv/run_gpt3_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}_did_{args.data_id}_runidx_{args.run_index}.csv"
            # load dataset
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(int(data_id), run_idx,mixup=args.mixup)
            train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
            train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
            jsonl_files = load_jsonl(data_id, run_idx, False) # load w/ or w/o feature names

            in_context_prefix = ''
            val_prompts = extract_prompts(jsonl_files['val'],in_context_prefix)
            test_prompts = extract_prompts(jsonl_files['test'],in_context_prefix)
            

            print("Training and Validation jsonl:", jsonl_files)
            print("===============config==============")
            print(CONFIG)

            y_pred_val, y_pred, gpt3_fine_tuner = run_gpt3(data_id, jsonl_files['train'], jsonl_files['val'], val_prompts,test_prompts, args.in_context,openai_config, positive_class=int(y_test[0]))

            acc_val = round(accuracy_score(y_val, y_pred_val) * 100, 2)
            f1_val = round(f1_score(y_val, y_pred_val) * 100, 2)
            precision_val = round(precision_score(y_val, y_pred_val) * 100, 2)
            recall_val = round(recall_score(y_val, y_pred_val) * 100, 2)
            results_val = " ".join(["%.2f" % x for x in [acc_val, f1_val, precision_val, recall_val]])

            acc = round(accuracy_score(y_test, y_pred) * 100, 2)
            f1 = round(f1_score(y_test, y_pred) * 100, 2)
            precision = round(precision_score(y_test, y_pred) * 100, 2)
            recall = round(recall_score(y_test, y_pred) * 100, 2)
            results = " ".join(["%.2f" % x for x in [acc, f1, precision, recall]])
            log(logf, f"{data_id} {run_idx} {args.epochs} {args.lrm} {results_val} {results}")

            try:
                pd.DataFrame(y_pred).to_csv(csv_fpath)
            except:
                print("!!!! can't save y_pred")
                
elif args.task == cfgs.LABEL_CORRUPTION:
    random_label_noises = [0.05, 0.1, 0.2]
    for data_id in data_ids:
        log(logf, "did noise run_idx epoch lrm val_acc test_acc")
        for random_label_noise in random_label_noises:
            for run_idx in range(start_idx,end_idx):

                csv_fpath = f"results/csv/run_gpt3_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}_did_{args.data_id}_runidx_{args.run_index}.csv"
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
                print("===============config==============")
                print(CONFIG)
                y_pred_val, y_pred, gpt3_fine_tuner = run_gpt3(data_id, jsonl_files['train'], jsonl_files['val'], val_prompts,test_prompts, args.in_context,openai_config, positive_class=int(y_test[0]))
                acc_val = get_accuracy(y_pred_val, y_val)
                acc = get_accuracy(y_pred, y_test)
                log(logf, f"{data_id} {random_label_noise} {run_idx} {args.epochs} {args.lrm} {acc_val} {acc}") 

                try:
                    pd.DataFrame(y_pred).to_csv(csv_fpath)
                except:
                    print("!!!! can't save y_pred")

elif args.task == cfgs.TEACH:
    test_values = 200
    for data_id in data_ids:
        log(logf, "did run_idx epoch lrm val_acc test_acc")
        for run_idx in range(1):
            csv_fpath = f"results/csv/run_gpt3_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}_did_{args.data_id}_runidx_{args.run_index}.csv"
            X_train, y_train, X_val, y_val, X_test, y_test = load_data(int(data_id), run_idx,mixup=args.mixup)
            train_df, val_df, test_df = pd.DataFrame(X_train), pd.DataFrame(X_val), pd.DataFrame(X_test)
            train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
            jsonl_files = load_jsonl(data_id, i, False)

            in_context_prefix = ''
            val_prompts = extract_prompts(jsonl_files['val'],in_context_prefix)
            test_prompts = extract_prompts(jsonl_files['test'],in_context_prefix)

            jsonl_train_sampled = 'data/teacher_datasets/dataset_{}_teachermodel_train.jsonl'.format(data_id)

            # Fine-tune model
            jsonl_files["train"] = jsonl_train_sampled
            print("Training and Validation jsonl:", jsonl_files)
            print("===============config==============")
            y_pred_val, y_pred, gpt3_fine_tuner = run_gpt3(data_id, jsonl_files['train'], jsonl_files['val'], val_prompts,test_prompts, args.in_context,openai_config, positive_class=int(y_test[0]))
            acc_val = get_accuracy(y_pred_val, y_val)
            acc = get_accuracy(y_pred, y_test)
            log(logf, f"{data_id} {run_idx} {args.epochs} {args.lrm} {acc_val} {acc}") 

            # Get predicted y for randomly sampled rest
            filename_rest = 'data/teacher_datasets/dataset_{}_teachermodel_rest.csv'.format(data_id)
            rest_df = pd.read_csv(filename_rest, sep='|', header=None)
            use_model = gpt3_fine_tuner.ft_model
            prompts = ([[float(x) for x in arr[1:-1].split(',')] for arr in rest_df.iloc[:test_values, 0]]) # Parse string array into actual array
            prompts = array2prompts(prompts) # Convert each prompt into a sentence for GPT
            y_pred_teach = generate_output_in_context(prompts, use_model) # Feed prompts to GPT

            # Test on all teacher datasets
            log(logf, "data_id model_name clf_name acc_teach")
            for model_name in clf_model_teachers:
                clf_dict = clf_model_teachers[model_name]
                for clf_name,clf in clf_dict.items():
                    filename='data/teacher_datasets/dataset_{}_teachermodel_{}_{}.csv'.format(data_id,model_name, clf_name)
                    teach_df = pd.read_csv(filename, sep='|', header=None)
                    acc_teach = get_accuracy(y_pred_teach, teach_df.iloc[:test_values, 1])
                    log(logf, f"{data_id} {model_name} {clf_name} {acc_teach}")


            
elif args.task == cfgs.SAMPLING:
    for data_id in data_ids:
        log(logf, "data_id sample_fraction run_idx epochs lrm acc_val acc") 
        for run_idx in range(start_idx,end_idx):
            csv_fpath = f"results/csv/run_gpt3_{args.task}_featname_{args.use_feature_name}_incon_{args.in_context}_subset_{args.subset}_fraction_{args.subset_fraction}_did_{args.data_id}_runidx_{args.run_index}.csv"
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
                train_df['y'], val_df['y'], test_df['y'] = y_train, y_val, y_test
                jsonl_files = load_jsonl(data_id, run_idx, False) # load w/ or w/o feature names
                jsonl_files['train'] = df2jsonl(train_df, f"openml/{data_id}_{s}_train.jsonl")

                in_context_prefix = ''
                val_prompts = extract_prompts(jsonl_files['val'],in_context_prefix)
                test_prompts = extract_prompts(jsonl_files['test'],in_context_prefix)
                

                print("Training and Validation jsonl:", jsonl_files)
                print("===============config==============")
                print(CONFIG)

                y_pred_val, y_pred, gpt3_fine_tuner = run_gpt3(data_id, jsonl_files['train'], jsonl_files['val'], val_prompts,test_prompts, args.in_context,openai_config, positive_class=int(y_test[0]))

                acc_val = round(accuracy_score(y_val, y_pred_val) * 100, 2)
                acc = get_accuracy(y_pred, y_test)
                log(logf, f"{data_id} {s} {run_idx} {args.epochs} {args.lrm} {acc_val} {acc}") 

                try:
                    pd.DataFrame(y_pred).to_csv(csv_fpath)
                except:
                    print("!!!! can't save y_pred")


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
        config = {'model_type':'ada',"num_epochs":10,"batch_size":50}
        log(logf, f"==== NN Generator at epoch {epochs} ===")  

        gpt3_fine_tuner = GPT3FineTuner(config,data_gen.train_jsonl, data_gen.val_jsonl)
        gpt3_fine_tuner.fine_tune(clf_cfgs={'n_classes': 0})

        ans = []
        bs = 20
        count = 0
        while count < len(t_prompts):
            start = count 
            end = min(count + bs, len(t_prompts))
            batch = t_prompts[start:end]
            out = gpt3_fine_tuner.query(batch)
            ans += out
            count = end

        y_preds = [prompt2value(x) for x in ans]
        n_test = len(test_prompts)
        y_pred, y_grid = y_preds[:n_test], y_preds[n_test:]
        acc = (y_pred == y_test).mean()
        print('Accuracy of gpt', acc)

        true_y_grid = nnet2(x_grid)
        df = pd.DataFrame(data={0: x_grid[:, 0], 1: x_grid[:, 1], 'y': np.asarray(y_grid)})
        plot_path = 'results/plots/'
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)
        df.to_csv(f'gpt3_grid_ep_{epoch}_corrupted_{args.corrupted}.csv')
        plot_decision_boundry_custom_color('gpt3' + f'_ep{epoch}_corrupted_{args.corrupted}', df, resolution=300,
                                color_1='#a39faa',color_2='#fbece1',bd_color='#a39faa',
                                save_dir=plot_path)



