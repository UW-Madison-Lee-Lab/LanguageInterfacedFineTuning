import numpy as np

import argparse
from utils.corrupt_labels import corrupt_labels
from utils.helper import log
from utils.classification_data_generator import DataGenerator, load_openml
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
# from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
# from sklearn.model_selection import HalvingRandomSearchCV
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import pandas as pd

from utils.prepare_data import prepare_data
from models.baselines import clf_model, param_grids
import utils.configs as cfgs
from utils.classificationDataGen import dataGenerate
from utils.train_nn_generator import DataNN
from utils.helper import add_ending_symbol
from plotters.plotter import plot_decision_boundry_custom_color

parser = argparse.ArgumentParser(description='Baselines')
parser.add_argument("-d", "--data", default=-1, type=int)
parser.add_argument("-c", "--corrupted", default=0, type=float)
parser.add_argument("-t", "--task", default='accuracy', type=str)
parser.add_argument("-m", "--mixup", default=0, type=int)


######  Load configurations ##############
args = parser.parse_args()


model_names = ['majorguess', 'logreg', 'knn', 'tree', 'nn', 'svm', 'rf', 'xgboost']
if args.mixup:
    raise NotImplementedError
    model_names = ['nnmixup']#['random', 'logreg', 'knn', 'tree', 'nn', 'svm', 'rf', 'xgboost', 'nn_mixup']
# model_names = ['nn']
# eval_prefix = 'results/evals/rerun_all_acc_clf_baselines'
# csv_prefix = 'results/csv/rerun_all_acc_clf_baselines'
eval_prefix = 'results/evals/rnd_rerun_all_acc_clf_baselines'
csv_prefix = 'results/csv/rnd_rerun_all_acc_clf_baselines'




done_keys = []
sorted_done_key = 0
######## Data
# data_lst = cfgs.synthetic_data_ids.values()
data_lst =  cfgs.openml_data_ids.keys()
data_ids = [args.data] if args.data > -1 else data_lst
lst_ids = []
for k in data_ids:
    if k not in done_keys and k > sorted_done_key:
        lst_ids.append(k)
data_ids = np.sort(lst_ids)

def load_baseline_data(did, split):
    # fname = f'data/{did}_split{split}.npy'
    fname = f'data/{did}_dev_test_split.npy'
    if not os.path.isfile(fname):
        print('prepare data', did)
        prepare_data(did, context=False)
    ith_fname = f'data/{did}_train_val_test_split{split}.npy'
    data = np.load(ith_fname, allow_pickle=True)
    data = data.item()
    X_train, X_val, X_test = data['X_norm_train'], data['X_norm_val'], data['X_norm_test']
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_acc(clf, X_train, y_train, X_val, y_val, X_test, y_test):
    try:
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        y_pred_val = clf.predict(X_val)
        val_acc = round((y_pred_val == y_val).mean() * 100, 2)
        test_acc = round((y_pred_test == y_test).mean() * 100, 2)
    except:
        val_acc = -1
        test_acc = -1
    return val_acc, test_acc


############## Running ####################
if args.task == cfgs.ACCURACY:
    log_fpath = f'{eval_prefix}_{args.task}.txt'
    csv_fpath = f'{csv_prefix}_{args.task}.csv'
    logf = open(log_fpath, 'a+')

    results = []
    for data_id in data_ids:
        try:
            X_train, y_train, X_val, y_val, X_test, y_test = load_baseline_data(int(data_id), 0)
            X = np.concatenate([X_train, X_val], axis=0)
            y = np.concatenate([y_train, y_val], axis=0)
            # sh = HalvingGridSearchCV(base_estimator, param_grid, cv=5, factor=2, min_resources=20).fit(X, y)
            data_accuracies = []
            for clf_name in model_names:
                if clf_name in ['random', 'majorguess']:
                    best_clf = clf_model[clf_name]
                else:
                    base_estimator = clf_model[clf_name]
                    print("We are here...")
                    search = HalvingGridSearchCV(base_estimator, param_grids[clf_name], random_state=0).fit(X, y)
                    best_clf = search.best_estimator_
                best_val = -1
                best_test = -1
                for split in range(3):
                    X_train, y_train, X_val, y_val, X_test, y_test = load_baseline_data(int(data_id), split)
                    val_acc, test_acc = get_acc(best_clf, X_train, y_train, X_val, y_val, X_test, y_test)
                    if val_acc > best_val:
                        best_val = val_acc
                        best_test = test_acc
                    log(logf, f"{data_id} {clf_name} {split} {val_acc} {test_acc}")
                acc = best_test
                message = f"{data_id} {clf_name} {acc}"
                # print(message)
                data_accuracies.append(acc)
            results.append(data_accuracies)
        except Exception as e:
            print('Errror ', data_id, e)
    np.savetxt(csv_fpath, np.asarray(results))

elif args.task == cfgs.IMBALANCE:
    precision = []
    recall = []
    f1 = []
    for data_id in data_ids:
        log_fpath = f'{eval_prefix}_{args.task}_{data_id}.txt'
        csv_fpath = f'{csv_prefix}_{args.task}_{data_id}.csv'
        logf = open(log_fpath, 'a+')
        X_train, y_train, X_val, y_val, X_test, y_test = load_baseline_data(int(data_id), 0)
        X = np.concatenate([X_train, X_val], axis=0)
        y = np.concatenate([y_train, y_val], axis=0)
        for name in model_names:
            if name == 'random' or 'majorguess':
                best_clf = clf_model[name]
            else:
                base_estimator = clf_model[name]
                search = HalvingGridSearchCV(base_estimator, param_grids[name], random_state=0).fit(X, y)
                best_clf = search.best_estimator_
            accuracies = []
            for idx in range(3):
                X_train, y_train, X_val, y_val, X_test, y_test = load_baseline_data(int(data_id), idx)
                try:
                    best_clf.fit(X_train, y_train)
                    # validation
                    y_pred_val = best_clf.predict(X_val)
                    precision_val = round(precision_score(y_val, y_pred_val) * 100, 2)
                    recall_val = round(recall_score(y_val, y_pred_val) * 100, 2)
                    acc_val = round(accuracy_score(y_val, y_pred_val) * 100, 2)
                    f1_val = round(f1_score(y_val, y_pred_val) * 100, 2)
                    #test
                    y_pred = best_clf.predict(X_test)
                    precision = round(precision_score(y_test, y_pred) * 100, 2)
                    recall = round(recall_score(y_test, y_pred) * 100, 2)
                    acc = round(accuracy_score(y_test, y_pred) * 100, 2)
                    f1 = round(f1_score(y_test, y_pred) * 100, 2)
                except:
                    acc_val, precision_val, recall_val, f1_val = -1, -1, -1, -1
                    acc, precision, recall, f1 = -1, -1, -1, -1
                accs = [acc_val, f1_val, precision_val, recall_val,acc, f1, precision, recall]
                # print(name, accs)
                message = f"{data_id} {idx} " + " ".join(["%.2f" % x for x in accs])
                log(logf, message)
                # accuracies.append(accs)
        # save the csv
        # np.savetxt(csv_fpath, np.asarray(accuracies))

elif args.task == cfgs.SAMPLING:
    sample_values = [0.05, 0.1, 0.2, 0.5]
    # sample_values = [500]
    for data_id in data_ids:
        log_fpath = f'{eval_prefix}_{args.task}_{data_id}.txt'
        csv_fpath = f'{csv_prefix}_{args.task}_{data_id}.csv'
        logf = open(log_fpath, 'a+')
        accuracies = []
        # data
        X_train, y_train, X_val, y_val, X_test, y_test= load_baseline_data(data_id)
        # sample
        count = 0
        n = X_train.shape[0]
        slice = np.arange(n)

        for s in sample_values:
            # stratified separation
            while True:
                np.random.shuffle(slice)
                # slice = np.arange(n)
                if s < 1:
                    m = max(10, int(n* s))
                else:
                    m = s
                X_train_subset = X_train[slice[:m]]
                y_train_subset = y_train[slice[:m]]
                if len(set(y_train_subset)) > 1:
                    break
            accs = []
            for name in model_names:
                acc = get_acc(name, X_train_subset, y_train_subset, X_test, y_test)
                accs.append(acc)
                print(data_id, s, name, "{:.2f}".format(acc))
            accuracies.append(accs)
            message = f"{s} " + " ".join(["%.2f" % x for x in accs])
            log(logf, message)
        # save the csv
        np.savetxt(csv_fpath, np.asarray(accuracies))

# Michael's changes
elif args.task == cfgs.LABEL_CORRUPTION:
    random_label_noises = [0, 0.05, 0.1, 0.2]
    for data_id in data_ids:
        for random_label_noise in random_label_noises:

            log_fpath = f'{eval_prefix}_{args.task}_{data_id}_{random_label_noise}.txt'
            csv_fpath = f'{csv_prefix}_{args.task}_{data_id}_{random_label_noise}.csv'
            logf = open(log_fpath, 'a+')

            print('For random label noise', random_label_noise)
            print('Log path', log_fpath)
            print('Results path', csv_fpath)

            results = []
            for data_id in data_ids:
                try:
                    X_train, y_train, X_val, y_val, X_test, y_test = load_baseline_data(int(data_id), 0)

                    # Corrupt labels
                    y_train = corrupt_labels(y_train, random_label_noise)
                    y_val = corrupt_labels(y_val, random_label_noise)

                    X = np.concatenate([X_train, X_val], axis=0)
                    y = np.concatenate([y_train, y_val], axis=0)
                    data_accuracies = []
                    for clf_name in model_names:
                        if clf_name == 'random':
                            best_clf = clf_model[clf_name]
                        else:
                            base_estimator = clf_model[clf_name]
                            search = HalvingGridSearchCV(base_estimator, param_grids[clf_name], random_state=0).fit(X, y)
                            best_clf = search.best_estimator_
                        best_val = -1
                        best_test = -1
                        for split in range(3):
                            X_train, y_train, X_val, y_val, X_test, y_test = load_baseline_data(int(data_id), split)

                            # Corrupt labels
                            y_train = corrupt_labels(y_train, random_label_noise)

                            val_acc, test_acc = get_acc(best_clf, X_train, y_train, X_val, y_val, X_test, y_test)
                            if val_acc > best_val:
                                best_val = val_acc
                                best_test = test_acc
                        acc = best_test
                        message = f"{data_id} {clf_name} {acc}"
                        print(message)
                        data_accuracies.append(acc)
                    log(logf, f"{data_id} " + " ".join(["%.2f" % x for x in data_accuracies]))
                    results.append(data_accuracies)
                except Exception as e:
                    print('Errror ', data_id, e)
            np.savetxt(csv_fpath, np.asarray(results))


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


        y_grid = {}
        accs = []
        for name in model_names:
            clf = clf_model[name]
            clf.fit(x_train, y_train)
            y_pred = clf.predict(x_test)
            y_grid[name] = clf.predict(x_grid)
            acc = (y_pred == y_test).mean()
            accs.append(acc)
            print(name, acc)
        accuracies.append(accs)
            

        true_df = pd.DataFrame(data={0: x_grid[:, 0], 1: x_grid[:, 1], 'y': nnet2(x_grid)})

        plot_path = 'results/plots/'
        if not os.path.isdir(plot_path):
            os.mkdir(plot_path)
        plot_decision_boundry_custom_color(f'function_ep{epoch}_corrupted_{args.corrupted}', true_df, resolution=300,
                    color_1='#a39faa',color_2='#fbece1',bd_color='#a39faa',
                    save_dir=plot_path)
        for name in y_grid.keys():
            df = pd.DataFrame(data={0: x_grid[:, 0], 1: x_grid[:, 1], 'y': y_grid[name]})
            plot_decision_boundry_custom_color(name + f'_ep{epoch}_corrupted_{args.corrupted}', df, resolution=300,
                                   color_1='#a39faa',color_2='#fbece1',bd_color='#a39faa',
                                    save_dir=plot_path)
        
    np.savetxt(f'results/acc_baselines_corrupted_{args.corrupted}_xg.csv', np.asarray(accuracies))