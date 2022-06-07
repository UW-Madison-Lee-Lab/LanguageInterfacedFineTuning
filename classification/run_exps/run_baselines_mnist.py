import sys
sys.path.append('./')
sys.path.append('./../')
import numpy as np
from models.baselines import clf_model
import utils.configs as cfgs
from utils.helper import log
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='GPT')
parser.add_argument("-d", "--data_name", default='mnist', type=str,choices=['mnist','fmnist'])
parser.add_argument("-c", "--corrupted", default=0, type=float)
parser.add_argument("-p", "--is_permuted", action="store_true")
parser.add_argument("-t", "--task", default='accuracy', type=str)


######  Load configurations ##############
args = parser.parse_args()

model_names = ['majorguess', 'logreg', 'knn', 'tree', 'nn', 'svm', 'rf', 'xgboost']



# data_name = 'fmnist'
data_name = args.data_name
is_permuted = args.is_permuted
permuted = 'permuted_' if is_permuted else ''

data_id = f'{permuted}{data_name}'
data = np.load(f'data/{data_id}.npy', allow_pickle=True)
data = data.item()
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

X_train = X_train/255
X_test = X_test/255


accuracies = []
for name in model_names:
    try:
        clf = clf_model[name]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = (y_pred == y_test).mean()
        acc = round(acc * 100, 2)
    except:
        acc = -1
    print(name, "{:.2f}".format(acc))
    accuracies.append(acc)

############## Running ####################
log_fpath = f'results/evals/clf_baselines_{data_id}.txt'
csv_fpath = f'results/csv/clf_baselines_{data_id}.csv'
logf = open(log_fpath, 'a+')
message = f"{data_id} " + " ".join(["%.2f" % x for x in accuracies])
log(logf, message)
np.savetxt(csv_fpath, np.asarray(accuracies))

