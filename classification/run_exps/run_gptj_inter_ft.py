
import json, os, torch
import numpy as np
import pandas as pd
import numpy as np
from functools import partial
import argparse
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from utils.helper import query
from models import lora_gptj
from utils.helper import log
from utils.classification_data_generator import DataGenerator
import utils.configs as cfgs

parser = argparse.ArgumentParser(description='GPT')
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("-d", "--did", default=1, type=int)
parser.add_argument("-m", "--method", default='ift', type=str)
parser.add_argument("-b", "--batch_size", default=4, type=int)
args = parser.parse_args()

def data2text(row, label = True, init = '', end = ''):
    prompt = init 
    for i in range(len(row)-label):
        v = row[i]
        prompt += "x%d=%.2f, " % (i+1, v)
    prompt += end

    if not label:
        final_prompt = f"{prompt}###"
    else:
        # v = round(float(row['y']), 3)
        completion = "%d" % int(row['y'])
        final_prompt = "{\"prompt\":\"%s###\", \"completion\":\"%s@@@\"}" % (prompt, completion)
    return final_prompt

def df2jsonl(df, filename, init = '', end = ''):
    jsonl = '\n'.join(df.apply(func = partial(data2text, init = init, end = end), axis = 1).tolist())
    fpath = os.path.join('data', filename)
    with open(fpath, 'w') as f:
        f.write(jsonl)
    return fpath

def df2propmts(df, data2text, init = '', end = '', context=False):
    jsonl = df.apply(func = partial(data2text, init = init, end = end), axis = 1).tolist()
    return jsonl

def write_jsonl(jsonl, filename):
    fpath = os.path.join('data', filename)
    with open(fpath, 'w') as f:
        f.write(jsonl)
    return fpath

def array2prompts(X, init = '', end = ''):
    return list(map(partial(data2text, 
                            label = False,
                            init = init, 
                            end = end
                           ), X))
def extract_prompts(jsonl_file):
    test_prompts = []
    with open(jsonl_file) as fp:
        for line in fp:
            json_obj = json.loads(line)
            test_prompts.append(json_obj['prompt'])
    return test_prompts

def extract_completion(jsonl_file):
    completions = []
    with open(jsonl_file) as fp:
        for line in fp:
            json_obj = json.loads(line)
            completions.append(json_obj['completion'])

def get_jsonl(data_name, X, y, data2text, init='Given that ', end = '', context = False):
    df = pd.DataFrame(X)
    df['y'] = y
    full_prompts = df2propmts(df, data2text, init, end, context)
    js = write_jsonl('\n'.join(full_prompts), f'{data_name}_{context}.jsonl')
    prompts = extract_prompts(js)
    return js, prompts, full_prompts

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def generate_data(W, N=2, num_samples=100, with_noise=False):
    W = np.asarray(W)
    X = np.random.rand(num_samples, N)
    if with_noise:
        e = np.random.randn(num_samples)
    else:
        e = 0
    y = X@W.T+e
    return X, y

def prompt2value(x):
    # print("Output:",x)
    c = x.strip().split('@@@')[0]
    if c == '':
        return None
    try:
        return int(c)
    except:
        return None

def get_accuracy(y_pred_val, y_val):
    acc_val = (y_pred_val == y_val).mean()
    acc_val = round(acc_val * 100, 2)
    return acc_val

# [[0.10506001495333138, 0.09804173432224533, 0.10199889848794548]]
# [[0.17704899031834903, 0.1925020922010446, 0.16748499387976368],
#  [0.2956480522595278, 0.3215939974325852, 0.2975796483670988],
#  [0.537864626471848, 0.5576800893904527, 0.6592448791478737],
#  [0.890254662473231, 0.8331797037031325, 0.9069316664463922]]
# Invalid
# [0.9072959768247009, 0.8388024941732575, 0.9108658302028555],
#  [1.5537964382081677, 1.4666238308163515, 1.4165365253942714],
#  [1.5932395016196454, None],
#  [1.943976178810879, 1.849568689496792, 2.12757065959928]

####
# 200 [0.11917805304045445, 0.12381201041483246, 0.12490990932462678]
# 100 [0.18289619945763452, 0.18546313184778968, 0.17242861283740668]
# 50 [0.35090368389858184, 0.3147572735318331, 0.34944138622833704]
# 20 [0.6235879232637113, 0.5947071691364909, 0.5888100497459052]
# 10 [0.8061939607944861, 0.8180120903250612, 0.7853001642709188]
# 5 [1.2836420309823393, 1.3065900592163369, 1.2852619706684878]
# # 2 [1.310923931023374, 1.370633451762684, 1.2702043584289908]


# [[0.13262273415951575, 0.12209908823847723, 0.11934051662960622],
#  [0.18053434426194864, 0.15535055310881665, 0.1772982799715835],
#  [0.3307396143772844, 0.3726730675583203, 0.5197854112355987],
#  [0.6200520792367131, 0.587004286746931, 0.5848703842288435],
#  [0.8025288380312707, 0.8697140471237773, None],
#  [None],
#  [1.6455406294244652, 1.5565452781870943, 1.507627269669047],
#  [None]]

# [[0.5456229663601488, 0.6257083478342996, 0.53490757145865],
#  [None],
#  [1.1383148747219105, 1.2004544146286682, 1.2379540277046557],
#  [1.3841056311094067, 1.3773663942765138, 1.3148823460097865],
# 1 [1.7866004385514351, 2.2819909831748206, 1.864006579076018]]

device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(args.gpu_id)

did = args.did
end = 'What is the cluster?'

# make classification data 
fname = 'data/test_clf_{did}_inter_tuning.npy'
if os.path.isfile(fname):
    data = np.load(fname, allow_pickle=True)
    data = data.item()
    X_train = data['X_train']
    y_train = data['y_train']
    y_test = data['y_test']
    train_js1 =  f'data/task1_clf_{did}_train_False.jsonl'
    train_js2 =  f'data/task2_clf_{did}_train_False.jsonl'
    val_js1 =  f'data/task1_clf_{did}_val_False.jsonl'
    val_js2 =  f'data/task2_clf_{did}_val_False.jsonl'
    val_js =  f'data/task3_clf_{did}_val_False.jsonl'
    test_js =  f'data/task3_clf_{did}_test_False.jsonl'
    test_prompts = extract_prompts(test_js)
else:
    ### target
    if did < 10:
        data_gen = DataGenerator(did)    
        y3, X3, _, att_names = data_gen.load_synthetic_datatsets(did)
        X_dev, X_test, y_dev, y_test = train_test_split(X3, y3, test_size=0.25, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=0.2, random_state=42)
        np.save('data/test_inter_tuning', {'X_train': X_train, 'X_val': X_val, 'X_test': X_test, 'y_val': y_val, 'y_test': y_test, 'y_train': y_train})
    else:
        ith_fname = f'data/{did}_train_val_test_split0.npy'
        data = np.load(ith_fname, allow_pickle=True)
        data = data.item()
        X_train, X_val, X_test = data['X_raw_train'], data['X_raw_val'], data['X_raw_test']
        y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    val_js, val_prompts, _ = get_jsonl('task3_clf_{did}_val', X_val, y_val, init=f'Given task 3, ', end=end, data2text=data2text)
    _, test_prompts, _ = get_jsonl('task3_clf_{did}_test', X_test, y_test, init=f'Given task 3, ', end=end, data2text=data2text)

    ### pretexts
    n_classes = cfgs.openml_data_ids[did]
    n_features = X_train.shape[1]

    X1, y1 = make_classification(n_samples=100, n_classes=n_classes, n_features=n_features, n_informative=n_features, n_redundant=0, n_clusters_per_class=1, flip_y=0, random_state=1)
    X2, y2 = make_classification(n_samples=100, n_classes=n_classes, n_features=n_features, n_informative=n_features, n_redundant=0, n_clusters_per_class=1, flip_y=0, random_state=2)

    X1_train, X1_val, y1_train, y1_val = train_test_split(X1, y1, test_size=0.2, random_state=42)
    X2_train, X2_val, y2_train, y2_val = train_test_split(X2, y2, test_size=0.2, random_state=42)

    train_js1, _, _ = get_jsonl('task1_clf_{did}_train', X1_train, y1_train, init=f'Given task 1, ', end=end, data2text=data2text)
    val_js1, _, _ = get_jsonl('task1_clf_{did}_val', X1_val, y1_val, init=f'Given task 1, ', end=end, data2text=data2text)
    train_js2, _, _ = get_jsonl('task2_clf_{did}_train', X2_train, y2_train, init=f'Given task 2, ', end=end, data2text=data2text)
    val_js2, _, _ = get_jsonl('task2_clf_{did}_val', X2_val, y2_val, init=f'Given task 2, ', end=end, data2text=data2text)
    

method = args.method
accs = []
log_fpath = f'results/inter_ft_clf_{did}_{method}.txt'
logf = open(log_fpath, 'a+')
if method == 'bl':
    n_trains = [0]
else:
    # n_trains = [10, 20, 50, 100, 200, 500, 1000]
    n_trains = [100, 200, 500, 1000]

if args.did < 10:
    args.batch_size = 16
elif args.did < 1000:
    args.batch_size = 4
else:
    args.batch_size = 2

for n_train in n_trains:
    # n_train = int(n_train * X_train.shape[0])
    if not(method == 'bl'):
        X3_train1, y3_train1 = X_train[:n_train], y_train[:n_train]
        train_js, _, _ = get_jsonl('task3_clf_{did}_train', X3_train1, y3_train1, init=f'Given task 3, ', end=end, data2text=data2text)

    gpt = lora_gptj.LoRaQGPTJ(adapter=True, device=device)
    if method in ['ift', 'bl']:
        train_configs={'learning_rate': 1e-4, 'batch_size': args.batch_size, 'epochs':3,  'weight_decay': 0.01, 'warmup_steps': 6}
        gpt.finetune(train_js1, val_js1, train_configs, saving_checkpoint=False)
        gpt.finetune(train_js2, val_js2, train_configs, saving_checkpoint=False)
        gpt.finetune(train_js1, val_js1, train_configs, saving_checkpoint=False)
        gpt.finetune(train_js2, val_js2, train_configs, saving_checkpoint=False)
    # main
    if method in ['ift', 'ft']:
        train_configs={'learning_rate': 1e-4, 'batch_size': args.batch_size, 'epochs':15,  'weight_decay': 0.01, 'warmup_steps': 6}
        gpt.finetune(train_js, val_js, train_configs, saving_checkpoint=False)
    err = []
    try:
        for k in range(2):
            ans, outputs = query(gpt, test_prompts, bs=args.batch_size)
            y_pred = [prompt2value(x) for x in ans]
            err1 = get_accuracy(y_pred, y_test)
            err.append(err1)
    except:
        err.append(None)
    accs.append(err)
    results = " ".join(["%.2f" % x for x in err])
    log(logf, f"{n_train} {results}")
    print(n_train, err)

print(accs)
logf.close()