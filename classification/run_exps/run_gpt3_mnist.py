import sys
sys.path.append('./')
sys.path.append('./../')
import json
import os
import numpy as np
import torch
import importlib
import openai
import time
from models.GPT3FineTuner import GPT3FineTuner
from utils import mnist
import utils.configs as cfgs
import argparse
from utils.helper import log
parser = argparse.ArgumentParser(description='GPT3-MNIST')
parser.add_argument("-a", "--adv", action='store_true')
parser.add_argument("--source", default='lenet', type=str)
parser.add_argument("-n", "--noisy", action="store_true")
parser.add_argument("-t", "--type", default='const', type=str, choices=['const', 'unif', 'normal', 'sign'])
parser.add_argument("-d", "--data_name", default='mnist', type=str,choices=['mnist','fmnist'])
parser.add_argument("-e", "--eps", default=0.3, type=float)
parser.add_argument("-j", "--job_clear", default=0, type=int)
parser.add_argument("-o", "--openai_key", default='', type=str)
parser.add_argument("-p", "--is_permuted", action="store_true")

args = parser.parse_args()

device = torch.device('cuda:0') if torch.cuda.is_available() else 'cpu'


def operate_all_FineTunes_on_openai(op='list', openai_key = ''):
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


def prompt2value(x):
    # print("Output:",x)
    c = x.strip().split('@@@')[0]
    if c == '':
        return None
    try:
        return int(c)
    except:
        return None

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
    return completions

def generate_output(gpt3_fine_tuner, val_prompts):
    # Validation
    ans, bs, count = [], 20, 0
    while count < len(val_prompts):
        start, end = count, min(count + bs, len(val_prompts))
        count = end
        batch = val_prompts[start:end]
        # print('Input:',batch)
        ans += gpt3_fine_tuner.query(batch)
    return [prompt2value(x) for x in ans]

def get_accuracy(y_pred_val, y_val):
    acc_val = (np.asarray(y_pred_val) == np.asarray(y_val)).mean()
    acc_val = round(acc_val * 100, 2)
    return acc_val


data_name = args.data_name#'mnist' #'fmnist'
is_adv = args.adv
is_noisy = args.noisy
eps_val = args.eps #0.3
is_permuted = args.is_permuted
permuted = 'permuted_' if is_permuted else ''
adv = '_adv' if is_adv else ''
if is_noisy:
    if args.type == 'const':
        noisy = '_noisy_const'
    elif args.type == 'unif':
        noisy = '_noisy_uniform' 
    elif args.type == 'normal':
        noisy = '_noisy_normal'    
    elif args.type == 'sign':
        noisy = '_noisy_sign'    
    else:
        raise NotImplementedError
else:
    noisy = ''
eps = f'_{eps_val}'.replace('.', '_') if is_adv or is_noisy else ''
source = f'_{args.source}'

train_js =  f'data/{permuted}{data_name}_train.jsonl'
val_js = f'data/{permuted}{data_name}_val.jsonl'
if is_noisy:
    test_js = f'data/{permuted}{data_name}{noisy}{eps}_test.jsonl'
else:
    test_js = f'data/{permuted}{data_name}{adv}{source}{eps}_test.jsonl'
    
val_prompts = extract_prompts(val_js)
test_prompts = extract_prompts(test_js)

val_completions = extract_completion(val_js)
y_val = [prompt2value(x) for x in val_completions]
test_completions = extract_completion(test_js)
y_test = [prompt2value(x) for x in test_completions]

openai.api_key = args.openai_key
openai_config = {'model_type':'ada', "num_epochs":5, "batch_size":128}

if args.job_clear:
    operate_all_FineTunes_on_openai(op='cancel', openai_key=openai.api_key)
gpt3_fine_tuner = GPT3FineTuner(openai_config, train_js, val_js)
clf_cfgs = {'n_classes': 10}
gpt3_fine_tuner.fine_tune(clf_cfgs)


y_pred_val = generate_output(gpt3_fine_tuner, val_prompts)
y_pred = generate_output(gpt3_fine_tuner, test_prompts)

acc_val = get_accuracy(y_pred_val, y_val)
acc = get_accuracy(y_pred, y_test)

log_fpath = f'results/evals/clf_gpt3_{permuted}{data_name}.txt'
logf = open(log_fpath, 'a+')
message = f"{permuted}{data_name} {acc_val} {acc}"
log(logf, message)

print('Val accuracy', acc_val)
print('Test accuracy', acc)
