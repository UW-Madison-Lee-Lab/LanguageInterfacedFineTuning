#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append('./')
sys.path.append('./../')
import json
import os
import numpy as np
import time
import torch
import random

from utils.helper import query
from models import ddp_lora_gptj as lora_gptj
import utils.configs as cfgs

import argparse
parser = argparse.ArgumentParser(description='GPT')
parser.add_argument("-d", "--data_name", default='mnist', type=str,choices=['mnist','fmnist'])
parser.add_argument("-e", "--eps", default=0.3, type=float)
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("-a", "--adv", action='store_true')
parser.add_argument("--source", default='lenet', type=str)
parser.add_argument("-p", "--is_permuted", action="store_true")
parser.add_argument("-n", "--noisy", default=0, type=int)
parser.add_argument("-t", "--type", default='const', type=str, choices=['const', 'unif', 'normal', 'sign'])
parser.add_argument("-s", "--sigma", default=1.0, type=float)
parser.add_argument("-m", "--n_samples", default=10000, type=int)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--seed", default=12345, type=int)
parser.add_argument("-v", "--eval", default=0, type=int)
parser.add_argument("--noisy_train", default=0, type=int)
#parser.add_argument("--save_ckpt", default=0, type=int)


def setup_args_gpu(args):
    """
     Setup arguments CUDA, GPU & distributed training
    """
    if args.local_rank == -1:  # single-node multi-gpu (or cpu) mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # distributed mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    ws = os.environ.get('WORLD_SIZE')

    args.distributed_world_size = int(ws) if ws else 1


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)



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


args = parser.parse_args()
start_time = time.time()

setup_args_gpu(args)
set_seed(args)

data_name = args.data_name #'fmnist'
is_adv = args.adv # False
eps_val = args.eps #0.3
is_noisy = args.noisy
#is_uniform = args.uniform
sigma_val = args.sigma
is_permuted = args.is_permuted #True
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
sigma = f'_{sigma_val}'.replace('.', '_') if is_noisy else ''
is_noisy_train = '_noisy_train' if args.noisy_train else ''

fname = f'{permuted}{data_name}{is_noisy_train}'
source = f'_{args.source}'

if args.noisy_train:
    train_js =  f'data/{permuted}{data_name}{noisy}{eps}_train.jsonl'
    val_js = f'data/{permuted}{data_name}{noisy}{eps}_val.jsonl'
else:
    train_js =  f'data/{permuted}{data_name}_train.jsonl'
    val_js = f'data/{permuted}{data_name}_val.jsonl'

if is_noisy:
    test_js = f'data/{permuted}{data_name}{noisy}{eps}_test.jsonl'
else:
    test_js = f'data/{permuted}{data_name}{adv}{source}{eps}{noisy}{sigma}_test.jsonl'
print(train_js)
print(val_js)
print(test_js)

val_prompts = extract_prompts(val_js)
test_prompts = extract_prompts(test_js)

val_completions = extract_completion(val_js)
y_val = [prompt2value(x) for x in val_completions]
test_completions = extract_completion(test_js)
y_test = [prompt2value(x) for x in test_completions]

if args.eval == 0:
    model_name=f'results/gpt-j/{fname}/pytorch_model.bin'
    print(model_name)
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    
    print('count', torch.cuda.device_count())
    # device = torch.device('cpu')
    # operate_all_Files_on_openai(op='delete')
    train_configs={'learning_rate': 1e-5, 'batch_size':1, 'epochs':1,  'weight_decay': 0.01, 'warmup_steps': 6, 'local_rank': args.local_rank}
    gpt = lora_gptj.LoRaQGPTJ(adapter=True, device=device) #, model_path=model_name)
    if os.path.isfile(model_name):
        gpt.model.load_state_dict(torch.load(model_name))
    gpt.finetune(train_js, val_js, train_configs, saving_checkpoint=False) #args.save_ckpt)
    gpt.save_networks(output_dir = f'results/gpt-j/{fname}/')
else:
    device = torch.device(f"cuda:{args.gpu_id}") if torch.cuda.is_available() else 'cpu'
    print(device)
    torch.cuda.set_device(args.gpu_id)
    gpt = lora_gptj.LoRaQGPTJ(adapter=True, device=device) #device='cpu') # device=device
    model_name=f'results/gpt-j/{fname}/pytorch_model.bin'
    gpt.model.load_state_dict(torch.load(model_name, map_location=device)) #, map_location=device) #'cpu')
    try:
        n_samples=args.n_samples
        if args.gpu_id == 0:
            bs=16 # 1 
        else:
            bs=16 # 32
        ans, outputs = query(gpt, test_prompts[:n_samples], bs=bs)
        y_pred = [prompt2value(x) for x in ans]
        acc = get_accuracy(y_pred, y_test[:n_samples])
        print(acc)
        #from IPython import embed; embed()
    except:
        from IPython import embed; embed()

end_time = time.time()

print('running time: ', end_time - start_time)