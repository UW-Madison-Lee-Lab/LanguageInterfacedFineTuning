#!/usr/bin/env python
# coding: utf-8
import json
import os
import numpy as np
import torch

from utils.helper import query
from models import lora_gptj as GPTJ
import utils.configs as cfgs
from run_exps_helper import *

import argparse
parser = argparse.ArgumentParser(description='GPT')
parser.add_argument("-d", "--data_name", default='mnist', type=str,choices=['mnist','fmnist'])
parser.add_argument("-g", "--gpu_id", default=0, type=int)
parser.add_argument("--local_rank", default=-1, type=int)
parser.add_argument("--seed", default=12345, type=int)
parser.add_argument("-p", "--is_permuted", action="store_true")
parser.add_argument("-v", "--eval", default=0, type=int)


args = parser.parse_args()


data_name = args.data_name
is_adv = False
is_permuted = args.is_permuted #True
permuted = 'permuted_' if is_permuted else ''
adv = '_adv' if is_adv else ''

fname = f'{permuted}{data_name}'
train_js =  f'data/{fname}_train.jsonl'
val_js = f'data/{fname}_val.jsonl'
test_js = f'data/{fname}{adv}_test.jsonl'

val_prompts = extract_prompts(val_js)
test_prompts = extract_prompts(test_js)
val_completions = extract_completion(val_js)
y_val = [prompt2value(x) for x in val_completions]
test_completions = extract_completion(test_js)
y_test = [prompt2value(x) for x in test_completions]


device = torch.device(f'cuda:{args.gpu_id}') if torch.cuda.is_available() else 'cpu'
torch.cuda.set_device(args.gpu_id)

gpt = GPTJ.LoRaQGPTJ(adapter=True, device=device)
model_name = f'results/gpt-j/{fname}_best_model.pth'
pretrained_path = f'results/gpt-j/{fname}/pytorch_model.bin'
if os.path.isfile(pretrained_path):
    gpt.model.load_state_dict(torch.load(pretrained_path))
# #### Training
if args.eval == 0:
    train_configs={'learning_rate': 1e-5, 'batch_size': 2, 'epochs':1,  'weight_decay': 0.01, 'warmup_steps': 6}
    gpt.finetune(train_js, val_js, train_configs, saving_checkpoint=True, save_path=model_name, local_rank=args.local_rank)
else:
    gpt.load_networks(model_name)
ans, outputs = query(gpt, test_prompts, bs=16)
y_pred = [prompt2value(x) for x in ans]
acc = get_accuracy(y_pred, y_test)
print(acc)
