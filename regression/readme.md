# LIFT: Regression

We have performed extensive experiments to investigate the performance of LIFT in standard regression tasks. In this folder, we introduce how to reproduce our experiment results based on the following sets of experiments:

* Regression tasks on synthetic datasets 
* Regressoin tasks on real datasets

## Regression tasks on synthetic datasets

Working directory: `.`

In this experiment, we conduct experiments on synthetic datasets with different number of training samples, features, and also investigate the regression (extrapolation) performance of various methods including LIFT/GPT-3, LIFT/GPT-J, and strong baselines such as Gaussian Process, Gradient Boosting Tree, and Multilayer Perceptron. 

### Loading python modules

```python
import sys, os, json
sys.path.insert(1, 'utils')
from run_exp import *
```

### Running experiments on LIFT

We start generating datasets and finetuning GPT-3. 

Calling the following function will run a group of experiments with different data, number of features, different numeric input types, etc. For example, if we specify `p_list = [1,50]`, `data_list=['linear', 'exponential']`, calling the following functions will generate 4 datasets that have i) linear relationship with 1 feature, ii) exponential relationship with 1 feature, iii) linear relationship with 50 features, and iv) exponential relationship with 50 features. The `=` after each parameter gives the default setting. 

```python
run_setting_gpt3(
  data_dir, # the directory for saving the generated synthetic datasets, experiments results and the GPT-3 model information
  n_sims = 3, # repeat each experiments 3 times
  n_train = 200, # generate the datasets with 200 training samples
  n_valid = 50, # generate the datasets with 50 training samples
  n_test = 100, # generate the datasets with 100 training samples
  data_list = ['linear', 'exponential'], # run experiments on both linear and exponential datasets. Options: linear, quadratic, cosine, exponential, l1norm, piecewise
  p_list = [1,50,120], # the number of features the datasets will have
  integer_list = [False], # the input type the datasets will have, True indicates integers, False indicates real numbers
  lb_ub_list = [(-10,10)], # each tuple indicates the range of x_1, ..., x_p
  noise_level_list = [0.1], # the list of gaussian noise levels
  beta = None, # the slope of the linear dataset, default is np.ones(p)*0.9
  resolution = 200, # the resolution for visualization, will generate a grid dataset with 200 samples
  num_epochs = 10, # number of epochs
  batch_size = 5, # batch size
  grid_int = None, # manually specify the interval for generating the grid dataset
  donut = False, # whether we want the training samples to be donut-shaped
  outliers = None, # adding outliers (X_out, y_out)
  lr_list = [0.05, 0.1, 0.2], # select learning rate multiplier in GPT-3 from lr_list
  openai_key = '[REPLACE IT WITH YOUR OPENAI KEY]', # openai key
  valid_temperature = 0.75 # the randomness temperature of GPT-3 prediction
)

```

Then, we can finetune GPT-J. Most of the parameters are identical to `run_setting_gpt3`. 

```python
run_setting_gptj(
  data_dir, # where to load the data and store the experiment results
  n_sims = 3, # repeat each experiment for 3 times
  n_train = 200, 
  n_valid = 50, 
  n_test = 100, 
  data_list = ['linear'], 
  p_list = [1,50,120],
  integer_list = [False], 
  lb_ub_list = [(-10,10)],
  noise_level_list = [0.1], 
  epochs = [2,6,10], # select the epochs from this list
  cuda_idx = 0, # which GPU to use
  metric = 'RAE', 
  batch_size = 4
)
```

The results of each experiments will be saved into files end with `all.json`.

## Regression tasks on real datasets

Working directory: `realdata`.

### Loading python modules

```python
from run_real import run_setting_gpt3
from run_real import run_setting_gptj
```

### Running experiments on LIFT

Note that most of the parameters are the identical to that of synthetic datasets.

```python
run_setting_gpt3(
  data_dir, 
  n_sims = 3, 
  num_epochs = 10, 
  batch_size = 5,
  data_list = ['servo', 'CCPP', 'insurance'], # options: servo, CCPP, insurance, student
  lr_list = [0.05, 0.1, 0.2],
  prefix_list = ['_', '_fn_'], # '_': without feature name; '_fn_': with feature name
  pc_list = ['20', '40', '60', '80', 'full'], # percentage of the dataset used for training
  openai_key = '[REPLACE IT WITH YOUR OPENAI KEY]',
  valid_temperature = 0.75
)
```

```python
run_setting_gptj(
	data_dir, 
	n_sims = 3, 
	data_list = ['servo', 'CCPP', 'insurance'], 
	lr_list = [0.05, 0.1, 0.2],
	prefix_list = ['_', '_fn_'],
	pc_list = ['20', '40', '60', '80', 'full'],
	epochs =  [2,6,10],
	cuda_idx = 0,
	batch_size = 4
)
```

The results of each experiments will be saved into files end with `all.json`.