# LIFT Classification

## Overview

This folder contains code to run different classification tasks on GPT-J, GPT-3, and other baseline models on synthetic datasets, OpenML datasets, and MNIST datasets.

## Usage

### installation 

```
pip install openai
pip install openml
pip install transformers==4.14.1
pip install bitsandbytes-cuda111==0.26.0
pip install datasets==1.16.1
pip install einops
```

### OpenAI Key

In order to run GPT-3 on your own, you have to have an OpenAI account and specify your key when you run ```LIFT/classification/run_exps/run_gpt3.py``` or ```LIFT/classification/run_exps/run_gpt3_mnist.py``` by setting up ```-o <your openai key>```

### Prepare datasets
#### Synthetic datasets and OpenML datasets 
All synthetic datasets and OpenML datasets will be automatically prepared if corresponding files don't exist under appropriate directories. 

#### MNIST datasets
You have to prepare MNIST datasets before using them by running ```/home/user0/Desktop/LIFT/classification/utils/prepare_mnist.py```
To choose between MNIST and Fashion-MNIST, set ```-d``` (stands for dataset) to be either ```mnist``` or ```fmnist``` 
To use their permuted variants, further set up boolean argument ```-p``` (stands for is permuted)
To add noises, set up boolean argument ```-n``` (stands for noisy) and specify the noise level sigma ```-s <noise level>```
To create adversarial MNIST examples, specify epsilon ```-e <epsilon>```


All data files will be save under ```./data```

### Run experiments on different datasets

To run experiments on synthetic datasets or OpenML datasets, use ```LIFT/classification/run_exps/run_baselines.py``` or ```LIFT/classification/run_exps/run_gptj.py``` or ```LIFT/classification/run_exps/run_gpt3.py```

You can set up  ```-d``` to specify dataset ID that you want to use. We keep original dataset ID for all OpenML datasets with dataset ID larger then 10. We reserve digits 1-10 for our own synthetic datasets. ```[1:9Gaussians, 2:Blobs, 3:Circle, 4:Moons, 6:TwoCircles ]```

To run experiments on MNIST dataset, use  ```LIFT/classification/run_exps/run_baselines_mnist.py``` or ```LIFT/classification/run_exps/run_gptj_mnist.py``` or ```LIFT/classification/run_exps/run_gptj_mnist_perturbed.py``` or ```LIFT/classification/run_exps/run_gptj_mnist.py```

### choose Type of Tasks to Run (Synthetic Datasets and OpenML Datasets)

For Synthetic Datasets and OpenML Dataset, there're different types of tasks to evaluate different models' performance in different settings. 
To change the type of tasks, change the value of ```-t``` (stands for tasks) choosing from ```[accuracy, imbalance, sampling, label_corruption, teach, nn_gen_data]``` 

Meaning of each arguments:
```accuracy``` Calculate basic classification accuracy
```imbalance``` Calculate basic classification accuracy, f1 score, recall, precision 
```sampling``` To test sample complexity properties of models, automatically run classification using 5%, 10%, 20% 50% of training samples. 
```label_corruption``` To test robustness on label corruption, automatically run classification with 5%, 10% 20% of training labels being corrupted.
```teach``` Test inductive bias of GPT-J and GPT-3.
```nn_gen_data``` Run classification tasks on NN-generated data. In default setting, the script will use data generated from Neural Network after 10, 80, 490 epochs of training.

For example, if you want to run GPT-J on imbalance tasks using OpenML dataset 1444, you can run:
    ```
    python run_exps/run_gptj.py -t imbalance -d 1444
    ```

#### Classification with feature names and in-context classification
To run classification with feature names, set up the boolean argument```-f``` (stand for use feature names)
To run in-context classification, set up the boolean argument```-n``` (stand for in-context)

Note that for classification with feature names, and in-context classification, only datasets used in LIFT paper are supported in our code.

#### Classification with Mixup on OpenML Dataset
To run classification on GPT-J and GPT-3 with a popular data augmentation that makes virtual samples by mixing the features/labels of real samples, set up the boolean argument ```-m``` (stand for mixup)
#### Two-Stage Fine-Tuning on GPT-J
To run two-stage fine-tuning on GPT-J, run ```LIFT/classification/run_exps/run_gptj_inter_ft.py``` and set ```-m``` (stands for method) to be 

### choose Type of Tasks to Run (MNIST Datasets)
#### Basic Classification Tasks on MNIST
To run basic classification using MNIST Dataset, run ```LIFT/classification/run_exps/run_baselines_mnist.py``` or ```LIFT/classification/run_exps/run_gptj_mnist.py``` or ```LIFT/classification/run_exps/run_gpt3_mnist.py```

To choose between MNIST and Fashion-MNIST, set ```-d``` (stands for dataset) to be either ```mnist``` or ```fmnist``` 
To use their permuted variant, further set up boolean argument ```-p``` (stands for is permuted)

#### Robustness to Adversarial Examples
To run GPT-J or GTP3 on MNIST adversarial examples, run ```LIFT/classification/run_exps/run_gptj_mnist_perturbed.py``` and  ```LIFT/classification/run_exps/run_gpt3_mnist.py``` and set up the boolean argument ```-a``` (stands for adversarial). Also, specify epsilon ```-e <epsilon>```

#### Robustness to Gaussian Noises 
To run run ```LIFT/classification/run_exps/run_gptj_mnist_perturbed.py``` with added Gaussian noises, set up boolean argument ```-n``` (stands for noisy) and specify the noise level sigma ```-s <noise level>```

### Get results
Results will be saved under the directory ```LIFT/classification/results```
