#python ./utils/prepare_mnist.py -n -u -e 0.05

#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ./run_exps/run_gptj_mnist.py -d fmnist
#python ./run_exps/run_gptj_mnist.py -d fmnist > log_train_fmnist 2>&1

#python ./run_exps/run_baseline_mnist.py -m lenet -e 0.3 
# python ./run_exps/run_baseline_mnist.py -m mlp -e 0 -v 1
# python ./run_exps/run_baseline_mnist.py -m mlp -e 0.01 -v 1
# python ./run_exps/run_baseline_mnist.py -m mlp -e 0.1 -v 1
# python ./run_exps/run_baseline_mnist.py -m mlp -e 0.3 -v 1

# python ./run_exps/run_baseline_mnist.py -m lenet -e 0 -v 1
# python ./run_exps/run_baseline_mnist.py -m lenet -e 0.01 -v 1
# python ./run_exps/run_baseline_mnist.py -m lenet -e 0.1 -v 1
# python ./run_exps/run_baseline_mnist.py -m lenet -e 0.3 -v 1

#python ./run_exps/run_gpt3_mnist.py -n -u -e 0.01 --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_noise_0_01 2>&1 &
#python ./run_exps/run_gpt3_mnist.py -n -u -e 0.1 --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_noise_0_1 2>&1 &
#python ./run_exps/run_gpt3_mnist.py -n -u -e 0.3 --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_noise_0_3 2>&1 &



#python ./run_exps/run_gpt3_mnist.py -a -e 0.01 --source mlp --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_adv_mlp_0_01 2>&1 &
#python ./run_exps/run_gpt3_mnist.py -a -e 0.1 --source mlp --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_adv_mlp_0_1 2>&1 &
#python ./run_exps/run_gpt3_mnist.py -a -e 0.3 --source mlp -j 1 --openai_key [REPLACE IT WITH YOUR OPENAI KEY] > run_gpt3_mnist_adv_mlp_0_3 2>&1 &

# for e in 0.02 0.05 0.1 0.3
# do
#     python ./utils/prepare_mnist.py -n -t sign -e $e
# done

#export CUDA_VISIBLE_DEVICES=1
#python ./utils/prepare_mnist.py -n -t sign -e 0.1 --noisy_train 1
python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -v 0 --noisy_train 1 -n 1 -t sign -e 0.1 -g 0 #> log_gptj_train_noisy_sign_0_1_mnist 2>&1

#python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -v 1 -a -e 0.3 --source mlp > run_gptj_mnist_adv_mlp_0_3 2>&1
#python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -v 1 -a -e 0.1 --source mlp > run_gptj_mnist_adv_mlp_0_1 2>&1
#python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -v 1 -a -e 0.01 --source mlp > run_gptj_mnist_adv_mlp_0_01 2>&1 # this will be rerun soon!


#python ./utils/prepare_mnist.py -n -t sign -e 0.01

# for e in 0.01 0.1 0.3
# do
#     python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -m 100 -v 1 -n 1 -t sign -e $e #> run_gptj_mnist_sign_noise_0_01_with_10000_samples 2>&1 
# done
