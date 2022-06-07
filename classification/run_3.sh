#run_3.sh

#Generate PGD from MLP 

#python ./utils/prepare_mnist.py -a 1 -e 0.1 --source mlp --target lenet
#python ./utils/prepare_mnist.py -a 1 -e 0.01 --source mlp --target lenet


# python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -v 1 -n 1 -u 1 -e 0.01 > run_gptj_mnist_noise_0_01 2>&1
# python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -v 1 -n 1 -u 1 -e 0.1 > run_gptj_mnist_noise_0_1 2>&1
# python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -v 1 -n 1 -u 1 -e 0.3 > run_gptj_mnist_noise_0_3 2>&1


#python ./utils/prepare_mnist.py -n -t const -e 0.001
#python ./utils/prepare_mnist.py -n -t const -e 0.3
#python ./run_exps/run_gptj_mnist_perturbed.py -m 100 -d mnist -v 1 -n 1 -t const -e 0.001 #> run_gptj_mnist_constant_noise_0_01 2>&1

#python ./utils/prepare_mnist.py -n -t const -e 0.01
#python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -v 1 -n 1 -t const -e 0.01 > run_gptj_mnist_constant_noise_0_01_with_10000_samples 2>&1

#export CUDA_VISIBLE_DEVICES=1
#python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -v 1 -n 1 -t const -e 0.1 > run_gptj_mnist_constant_noise_0_1_with_10000_samples 2>&1 
#python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -v 1 -n 1 -t const -e 0.3 > run_gptj_mnist_constant_noise_0_3_with_10000_samples 2>&1
#python ./utils/prepare_mnist.py -n -t const -e 0.1
#python ./utils/prepare_mnist.py -n -t const -e 0.3


#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 ./run_exps/run_gptj_mnist_perturbed.py -d mnist -m 100 -v 1 -n 1 -t normal -e 0.01

#export CUDA_VISIBLE_DEVICES=1
# python ./utils/prepare_mnist.py -n -t normal -e 0.02
# python ./run_exps/run_gptj_mnist_perturbed.py -d mnist -m 100 -v 1 -n 1 -t normal -e 0.02 #> run_gptj_mnist_normal_noise_0_01_with_10000_samples 2>&1

#python ./run_exps/run_gpt3_mnist.py -n -t normal -e 0.01 --openai_key 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 > run_gpt3_mnist_noisy_normal_0_01 2>&1 &
#python ./run_exps/run_gpt3_mnist.py -n -t normal -e 0.02 --openai_key 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 > run_gpt3_mnist_noisy_normal_0_02 2>&1 &
#python ./run_exps/run_gpt3_mnist.py -n -t normal -e 0.05 --openai_key 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 > run_gpt3_mnist_noisy_normal_0_05 2>&1 &
# python ./run_exps/run_gpt3_mnist.py -n -t normal -e 0.1 --openai_key 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 > run_gpt3_mnist_noisy_normal_0_1_rerun 2>&1 &
# python ./run_exps/run_gpt3_mnist.py -n -t normal -e 0.3 --openai_key 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 > run_gpt3_mnist_noisy_normal_0_3 2>&1 &

