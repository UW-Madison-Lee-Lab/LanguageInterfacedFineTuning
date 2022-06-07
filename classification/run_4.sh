

# python ./run_exps/run_gpt3_mnist.py -n -t sign -e 0.01 -j 1 --openai_key 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 > run_gpt3_mnist_noisy_sign_0_01 2>&1 &
# python ./run_exps/run_gpt3_mnist.py -n -t sign -e 0.1 -j 1 --openai_key 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 > run_gpt3_mnist_noisy_sign_0_1 2>&1 &
# python ./run_exps/run_gpt3_mnist.py -n -t sign -e 0.3 -j 1 --openai_key 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 > run_gpt3_mnist_noisy_sign_0_3 2>&1 &



# python ./run_exps/run_gpt3_mnist.py -n -t const -e 0.01 -j 1 --openai_key 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 > run_gpt3_mnist_noisy_const_0_01 2>&1 &
#python ./run_exps/run_gpt3_mnist.py -n -t const -e 0.1 --openai_key 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 > run_gpt3_mnist_noisy_const_0_1_rerun 2>&1 &
# python ./run_exps/run_gpt3_mnist.py -n -t const -e 0.3 -j 1 --openai_key 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 > run_gpt3_mnist_noisy_const_0_3 2>&1 &


# export CUDA_VISIBLE_DEVICES=0
# for m in lenet mlp
# do
#     for e in 0.01 0.1 0.3      
#     do
#         python ./run_exps/run_baseline_mnist.py -m $m -t sign -e $e -v 1
#     done
# done


