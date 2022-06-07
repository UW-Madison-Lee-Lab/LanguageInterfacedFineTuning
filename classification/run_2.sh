
# baseline - imb - use majarity class as the positive class
for DID in 1444 1467 1511
do
    python ./run_exps/run_baselines.py \
        -t imbalance \
        -d $DID 
done


# # gpt3 - imb - use majarity class as the positive class
# for DID in 1444 1511 #TOD: idx 2 of dataset1
# do
#     for LRM in 0.05 0.1 0.2
#     do 
#         for RUN_IDX in 0 1 2
#         do
#             echo $DID $LRM $RUN_IDX
#             python ./run_exps/run_gpt3.py \
#                 -t imbalance \
#                 -d $DID \
#                 -p 15 \
#                 -l $LRM \
#                 -o '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' \
#                 -i $RUN_IDX \
#                 -y 'ada'
#         done
#     done
# done