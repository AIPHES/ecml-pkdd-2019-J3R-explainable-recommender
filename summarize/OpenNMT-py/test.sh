#!/bin/bash
declare -a dataset=("yelp" "music" "kindle" "cds" "electronics" "tv" "toys")
nsize=${#dataset[@]}
declare -a checkpoints=("50000")
csize=${#checkpoints[@]}
for (( i=0; i<${nsize}; i++ ));
do
    mkdir -p data/outputs/attention/${dataset[$i]}/
    mkdir -p data/outputs/pointer/${dataset[$i]}/
    for (( j=0; j<${csize}; j++ ));
    do
        CUDA_VISIBLE_DEVICES=4 python translate.py -model models/attention/${dataset[$i]}/${dataset[$i]}_step_${checkpoints[$j]}.pt -src data/${dataset[$i]}/test.src -o data/outputs/attention/${dataset[$i]}/output_${checkpoints[$j]}_pred.txt -beam_size 10 -gpu 0 -replace_unk;
        CUDA_VISIBLE_DEVICES=4 python translate.py -model models/pointer/${dataset[$i]}/model_step_${checkpoints[$j]}.pt -src data/${dataset[$i]}/test.src -o data/outputs/pointer/${dataset[$i]}/output_${checkpoints[$j]}_pred.txt -beam_size 10 -gpu 0 -replace_unk;
    done
done
