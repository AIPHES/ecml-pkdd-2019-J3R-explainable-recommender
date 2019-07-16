#!/bin/bash
declare -a dataset=("yelp" "music" "kindle" "cds" "electronics" "tv" "toys")
nsize=${#dataset[@]}
for (( i=0; i<${nsize}; i++ ));
do
    mkdir -p models/pointer/${dataset[$i]}/
    mkdir -p data/tensorboard/pointer/${dataset[$i]}/
    echo "CUDA_VISIBLE_DEVICES=3 python train.py -data data/processed/${dataset[$i]}/${dataset[$i]} -save_model models/pointer/${dataset[$i]}/model -share_embeddings -copy_attn -reuse_copy_attn -world_size 1 -tensorboard_log_dir data/tensorboard/pointer/${dataset[$i]}/ -train_steps 50000 -save_checkpoint_steps 5000 -decay_steps 5000 -start_decay_steps 5000 -gpu_ranks 0;"
    CUDA_VISIBLE_DEVICES=3 python train.py -data data/processed/${dataset[$i]}/${dataset[$i]} -save_model models/pointer/${dataset[$i]}/model -share_embeddings -copy_attn -reuse_copy_attn -world_size 1 -tensorboard_log_dir data/tensorboard/pointer/${dataset[$i]}/ -train_steps 50000 -save_checkpoint_steps 5000 -decay_steps 5000 -start_decay_steps 5000 -gpu_ranks 0;
    mkdir -p models/attention/${dataset[$i]}/
    mkdir -p data/tensorboard/attention/${dataset[$i]}/
    echo "CUDA_VISIBLE_DEVICES=3 python train.py -data data/processed/${dataset[$i]}/${dataset[$i]} -save_model models/attention/${dataset[$i]}/${dataset[$i]} -share_embeddings -world_size 1 -tensorboard_log_dir data/tensorboard/attention/${dataset[$i]}/ -train_steps 50000 -save_checkpoint_steps 5000 -decay_steps 5000 -start_decay_steps 5000 -gpu_ranks 0;"
    CUDA_VISIBLE_DEVICES=3 python train.py -data data/processed/${dataset[$i]}/${dataset[$i]} -save_model models/attention/${dataset[$i]}/${dataset[$i]} -share_embeddings -world_size 1 -tensorboard_log_dir data/tensorboard/attention/${dataset[$i]}/ -train_steps 50000 -save_checkpoint_steps 5000 -decay_steps 5000 -start_decay_steps 5000 -gpu_ranks 0;
    
done
