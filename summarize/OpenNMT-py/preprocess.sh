declare -a dataset=("yelp" "music" "cds" "tv" "toys" "kindle" "electronics")
nsize=${#dataset[@]}
for (( i=0; i<${nsize}; i++ ));
do
    mkdir -p data/processed/${dataset[$i]}/
    python preprocess.py -train_src data/${dataset[$i]}/train.src -train_tgt data/${dataset[$i]}/train.tgt -valid_src data/${dataset[$i]}/valid.src -valid_tgt data/${dataset[$i]}/valid.tgt -save_data data/processed/${dataset[$i]}/${dataset[$i]} -share_vocab -dynamic_dict -src_vocab_size 50000
done