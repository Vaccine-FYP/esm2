year=$1 # "2018"
subtype=$2 # a_h3n2
gpu=$3 # 0
dir_ckpt="$4"
echo $year $subtype $gpu
split_method="random_split"
seed="1005"
month="02"
root_dir="$dir_ckpt/before_$year-$month/$subtype""_seed=$seed/random_split/max_steps_150k"
train_index_path="../data/antigenicity/hi_processed/before_$year-$month/"$subtype"_seed=$seed/random_split/train.csv"
valid_index_path="../data/antigenicity/hi_processed/before_$year-$month/"$subtype"_seed=$seed/random_split/valid.csv"
data_path="../data/gisaid/ha.fasta"
nohup python -m bin.train \
    --default_root_dir $root_dir \
    --data_module hi_regression_esm2 \
    --model esm2_regressor \
    --esm2_model esm2_t30_150M_UR50D \
    --freeze_esm2 true \
    --accelerator gpu \
    --devices $gpu, \
    --batch_size 16 \
    --learning_rate "1e-4" \
    --num_workers 8 \
    --precision bf16 \
    --max_epochs -1 \
    --max_steps 150000 \
    --train_index_path $train_index_path \
    --valid_index_path $valid_index_path \
    --category false > nohup.train_hi_predictor_esm2.$subtype.$year.log 2>&1 &
