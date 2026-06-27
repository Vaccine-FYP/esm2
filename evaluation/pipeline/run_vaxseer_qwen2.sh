subtype=$1
year=$2
device=$3
month="02"
ckpt_root_dir="../runs"

# qwen2 dominance checkpoint (trained on colab, copied to flu_lm_qwen2_eval)
lm_ckpt="$ckpt_root_dir/flu_lm_qwen2_eval/2003-10_to_"$year"-"$month"_2M/$subtype/checkpoints/last.ckpt"
# antigenicity predictor is unchanged (original esm model)
hi_ckpt=$(ls $ckpt_root_dir/flu_hi_msa_regressor/before_$year-$month/"$subtype"_seed=1005/random_split/max_steps_150k/lightning_logs/version_0/checkpoints/epoch=*-step=*.ckpt)
echo $lm_ckpt
echo $hi_ckpt

year_minus_three=`(expr $year - 3)`
year_plus_one=`(expr $year + 1)`
index=`expr \( $year - 2018 \) \* 2 + 30`
testing_time=$index
# qwen2 checkpoint is a 2M model so use the binned testing time
min_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 5`
max_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 7`

# [1] vaxseer-selected vaccines, written to a qwen2-specific dir
candidate_vaccine_path="../data/gisaid/ha_processed/$year_minus_three-"$month"_to_$year-"$month"_9999M/$subtype/human_minBinSize1000_minlenQuantile0.2_minCnt5.fasta"
testing_viruses_path="../data/gisaid/ha_processed/$year_minus_three-$month""_to_$year-$month""_9999M/$subtype/human_minBinSize1000_minlenQuantile0.2_minCnt5.fasta"
working_dir="../runs/pipeline_qwen2/$year-$month/$subtype/vaccine_set=$year_minus_three-$month-$year-$month""___virus_set=$year_minus_three-$month-$year-$month"
bash pipeline/run.sh --candidate_vaccine_path $candidate_vaccine_path --testing_viruses_path $testing_viruses_path --working_directory $working_dir --devices $device, --hi_predictor_ckpt $hi_ckpt --min_testing_time $min_testing_time --max_testing_time $max_testing_time --domiance_predictor_ckpt $lm_ckpt --model_type qwen2_time

# [2] WHO vaccines (same antigenicity model, qwen2 dominance)
candidate_vaccine_path="../data/recommended_vaccines_from_gisaid_ha/$year-$year_plus_one"_NH_"$subtype.fasta"
testing_viruses_path="../data/gisaid/ha_processed/$year_minus_three-$month""_to_$year-$month""_9999M/$subtype/human_minBinSize1000_minlenQuantile0.2_minCnt5.fasta"
working_dir="../runs/pipeline_qwen2/$year-$month/$subtype/vaccine_set=who___virus_set=$year_minus_three-$month-$year-$month"
bash pipeline/run.sh --candidate_vaccine_path $candidate_vaccine_path --testing_viruses_path $testing_viruses_path --working_directory $working_dir --devices $device, --hi_predictor_ckpt $hi_ckpt --min_testing_time $min_testing_time --max_testing_time $max_testing_time --domiance_predictor_ckpt $lm_ckpt --model_type qwen2_time
