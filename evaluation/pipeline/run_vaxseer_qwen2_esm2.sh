subtype=$1
year=$2
device=$3
month="02"
ckpt_root_dir="../runs"

# Qwen2 dominance checkpoint
lm_ckpt="$ckpt_root_dir/flu_lm_qwen2_eval/2003-10_to_"$year"-"$month"_2M/$subtype/checkpoints/last.ckpt"
# ESM-2 antigenicity checkpoint (trained per subtype, reused across years)
hi_ckpt="$ckpt_root_dir/flu_hi_esm2_eval/$subtype/checkpoints/last.ckpt"
echo "dominance (qwen2): $lm_ckpt"
echo "antigenicity (esm2): $hi_ckpt"

year_minus_three=`(expr $year - 3)`
year_plus_one=`(expr $year + 1)`
index=`expr \( $year - 2018 \) \* 2 + 30`
min_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 5`
max_testing_time=`expr \( $year - 2004 \) \* 6 + 1 + 7`

# [1] vaxseer-selected vaccines -> full-modern pipeline dir
candidate_vaccine_path="../data/gisaid/ha_processed/$year_minus_three-"$month"_to_$year-"$month"_9999M/$subtype/human_minBinSize1000_minlenQuantile0.2_minCnt5.fasta"
testing_viruses_path="../data/gisaid/ha_processed/$year_minus_three-$month""_to_$year-$month""_9999M/$subtype/human_minBinSize1000_minlenQuantile0.2_minCnt5.fasta"
working_dir="../runs/pipeline_qwen2_esm2/$year-$month/$subtype/vaccine_set=$year_minus_three-$month-$year-$month""___virus_set=$year_minus_three-$month-$year-$month"
bash pipeline/run.sh --candidate_vaccine_path $candidate_vaccine_path --testing_viruses_path $testing_viruses_path --working_directory $working_dir --devices $device, --hi_predictor_ckpt $hi_ckpt --min_testing_time $min_testing_time --max_testing_time $max_testing_time --domiance_predictor_ckpt $lm_ckpt --model_type qwen2_time --antigenicity_model esm2_regressor

# [2] WHO vaccines
candidate_vaccine_path="../data/recommended_vaccines_from_gisaid_ha/$year-$year_plus_one"_NH_"$subtype.fasta"
testing_viruses_path="../data/gisaid/ha_processed/$year_minus_three-$month""_to_$year-$month""_9999M/$subtype/human_minBinSize1000_minlenQuantile0.2_minCnt5.fasta"
working_dir="../runs/pipeline_qwen2_esm2/$year-$month/$subtype/vaccine_set=who___virus_set=$year_minus_three-$month-$year-$month"
bash pipeline/run.sh --candidate_vaccine_path $candidate_vaccine_path --testing_viruses_path $testing_viruses_path --working_directory $working_dir --devices $device, --hi_predictor_ckpt $hi_ckpt --min_testing_time $min_testing_time --max_testing_time $max_testing_time --domiance_predictor_ckpt $lm_ckpt --model_type qwen2_time --antigenicity_model esm2_regressor
