#!/usr/bin/env bash
# Run the full downstream evaluation for Qwen2 across multiple years, compare to GPT-2.
# Assumes: GPT-2 pipeline results already exist (../runs/pipeline), and the patched
# run.sh / run_vaxseer_qwen2.sh / sweep scripts are in place.
#
# Usage:  bash run_all_qwen2_eval.sh
# Run from: ~/vaxseer_run/evaluation

set -u
SUBTYPE="a_h3n2"
CKPT_SRC="/mnt/c/Users/barak/Downloads/ckpts"
YEARS=(2012 2013 2014 2015 2016 2017)
DEVICE=0

echo "=================================================="
echo " Qwen2 downstream evaluation: ${YEARS[*]} ($SUBTYPE)"
echo "=================================================="

for year in "${YEARS[@]}"; do
    echo ""
    echo "########## YEAR $year ##########"

    # 1. place the checkpoint where run_vaxseer_qwen2.sh expects it
    dest_dir="../runs/flu_lm_qwen2_eval/2003-10_to_${year}-02_2M/${SUBTYPE}/checkpoints"
    mkdir -p "$dest_dir"
    if [ -f "$CKPT_SRC/${year}_last.ckpt" ]; then
        cp "$CKPT_SRC/${year}_last.ckpt" "$dest_dir/last.ckpt"
        echo "[ok] placed checkpoint for $year"
    else
        echo "[SKIP] no checkpoint $CKPT_SRC/${year}_last.ckpt - skipping $year"
        continue
    fi

    # 2. run the qwen2 prediction pipeline (dominance + antigenicity) for this year
    echo "[run] pipeline for $year ..."
    bash pipeline/run_vaxseer_qwen2.sh "$SUBTYPE" "$year" "$DEVICE" > "log.qwen2_pipeline.$year.txt" 2>&1
    echo "[ok] pipeline done (log: log.qwen2_pipeline.$year.txt)"

    # 3. run the sweeps for this year (point the sweep scripts at this year)
    sed -i "s/seq [0-9]* [0-9]*/seq $year $year/" pipeline/run_sweep_comb_qwen2.sh
    sed -i "s/seq [0-9]* [0-9]*/seq $year $year/" pipeline/run_sweep_comb_who_qwen2.sh
    rm -f nohup.sweep_qwen2.*.log nohup.sweep_who_qwen2.*.log
    bash pipeline/run_sweep_comb_qwen2.sh
    bash pipeline/run_sweep_comb_who_qwen2.sh

    # wait for the background sweep jobs to finish
    echo "[wait] sweeps for $year ..."
    sleep 90
    # extra wait if results not yet present
    for i in 1 2 3 4; do
        n=$(find "../runs/pipeline_qwen2/${year}-02" -name "vaccine_score_and_gt.csv" 2>/dev/null | wc -l)
        if [ "$n" -ge 1 ]; then break; fi
        echo "   still waiting ($i) ..."
        sleep 30
    done

    n=$(find "../runs/pipeline_qwen2/${year}-02" -name "vaccine_score_and_gt.csv" 2>/dev/null | wc -l)
    echo "[ok] $year produced $n score file(s)"
done

echo ""
echo "=================================================="
echo " All years processed. Generating summary ..."
echo "=================================================="
python3 pipeline/summarize_comparison.py "${YEARS[@]}"
