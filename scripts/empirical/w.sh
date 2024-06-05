#!/bin/bash

mkdir -p results/runs
seed=42
temp=3e-3
output_dir=results/runs/empirical/cls/w/$temp

python train.py \
    --model_name_or_path ~/download/model/bert-base-uncased \
    --train_file data/wiki.jsonl \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --learning_rate 10e-6 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --loss_fn_settings sg.w_gd.w_w.w_r \
    --loss_param_settings "m=0.30,ratio=1.00,temp=${temp}" \
    --num_layers 12 \
    --mt_size 768 \
    --feature_size 768 \
    --pooler_type cls \
    --model_type bert \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --seed $seed

bash scripts/evaluation.sh \
    $output_dir

python nop.py