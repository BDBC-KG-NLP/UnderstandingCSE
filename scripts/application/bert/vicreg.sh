#!/bin/bash

mkdir -p results/runs
loss_fn_settings=vicreg
lr=1e-5
pooler_type=cls
seed=42

lambda=3e1

output_dir=results/runs/application/cls/bert/vicreg/${seed}

python train.py \
    --model_name_or_path ~/download/model/bert-base-uncased \
    --train_file data/wiki.jsonl \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model stsb_spearman \
    --load_best_model_at_end \
    --eval_steps 125 \
    --learning_rate $lr \
    --num_train_epochs 1 \
    --per_device_train_batch_size 128 \
    --loss_fn_settings $loss_fn_settings \
    --loss_param_settings lambda=${lambda} \
    --num_layers 12 \
    --mt_size 768 \
    --feature_size 768 \
    --pooler_type $pooler_type \
    --model_type bert \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --seed $seed

bash scripts/evaluation.sh \
    $output_dir

python nop.py
