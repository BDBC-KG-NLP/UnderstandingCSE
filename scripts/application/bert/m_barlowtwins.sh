#!/bin/bash

mkdir -p results/runs
loss_fn_settings=barlowtwins.w_gd.w_w.w_r
lr=1e-5
pooler_type=cls
model_type=bert
seed=42

lambda=1.00
m=0.30
temp=5e-2
ratio=1.50

output_dir=results/runs/application/cls/bert/m_barlowtwins/${seed}

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
    --loss_param_settings lambda=${lambda},m=${m},temp=${temp},ratio=${ratio} \
    --num_layers 12 \
    --mt_size 768 \
    --feature_size 768 \
    --pooler_type $pooler_type \
    --model_type $model_type \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --seed $seed

bash scripts/evaluation.sh \
    $output_dir

python nop.py
