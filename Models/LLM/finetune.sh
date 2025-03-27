#!/bin/bash
DATASET_DIR=PATH TO THE TEXT DATA # COMMENT OUT THIS VARIABLE IN THE FIRST FINETUNE USING THE HUGGINGFACE DATASET 
RESULT_DIR=PATH TO SAVE MODEL, DATA, LOGS
srun accelerate launch \
    --config_file=accelerate_fsdp_config_xgpus.yaml \
     --num_processes=2 \
     --num_machines=1 \
     --machine_rank=0 \
     pllama_7b_instruct_fsdp.py \
     --dataset_dir ${DATASET_DIR} \
     --output_dir ${RESULT_DIR} \
     --batch_size 4\
     --num-workers 10
