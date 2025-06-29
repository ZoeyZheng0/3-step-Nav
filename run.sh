#!/bin/bash

## Full dataset
# episodes_to_load=100

## Debug
episodes_to_load=1

flag="--exp_name debug
      --exp-config run_OpenNav.yaml
      --llm gpt-4o-2024-08-06
      --api_key your-api-key
      --episodes_to_load $episodes_to_load
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      EVAL.SPLIT val_unseen
      "
CUDA_VISIBLE_DEVICES=3 python run.py $flag