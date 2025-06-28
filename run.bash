#!/bin/bash

episodes_to_load=1

flag="--exp_name debug
      --exp-config run_OpenNav.yaml
      --llm gpt-4o-2024-08-06
      --api_key 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
      --episodes_to_load $episodes_to_load
      SIMULATOR_GPU_IDS [0]
      TORCH_GPU_ID 0
      TORCH_GPU_IDS [0]
      EVAL.SPLIT val_unseen
      "
CUDA_VISIBLE_DEVICES=3 python run.py $flag