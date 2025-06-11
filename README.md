## 🚀 Inference

To run inference with Open-Nav, use the provided script:

```bash
bash run_OpenNav.bash
```

### 🔧 Choosing the Language Model
You can specify which LLM to use via the --llm argument in the script. Supported options include:

	• gpt4o (default): Uses GPT-4o via OpenAI API
	• Qwen2, Llama3.1, Gemma, Phi3, etc.: Open-source LLMs (require local deployment)

⚠️ Open-source LLMs must be deployed separately and configured before use.


### 📐 Modifying Evaluation Episodes
To change the number of evaluation episodes, edit the following field in:
```
habitat_extensions/config/vlnce_task.yaml
```
Locate this section and modify EPISODES_TO_LOAD:

```yaml
DATASET:
  TYPE: VLN-CE-v1
  SPLIT: val_unseen
  DATA_PATH: data/datasets/R2R_VLNCE_v1-2_preprocessed/{split}/OpenNav_R2R-CE_100_bertidx.json.gz
  SCENES_DIR: data/scene_datasets/
  EPISODES_TO_LOAD: 1  # Change this to run more episodes
```
