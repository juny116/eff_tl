# Efficient transfer learning

## Installation
* Install requirements
  * pip install -r requirements
* Install DeepSpeed
  * DDP~ZeRO1 -> pip install deepspeed
  * DDP~ZeRO3 -> install from source https://www.deepspeed.ai/tutorials/advanced-install/
* Install custom transformers
  * pip install -e .

## How to run
* e.g. deepspeed main.py --task_name sst2 --model_name_or_path gpt2-xl --ds_config ds_configs/ddp.json --output_dir OUTPATH
## Implementation & Tests

|Method         | DDP   | FP16  | ZeRO1 | ZeRO2 | ZeRO3 |
|---            |---    |---    |---    |---    |---    |
|Fine tuning    |O      |O      |O      |O      |O      |
|LoRA           |O      |O      |O      |O      |X      |
|Prefix tuning  |O      |O      |O      |O      |O      |
|Adapter H      |X      |X      |X      |X      |X      |
|Adapter P      |X      |X      |X      |X      |X      |
|Prompt tuning  |X      |X      |X      |X      |X      |
|IDPG (Original)|O      |O      |O      |O      |X      |
|IDPG (Trained) |O      |O      |O      |O      |X      |


## Benchmark Results
### GPT2-XL (1.5B)
|Method            |PARAM | MNLI 10% m | SST-2 | RTE   |MNLI   |
|---               |---   |---         |---    |---    |---    |
|Fine tuning       |100%  |            |       |       |       |
|LoRA              |0.47  |83.6        |       |       |       |
|Prefix tuning     |      |            |       |       |       |
|Adapter H         |      |            |       |       |       |
|Adapter P         |      |            |       |       |       |
|Prompt tuning     |      |            |       |       |       |


## Dataset split details
|Dataset        |Train   | Validation  | Test      |
|---            |---     |---          |---        |
|MNLI           | 392702 | 9815 / 9832 |9796 / 9847|
|SST-2          | -      |     -       |    -      |
|RTE            | 2490   |     277     |    -      |

----

# Input Dependent Prompt
## Best Results
### GPT2-XL (1.5B)
|Method            |PARAM | MNLI 10% m | SST-2 | RTE   |MNLI   |
|---               |---   |---         |---    |---    |---    |
|Fine tuning       |100   |            |       |       |       |
|IGPG              |0.744 |82.28       |       |       |       |
|IGPG (prompt-only)|0.746 |82.52       |       |       |       |

## All Results
|Method            |PARAM |Task       |Learning Rate|Train Epochs|Warmup Step|Accuracy    |
|---               |---   |---        |---          |---         |---        |---         |
|---               |---   |---        |---          |---         |---        |---         |
|IGPG              |100   |MNLI 10% m |1e-4         |40          |1,000      |58.39       |
|IGPG              |100   |MNLI 10% m |5e-5         |40          |1,000      |<b>82.28</b>|
|IGPG              |100   |MNLI 10% m |1e-5         |40          |1,000      |76.09       |
|IGPG (prompt only)|100   |MNLI 10% m |1e-4         |40          |1,000      |<b>82.52</b>|
|IGPG (prompt only)|100   |MNLI 10% m |5e-5         |40          |1,000      |-           |
|IGPG (prompt only)|100   |MNLI 10% m |1e-5         |40          |1,000      |76.26       |
|---               |---   |---        |---          |---         |---        |---         |
|IGPG              |100   |RTE        |5e-3         |120         |1,000      |58.87       |
|IGPG              |100   |RTE        |1e-3         |120         |1,000      |56.74       |
|IGPG              |100   |RTE        |5e-4         |120         |1,000      |53.19       |
|IGPG (prompt only)|100   |RTE        |5e-3         |120         |1,000      |     |
|IGPG (prompt only)|100   |RTE        |1e-3         |120         |1,000      | |
|IGPG (prompt only)|100   |RTE        |5e-4         |120         |1,000      |     |

## Architecture
![image](https://user-images.githubusercontent.com/29649894/146304303-9a773178-470b-4a96-8026-e832d51bcb48.png)

- IDPG : 12620096 / 1694671104 (0.744%)
  - output_processor.score.weight > [3, 1600]
  - input_processor.encoder_generator.0.weight > [384, 768]
  - input_processor.encoder_generator.0.bias > [384]
  - input_processor.encoder_generator.2.weight > [32000, 384]
  - input_processor.encoder_generator.2.bias > [32000]
- IDPG (prompt-only) : 12635456 / 1694686464 (0.746%)
  - output_processor.score.weight > [3, 1600]
  - input_processor.encoder_generator.0.weight > [384, 768]
  - input_processor.encoder_generator.0.bias > [384]
  - input_processor.encoder_generator.2.weight > [32000, 384]
  - input_processor.encoder_generator.2.bias > [32000]
  - input_processor.encoder_prompt_embeddings.weight > [20, 768]
