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
|Adapter H      |O      |O      |O      |-      |-      |
|Adapter P      |O      |O      |O      |-      |-      |
|Prompt tuning  |O      |O      |O      |O      |O      |
|IDPG           |O      |O      |O      |O      |?      |


## Benchmark Results
### GPT2-XL (1.5B)
|Method            |PARAM | MNLI 10% m | SST-2 | RTE    |MNLI   |MRPC |
|---               |---   |---         |---    |---     |---    |---  |
|Fine tuning       |100%  |            |       |76.6    |       |     |
|LoRA              |0.47  |83.6        |       |        |       |     |
|Prefix tuning     |0.48  |83.1        |       |        |       |     |
|Adapter H         |      |            |       |        |       |     |
|Adapter P         |      |            |       |        |       |     |
|Prompt tuning     |0.005 |80.0        |       |        |       |     |
|IDPG              |0.744 |82.28       |95.65  |60.99(?)|       |75.00|
|PG                |0.746 |82.52       |95.53  |        |       |78.92|



## Dataset split details
|Dataset        |Train    | Validation    | Test        |
|---            |---      |---            |---          |
|MNLI           | 392,702 | 9,815 / 9,832 |9,796 / 9,847|
|RTE            | 2,490   |     277       |    3,000    |
|MRPC           | 3,668   |     408       |    1,725    |
|SST-2          | 67,349  |     872       |    1,821    |

----

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
