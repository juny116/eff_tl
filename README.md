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
|Prefix tuning     |0.48  |83.1        |       |       |       |
|Adapter H         |      |            |       |       |       |
|Adapter P         |      |            |       |       |       |
|Prompt tuning     |      |            |       |       |       |


## Dataset split details
|Dataset        |Train    | Validation    | Test        |
|---            |---      |---            |---          |
|MNLI           | 392,702 | 9,815 / 9,832 |9,796 / 9,847|
|RTE            | 2,490   |     277       |    3,000    |
|MRPC           | 3,668   |     408       |    1,725    |
|SST-2          | 67,349  |     872       |    1,821    |

----

# Input Dependent Prompt
## Best Results
### GPT2-XL (1.5B)
|Method            |PARAM | MNLI 10% m | SST-2 | RTE   |MNLI   |
|---               |---   |---         |---    |---    |---    |
|Fine tuning       |100   |            |       |       |       |
|IDPG              |0.744 |82.28       |       |       |       |
|IDPG (prompt-only)|0.746 |82.52       |       |       |       |

## All Results
|Method            |PARAM   |Task       |Learning Rate|Train Epochs|Warmup Step|Accuracy    |
|---               |---     |---        |---          |---         |---        |---         |
|---               |---     |---        |---          |---         |---        |---         |
|IDPG              |0.744   |MNLI 10% m |1e-4         |40          |1,000      |58.39       |
|IDPG              |0.744   |MNLI 10% m |5e-5         |40          |1,000      |<b>82.28</b>|
|IDPG              |0.744   |MNLI 10% m |1e-5         |40          |1,000      |76.09       |
|IDPG (prompt only)|0.746   |MNLI 10% m |1e-4         |40          |1,000      |<b>82.52</b>|
|IDPG (prompt only)|0.746   |MNLI 10% m |5e-5         |40          |1,000      |-           |
|IDPG (prompt only)|0.746   |MNLI 10% m |1e-5         |40          |1,000      |76.26       |
|---               |---     |---        |---          |---         |---        |---         |
|Fine-tuning       |100     |RTE        |1e-4         |20          |1,000      |-           |
|Fine-tuning       |100     |RTE        |5e-5         |20          |1,000      |-           |
|Fine-tuning       |100     |RTE        |1e-5         |20          |1,000      |-           |
|IDPG              |0.744   |RTE        |5e-2         |120         |1,000      |54.61       |
|IDPG              |0.744   |RTE        |1e-2         |120         |1,000      |<b>60.99</b>|
|IDPG              |0.744   |RTE        |5e-3         |120         |1,000      |58.87       |
|IDPG              |0.744   |RTE        |1e-3         |120         |1,000      |56.74       |
|IDPG              |0.744   |RTE        |5e-4         |120         |1,000      |53.19       |
|IDPG              |0.744   |RTE        |1e-4         |120         |1,000      |오류         |
|IDPG              |0.744   |RTE        |5e-5         |120         |1,000      |53.90       |
|IDPG              |0.744   |RTE        |1e-5         |120         |1,000      |오류         |
|IDPG (prompt only)|0.744   |RTE        |5e-2         |120         |1,000      |-           |
|IDPG (prompt only)|0.744   |RTE        |1e-2         |120         |1,000      |-           |
|IDPG (prompt only)|0.746   |RTE        |5e-3         |120         |1,000      |-           |
|IDPG (prompt only)|0.746   |RTE        |1e-3         |120         |1,000      |-           |
|IDPG (prompt only)|0.746   |RTE        |5e-4         |120         |1,000      |-           |
|---               |---     |---        |---          |---         |---        |---         |
|IDPG              |0.744   |RTE        |5e-3         |120         |100        |55.32       |
|IDPG              |0.744   |RTE        |1e-3         |120         |100        |55.32       |
|IDPG              |0.744   |RTE        |5e-4         |120         |100        |52.48       |
|IDPG (prompt only)|0.746   |RTE        |5e-3         |120         |100        |-           |
|IDPG (prompt only)|0.746   |RTE        |1e-3         |120         |100        |-           |
|IDPG (prompt only)|0.746   |RTE        |5e-4         |120         |100        |-           |
|---               |---     |---        |---          |---         |---        |---         |
|IDPG              |0.744   |MRPC       |5e-3         |120         |100        |-           |
|IDPG              |0.744   |MRPC       |1e-3         |120         |100        |-           |
|IDPG              |0.744   |MRPC       |5e-4         |120         |100        |-           |
|IDPG (prompt only)|0.746   |MRPC       |5e-3         |120         |100        |-           |
|IDPG (prompt only)|0.746   |MRPC       |1e-3         |120         |100        |-           |
|IDPG (prompt only)|0.746   |MRPC       |5e-4         |120         |100        |-           |
|---               |---     |---        |---          |---         |---        |---         |
|IDPG              |0.744   |SST-2      |1e-4         |40          |1,000      |            |
|IDPG              |0.744   |SST-2      |5e-5         |40          |1,000      |            |
|IDPG              |0.744   |SST-2      |1e-5         |40          |1,000      |            |
|IDPG (prompt only)|0.746   |SST-2      |1e-4         |40          |1,000      |            |
|IDPG (prompt only)|0.746   |SST-2      |5e-5         |40          |1,000      |            |
|IDPG (prompt only)|0.746   |SST-2      |1e-5         |40          |1,000      |            |
- RTE는 validation set에서는 68% 정도까지 성능이 나오는데, test set에서는 성능이 왜 저렇게 낮게 나오는지 의문... 테스트하는 데이터가 적어서 그런가 (validation : 138 / test : 139)
- 학습 다 하고 마지막에 test accuracy 측정할 때 이런 오류가 나서 모델을 로드 못하는 경우가 생김.
    ![image](https://user-images.githubusercontent.com/29649894/146470309-78a8ed25-d3a2-46c8-b07e-e74300a626c5.png)


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
