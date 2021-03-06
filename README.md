# Efficient transfer learning

## Installation
1. Install requirements
```bash
pip install -r requirements.txt
```
2. Install Custom Transformers Library
```bash
pip install -e .
```
3. Install DeepSpeed
  * If we only want to use <code>DDP~ZeRO1</code>
  ```bash
  pip install deepspeed
  ```
  * For <code>DDP~ZeRO3</code> -> install from source https://www.deepspeed.ai/tutorials/advanced-install/
  ```bash
  cd eff_tl
  
  git clone https://github.com/microsoft/DeepSpeed/
  
  cd DeepSpeed
  
  rm -rf build
  
  # use an appropriate version for TORCH_CUDA_ARCH_LIST
  TORCH_CUDA_ARCH_LIST="6.1;8.6" DS_BUILD_OPS=1 pip install . \
   --global-option="build_ext" --global-option="-j8" --no-cache -v \
   --disable-pip-version-check 2>&1 | tee build.log
  ```
  * [Troubleshooting] : you may want to install following libraries
  ```bash
  sudo apt install libaio-dev
  sudo apt install cmake
  ```
    
## How to run
```
deepspeed main.py 
 --task_name sst2 
 --model_name_or_path gpt2-xl 
 --ds_config ds_configs/ddp.json 
 --output_dir OUTPATH
```

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
|Method            |PARAM | MNLI 10% m | SST-2      |MRPC                | RTE    |MNLI   |
|---               |---   |---         |---         |---                 |---     |---    |
|Fine tuning       |100%  |82.9        |94.1        |                    |76.6    |       |
|LoRA              |0.47  |83.7        |95.2        |                    |        |       |
|Prefix tuning     |0.48  |83.1        |94.95       |                    |        |       |
|Adapter H         |      |            |            |                    |        |       |
|Adapter P         |      |            |            |                    |        |       |
|Prompt tuning     |0.005 |80.0        |            |                    |        |       |
|IDPG              |0.744 |82.28       |<b>95.65</b>|79.41 / 86.09       |60.99(?)|       |
|PG                |0.746 |82.52       |95.53       |78.92 / 85.42       |        |       |
|Reparameterization|0.804 |<b>82.68</b>|94.95       |<b>84.31 / 88.97</b>|        |       |
|IDPG-trained      |8.090 |81.72       |94.50       |                    |        |       |

### RoBERTa-large (355M)
|Method            |PARAM | MNLI 10% m | SST-2      |MRPC        | RTE    |MNLI   |
|---               |---   |---         |---         |---         |---     |---    |
|Fine tuning       |      |            |            |            |        |       |
|LoRA              |      |            |            |            |        |       |
|Prefix tuning     |      |            |            |            |        |       |
|Adapter H         |      |            |            |            |        |       |
|Adapter P         |      |            |            |            |        |       |
|Prompt tuning     |      |            |93.6        |            |        |       |
|IDPG              |      |            |94.4        |            |        |       |
|PG                |      |            |94.5        |            |        |       |
|Reparameterization|      |            |        |            |        |       |

## Dataset split details
|Dataset        |Train    | Validation    | Test        |
|---            |---      |---            |---          |
|MNLI           | 392,702 | 9,815 / 9,832 |9,796 / 9,847|
|RTE            | 2,490   |     277       |    3,000    |
|MRPC           | 3,668   |     408       |    1,725    |
|SST-2          | 67,349  |     872       |    1,821    |

----

## Architecture

- IDPG : 12620096 / 1694671104 (0.744%)

![image](https://user-images.githubusercontent.com/29649894/147305170-5303fff4-d48b-44bf-9284-1ac1afdaa656.png)

|Name                                       |Param       |
|---                                        |---         |
|output_processor.score.weight              |[3, 1600]   |
|input_processor.encoder_generator.0.weight |[384, 768]  |
|input_processor.encoder_generator.0.bias   |[384]       |
|input_processor.encoder_generator.2.weight |[32000, 384]|
|input_processor.encoder_generator.2.bias   |[32000]     |

- PG : 12635456 / 1694686464 (0.746%)

![image](https://user-images.githubusercontent.com/29649894/147305182-9b46cea8-74e8-42a6-8dd4-4fe1267258d1.png)


|Name                                             |Param       |
|---                                              |---         |
|output_processor.score.weight                    |[3, 1600]   |
|input_processor.encoder_generator.0.weight       |[384, 768]  |
|input_processor.encoder_generator.0.bias         |[384]       |
|input_processor.encoder_generator.2.weight       |[32000, 384]|
|input_processor.encoder_generator.2.bias         |[32000]     |
|input_processor.encoder_prompt_embeddings.weight |[20, 768]   |

- Reparameterization : 12620864 / 1570232064 (0.804%)

![image](https://user-images.githubusercontent.com/29649894/147435274-6f5b4a50-3e03-439e-a37a-dfb6b6b001b9.png)

|Name                                             |Param       |
|---                                              |---         |
|output_processor.score.weight                    |[3, 1600]   |
|input_processor.encoder_generator.0.weight       |[384, 768]  |
|input_processor.encoder_generator.0.bias         |[384]       |
|input_processor.encoder_generator.2.weight       |[32000, 384]|
|input_processor.encoder_generator.2.bias         |[32000]     |
|input_processor.encoder_prompt_embeddings.weight |[1, 768]    |
