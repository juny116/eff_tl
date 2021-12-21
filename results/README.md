## Dataset split details

|Dataset        |Train    | Validation    | Test        |
|---            |---      |---            |---          |
|MNLI-40000     | 392,702 | 9,815 / 9,832 |9,796 / 9,847|
|RTE            | 2,490   |     277       |    3,000    |
|MRPC           | 3,668   |     408       |    1,725    |
|SST-2          | 67,349  |     872       |    1,821    |

## Samples per label

|Dataset        |Seed |Train                     | Validation      | Test                 |
|---            |---  |---                       |---              |---                   |
|MNLI-40000     |1234 | 11,802 / 14,011 / 13,187 | 338 / 342 / 320 | 3,123 / 3,213 / 3,479|
|RTE            |1234 | 1,241 / 1,249            | 76 / 62         | 69 / 70              |
|MRPC           |1234 | 2,474 / 1,194            | 142 / 62        | 67 / 137             |
|SST-2          |1234 | 29,338 / 37,011          | 558 / 442       | 444 / 428            |

# Expertiment Results

## All Results

### GPT2-XL(encoder) - GPT2-XL(main model)
|Method            |PARAM   |Task       |Learning Rate|Train Epochs|Warmup Step|Accuracy    |
|---               |---     |---        |---          |---         |---        |---         |
|---               |---     |---        |---          |---         |---        |---         |
|Fine-tuning       |100     |MNLI 10% m |             |            |           |            |
|IDPG              |0.744   |MNLI 10% m |             |            |           |            |
|PG                |0.746   |MNLI 10% m |             |            |           |            |
|---               |---     |---        |---          |---         |---        |---         |
|Fine-tuning       |100     |RTE        |             |            |           |            |
|IDPG              |0.744   |RTE        |             |            |           |            |
|PG                |0.746   |RTE        |             |            |           |            |
|---               |---     |---        |---          |---         |---        |---         |
|Fine-tuning       |100     |MRPC       |             |            |           |            |
|IDPG              |0.744   |MRPC       |             |            |           |            |
|PG                |0.746   |MRPC       |             |            |           |            |
|---               |---     |---        |---          |---         |---        |---         |
|Fine-tuning       |100     |SST-2      |             |            |           |            |
|IDPG              |0.744   |SST-2      |             |            |           |            |
|PG                |0.746   |SST-2      |             |            |           |            |
|---               |---     |---        |---          |---         |---        |---         |


### BERT(encoder) - BERT(main model)
|Method            |PARAM   |Task       |Learning Rate|Train Epochs|Warmup Step|Accuracy    |
|---               |---     |---        |---          |---         |---        |---         |
|---               |---     |---        |---          |---         |---        |---         |
|Fine-tuning       |100     |MNLI 10% m |             |            |           |            |
|IDPG              |0.744   |MNLI 10% m |             |            |           |            |
|PG                |0.746   |MNLI 10% m |             |            |           |            |
|---               |---     |---        |---          |---         |---        |---         |
|Fine-tuning       |100     |RTE        |             |            |           |            |
|IDPG              |0.744   |RTE        |             |            |           |            |
|PG                |0.746   |RTE        |             |            |           |            |
|---               |---     |---        |---          |---         |---        |---         |
|Fine-tuning       |100     |MRPC       |             |            |           |            |
|IDPG              |0.744   |MRPC       |             |            |           |            |
|PG                |0.746   |MRPC       |             |            |           |            |
|---               |---     |---        |---          |---         |---        |---         |
|Fine-tuning       |100     |SST-2      |             |            |           |            |
|IDPG              |0.744   |SST-2      |             |            |           |            |
|PG                |0.746   |SST-2      |             |            |           |            |
|---               |---     |---        |---          |---         |---        |---         |

# GLUE Learderboard

![image](https://user-images.githubusercontent.com/29649894/146649318-c57bd7e4-7d01-46d5-8ae9-061b3487069e.png)

# Experiment Results
1. [IDPG(GPT2 - GPT2-xl](IDPG(gpt2 - gpt2-xl).md)
2. [IDPG(GPT2-xl - GPT2-xl)](#IDPG(gpt2-xl - gpt2-xl).md)
3. [IDPG(BERT-BERT)]()
