# Experiment Results

### RoBERTa-large(encoder) - RoBERTa-large(main model)

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
