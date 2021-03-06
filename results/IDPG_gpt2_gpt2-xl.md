# Experiment Results

### GPT2(encoder) - GPT2-XL(main model)

|Method            |SEED |PARAM(%)|Task       |lr           |Train Epochs|Warmup Step|Accuracy / F1 |
|---               |---  |---     |---        |---          |---         |---        |---           |
|---               |---  |---     |---        |---          |---         |---        |---           |
|IDPG              |1234 |0.744   |MNLI 10% m |1e-4         |40          |1,000      |58.39         |
|IDPG              |1234 |0.744   |MNLI 10% m |8e-5         |40          |1,000      |81.53         |
|IDPG              |1234 |0.744   |MNLI 10% m |5e-5         |40          |1,000      |<b>82.28</b>  |
|IDPG              |1234 |0.744   |MNLI 10% m |3e-5         |40          |1,000      |81.87         |
|IDPG              |1234 |0.744   |MNLI 10% m |1e-5         |40          |1,000      |76.09         |
|IDPG (trained)    |1234 |8.09    |MNLI 10% m |1e-3         |40          |1,000      |overflow      |
|IDPG (trained)    |1234 |8.09    |MNLI 10% m |1e-4         |40          |1,000      |74.19         |
|IDPG (trained)    |1234 |8.09    |MNLI 10% m |8e-5         |40          |1,000      |50.23         |
|IDPG (trained)    |1234 |8.09    |MNLI 10% m |5e-5         |40          |1,000      |<b>81.72</b>  |
|IDPG (trained)    |1234 |8.09    |MNLI 10% m |3e-5         |40          |1,000      |81.68         |
|IDPG (trained)    |1234 |8.09    |MNLI 10% m |1e-5         |40          |1,000      |78.57         |
|IDPG (trained)    |1234 |8.09    |MNLI 10% m |1e-6         |40          |1,000      |36.48         |
|PG                |1234 |0.746   |MNLI 10% m |3e-4         |40          |1,000      |60.18 (더 학습 가능)|
|PG                |1234 |0.746   |MNLI 10% m |1e-4         |40          |1,000      |<b>82.52</b>  |
|PG                |1234 |0.746   |MNLI 10% m |8e-5         |40          |1,000      |56.44         |
|PG                |1234 |0.746   |MNLI 10% m |5e-5         |40          |1,000      |오류           |
|PG                |1234 |0.746   |MNLI 10% m |1e-5         |40          |1,000      |76.26         |
|Reparameterization|1234 |0.804   |MNLI 10% m |1e-4         |40          |1,000      |<b>82.68</b>  |
|Reparameterization|1234 |0.804   |MNLI 10% m |5e-5         |40          |1,000      |81.38         |
|Reparameterization|1234 |0.804   |MNLI 10% m |1e-5         |40          |1,000      |35.13         |
|---               |---  |---     |---        |---          |---         |---        |---           |
|IDPG              |1234 |0.744   |RTE        |5e-2         |120         |1,000      |54.61         |
|IDPG              |1234 |0.744   |RTE        |1e-2         |120         |1,000      |<b>60.99</b>  |
|IDPG              |1234 |0.744   |RTE        |5e-3         |120         |1,000      |58.87         |
|IDPG              |1234 |0.744   |RTE        |1e-3         |120         |1,000      |56.74         |
|IDPG              |1234 |0.744   |RTE        |5e-4         |120         |1,000      |53.19         |
|IDPG              |1234 |0.744   |RTE        |1e-4         |120         |1,000      |오류           |
|IDPG              |1234 |0.744   |RTE        |5e-5         |120         |1,000      |53.90         |
|IDPG              |1234 |0.744   |RTE        |1e-5         |120         |1,000      |오류           |
|PG                |1234 |0.744   |RTE        |5e-2         |120         |1,000      |54.61         |
|PG                |1234 |0.744   |RTE        |1e-2         |120         |1,000      |49.65         |
|PG                |1234 |0.746   |RTE        |5e-3         |120         |1,000      |56.03         |
|PG                |1234 |0.746   |RTE        |1e-3         |120         |1,000      |-             |
|PG                |1234 |0.746   |RTE        |5e-4         |120         |1,000      |-             |
|Reparameterization|1234 |0.804   |RTE        |             |            |1,000      |              |
|Reparameterization|1234 |0.804   |RTE        |             |            |1,000      |              |
|Reparameterization|1234 |0.804   |RTE        |             |            |1,000      |              |
|---               |---  |---     |---        |---          |---         |---        |---           |
|Fine-tuning       |1234 |100     |RTE        |5e-4         |20          |100        |51.77         |
|Fine-tuning       |1234 |100     |RTE        |1e-4         |20          |100        |58.87         |
|Fine-tuning       |1234 |100     |RTE        |5e-5         |20          |100        |<b>76.6</b>   |
|Fine-tuning       |1234 |100     |RTE        |1e-5         |20          |100        |75.9          |
|IDPG              |1234 |0.744   |RTE        |5e-3         |120         |100        |55.32         |
|IDPG              |1234 |0.744   |RTE        |1e-3         |120         |100        |55.32         |
|IDPG              |1234 |0.744   |RTE        |5e-4         |120         |100        |52.48         |
|PG                |1234 |0.746   |RTE        |5e-3         |120         |100        |-             |
|PG                |1234 |0.746   |RTE        |1e-3         |120         |100        |-             |
|PG                |1234 |0.746   |RTE        |5e-4         |120         |100        |-             |
|Reparameterization|1234 |0.804   |RTE        |             |            |100        |              |
|Reparameterization|1234 |0.804   |RTE        |             |            |100        |              |
|Reparameterization|1234 |0.804   |RTE        |             |            |100        |              |
|---               |---  |---     |---        |---          |---         |---        |---           |
|IDPG              |1234 |0.744   |MRPC       |5e-3         |150         |100        |68.14 / 78.96 |
|IDPG              |1234 |0.744   |MRPC       |3e-3         |150         |100        |74.51 / 83.33 |
|IDPG              |1234 |0.744   |MRPC       |1e-3         |150         |100        |75.00 / 83.06 |
|IDPG              |1234 |0.744   |MRPC       |8e-4         |150         |100        |71.08 / 80.91 |
|IDPG              |1234 |0.744   |MRPC       |5e-4         |150         |100        |72.06 / 81.31 |
|IDPG              |1234 |0.744   |MRPC       |1e-4         |150         |100        |77.45 / 84.46 |
|IDPG              |1234 |0.744   |MRPC       |8e-5         |150         |100        |74.02 / 81.91 |
|IDPG              |1234 |0.744   |MRPC       |5e-5         |150         |100        |<b>79.41 / 86.09</b>  |
|IDPG              |1234 |0.744   |MRPC       |3e-5         |150         |100        |73.04 / 83.08 |
|IDPG              |1234 |0.744   |MRPC       |1e-5         |150         |100        |68.14 / 80.00 |
|IDPG (trained)    |1234 |8.09    |MRPC       |1e-4         |150         |100        |              |
|IDPG (trained)    |1234 |8.09    |MRPC       |5e-5         |150         |100        |              |
|IDPG (trained)    |1234 |8.09    |MRPC       |1e-5         |150         |100        |              |
|PG                |1234 |0.746   |MRPC       |5e-3         |150         |100        |69.61 / 80.98 |
|PG                |1234 |0.744   |MRPC       |3e-3         |250         |100        |76.47 / 83.78 |
|PG                |1234 |0.746   |MRPC       |1e-3         |150         |100        |<b>78.92 / 85.42 (더 학습 가능)</b>|
|PG                |1234 |0.744   |MRPC       |8e-4         |250         |100        |67.16 / 79.38 |
|PG                |1234 |0.746   |MRPC       |5e-4         |150         |100        |69.12 / 81.08 |
|PG                |1234 |0.744   |MRPC       |1e-4         |150         |100        |69.12 / 80.85 |
|PG                |1234 |0.744   |MRPC       |5e-5         |150         |100        |71.08 / 81.62 |
|PG                |1234 |0.744   |MRPC       |1e-5         |150         |100        |71.08 / 81.96 |
|Reparameterization|1234 |0.804   |MRPC       |1e-4         |150         |100        |<b>84.31 / 88.97</b>  |
|Reparameterization|1234 |0.804   |MRPC       |5e-5         |150         |100        |74.02 / 82.85 |
|Reparameterization|1234 |0.804   |MRPC       |1e-5         |150         |100        |69.61 / 80.63 |
|---               |---  |---     |---        |---          |---         |---        |---           |
|IDPG              |1234 |0.744   |SST-2      |1e-2         |30          |1,000      |94.50         |
|IDPG              |1234 |0.744   |SST-2      |5e-3         |30          |1,000      |95.07         |
|IDPG              |1234 |0.744   |SST-2      |3e-3         |30          |1,000      |94.62         |
|IDPG              |1234 |0.744   |SST-2      |1e-3         |30          |1,000      |<b>95.65</b>  |
|IDPG              |1234 |0.744   |SST-2      |8e-4         |30          |1,000      |94.27         |
|IDPG              |1234 |0.744   |SST-2      |5e-4         |30          |1,000      |85.34         |
|IDPG              |1234 |0.744   |SST-2      |1e-4         |30          |1,000      |95.3          |
|IDPG              |1234 |0.744   |SST-2      |5e-5         |30          |1,000      |95.19         |
|IDPG              |1234 |0.744   |SST-2      |1e-5         |30          |1,000      |94.73         |
|IDPG (trained)    |1234 |8.09    |SST-2      |1e-2         |30          |1,000      |94.15         |
|IDPG (trained)    |1234 |8.09    |SST-2      |1e-3         |30          |1,000      |<b>94.50</b>  |
|IDPG (trained)    |1234 |8.09    |SST-2      |5e-4         |30          |1,000      |오류           |
|IDPG (trained)    |1234 |8.09    |SST-2      |1e-4         |30          |1,000      |93.35         |
|IDPG (trained)    |1234 |8.09    |SST-2      |8e-5         |30          |1,000      |92.09         |
|IDPG (trained)    |1234 |8.09    |SST-2      |5e-5         |30          |1,000      |89.33         |
|IDPG (trained)    |1234 |8.09    |SST-2      |3e-5         |30          |1,000      |92.89         |
|IDPG (trained)    |1234 |8.09    |SST-2      |1e-5         |30          |1,000      |91.06         |
|IDPG (trained)    |1234 |8.09    |SST-2      |1e-6         |30          |1,000      |오류           |
|PG                |1234 |0.744   |SST-2      |5e-3         |30          |1,000      |오류           |
|PG                |1234 |0.744   |SST-2      |1e-3         |30          |1,000      |94.04         |
|PG                |1234 |0.744   |SST-2      |5e-4         |30          |1,000      |95.07         |
|PG                |1234 |0.746   |SST-2      |1e-4         |30          |1,000      |94.74         |
|PG                |1234 |0.746   |SST-2      |5e-5         |30          |1,000      |95.07         |
|PG                |1234 |0.746   |SST-2      |3e-5         |30          |1,000      |94.85         |
|PG                |1234 |0.746   |SST-2      |1e-5         |30          |1,000      |<b>95.53</b>  |
|PG                |1234 |0.746   |SST-2      |8e-6         |30          |1,000      |93.93         |
|Reparameterization|1234 |0.804   |SST-2      |1e-4         |30          |1,000      |<b>94.95</b>  |
|Reparameterization|1234 |0.804   |SST-2      |5e-5         |30          |1,000      |94.61         |
|Reparameterization|1234 |0.804   |SST-2      |1e-5         |30          |1,000      |93.58         |

