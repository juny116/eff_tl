# Experiment Results

### GPT2-XL(main model)

|Method            |SEED |PARAM(%)|Task       |lr           |Train Epochs|Warmup Step|Accuracy      |
|---               |---  |---     |---        |---          |---         |---        |---           |
|---               |---  |---     |---        |---          |---         |---        |---           |
|Fine tuning       |1234 |100     |MNLI 10% m |1e-5         |30          |1,000      |82.9          |
|LoRA              |1234 |0.47    |MNLI 10% m |8e-5         |40          |1,000      |83.6          |
|LoRA              |1234 |0.47    |MNLI 10% m |5e-5         |40          |1,000      |<br>83.7</br> |
|LoRA              |1234 |0.47    |MNLI 10% m |1e-5         |40          |1,000      |83.1          |
|LoRA              |1234 |0.47    |MNLI 10% m |5e-6         |40          |1,000      |82.2          |
|Prompt tuning     |1234 |0.005   |MNLI 10% m |5e-3         |80          |1,000      |80.0          |
|Prompt tuning     |1234 |0.01    |MNLI 10% m |5e-3         |80          |1,000      |66.6          |
|Prompt tuning     |1234 |0.005   |MNLI 10% m |1e-3         |80          |1,000      |78.1          |
|Prompt tuning     |1234 |0.001   |MNLI 10% m |1e-3         |40          |1,000      |69.5          |
|Prompt tuning     |1234 |0.005   |MNLI 10% m |5e-4         |80          |1,000      |65.18         |
|Prompt tuning     |1234 |0.005   |MNLI 10% m |1e-4         |80          |1,000      |53.9          |
