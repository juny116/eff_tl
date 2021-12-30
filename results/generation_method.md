# Generation method (OURS)


|Dataset        |Seed |# prompts | lr   | scheduler | warmup step | valid approx. acc. |
|---            |---  |---       |---   |---        |---          | ---                |
|MNLI-40000     |1234 |20        |1e-3  |linear     |1,500        |77.0                |
|MNLI-40000     |1234 |20        |1e-4  |linear     |1,500        |75.0                |
|MNLI-40000     |1234 |20        |1e-5  |linear     |1,500        |53.5                |
|MNLI-40000     |1234 |10        |5e-3  |linear     |1,500        |overflow            |
|MNLI-40000     |1234 |10        |3e-3  |linear     |1,500        |overflow            |
|MNLI-40000     |1234 |10        |1e-3  |linear     |1,500        |79.0                |
|MNLI-40000     |1234 |10        |8e-4  |linear     |1,500        |overflow            |
|MNLI-40000     |1234 |10        |5e-4  |linear     |1,500        |                    |
|MNLI-40000     |1234 |10        |1e-4  |linear     |1,500        |33.3                |
|MNLI-40000     |1234 |10        |5e-5  |linear     |1,500        |33.3                |
|MNLI-40000     |1234 |10        |1e-3  |linear     |5,000        |78 + overflow       |
|MNLI-40000     |1234 |10        |1e-3  |linear     |4,000        |78.0                |
|MNLI-40000     |1234 |10        |1e-3  |linear     |3,000        |                    |
|MNLI-40000     |1234 |10        |      |linear     |             |                    |
|MNLI-40000     |1234 |10        |      |linear     |             |                    |


# Test Results

|Dataset        |Seed |# prompts | lr   | scheduler |Train eposh | warmup step | valid approx. acc. |
|---            |---  |---       |---   |---        |---         |---          | ---                |
|MNLI-40000     |1234 |10        |5e-3  |linear     |30          |4000         |                    |
|MNLI-40000     |1234 |10        |1e-3  |linear     |30          |4000         |78 이후 overflow     |
|MNLI-40000     |1234 |10        |1e-4  |linear     |30          |4000         |77.7                |
|SST-2          |1234 |10        |1e-3  |linear     |20          |2,000        |overflow            |
|SST-2          |1234 |10        |5e-5  |linear     |20          |2,000        |93.81               |
|SST-2          |1234 |10        |5e-4  |linear     |20          |2,000        |93.6 이후 overflow   |
|SST-2          |1234 |10        |5e-5  |linear     |20          |2,000        |                    |
