# Experiment Results
1. [IDPG(GPT2 - GPT2-xl)](IDPG_gpt2_gpt2-xl.md)
2. [IDPG(GPT2-xl - GPT2-xl)](IDPG_gpt2-xl_gpt2-xl.md)
3. [IDPG(RoBERTa-large - RoBERTa-large)](IDPG_roberta-large_roberta-large.md)
4. [Benchmark](benchmark.md)
---

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

# GLUE Learderboard

![image](https://user-images.githubusercontent.com/29649894/146649318-c57bd7e4-7d01-46d5-8ae9-061b3487069e.png)

