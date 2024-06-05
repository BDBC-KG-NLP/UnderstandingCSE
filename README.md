## Environment

Run the following command to create the required conda environment:

```shell
conda env create -f environment.yml -n your_new_environment_name
```

## Data

+ Train files:

  Run the following command in the `data/` directory:
  
  ```shell
  bash download.sh
  ```

+ Evaluation files:
  
  Run the following command in the `SentEval/data/downstream/` directory:
  
  ```shell
  bash download_dataset.sh
  ```

## Experiments

+ For experiments in Section 3.1 Theoretical Analysys, refer to `simulation.py`;

+ For experiments in Section 3.2 Empirical Study, refer to scripts in `scripts/empirical`, and to obtain statistcs, refer to `statistics.py`;

+ For experiments in Section 4 Modification to Ineffective Losses, refer to scripts in `scripts/application`;

## Evaluation

For evaluation, refer to `evaluation.py`

## Implementation

For implementations of optimization objectives, refer to line `234-434` in `isotropy/models.py`