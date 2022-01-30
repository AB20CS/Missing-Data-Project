This repository contains the code to reproduce the experiments of the paper:

__NeuMiss networks: differential programming for supervised learning with missing values.__

https://arxiv.org/abs/2007.01627

The file **NeuMiss.yml** indicates the packages required as well as the
versions used in our experiments.

The methods used are implemented in the following files:
 * **neumannS0_mlp**: the NeuMiss network.
 * **mlp**: the feedforward neural network.
 * **estimators**: the other methods used.

 The files **ground_truth** and **amputation** contain the code for data
 simulation and the code for the Bayes predictors.

 To reproduce the experiments, use:
  * `python launch_simu_perf.py MCAR`
  * `python launch_simu_perf.py MAR_logistic`
  * `python launch_simu_perf.py gaussian_sm`
  * `python launch_simu_perf.py probit_sm`
  * `python launch_simu_depth_effect.py`
  * `python launch_simu_archi.py`

These scripts save their results as csv files in the **results** foder. The
plots can be obtained from these **csv** files by running the **plots_xxx**
files.

## Datasets
All datasets are found in the `datasets` folder.
- `diabetes.csv`: https://www.kaggle.com/mathchi/diabetes-data-set/version/1