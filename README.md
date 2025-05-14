---------------------------------

[![GitHub Repo Size](https://img.shields.io/github/repo-size/NicoSlvd/rumboost?logo=github&label=repo+size)](https://github.com/NicoSlvd/rumboost) [![Python Versions](https://img.shields.io/pypi/pyversions/rumboost.svg?logo=python&logoColor=white)](https://pypi.org/project/rumboost) [![PyPI Version](https://img.shields.io/pypi/v/rumboost.svg?logo=pypi&logoColor=white)](https://pypi.org/project/rumboost) [![PyPI Downloads](https://img.shields.io/pypi/dm/rumboost?logo=icloud&logoColor=white)](https://pypistats.org/packages/rumboost) [![Documentation Status](https://readthedocs.org/projects/rumboost/badge/?version=latest)](https://rumboost.readthedocs.io/) [![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](https://github.org/NicoSlvd/rumboost/LICENSE.md) [![arXiv](https://img.shields.io/badge/arXiv-2401.11954-b31b1b.svg)](https://arxiv.org/abs/2401.11954)


## Description

RUMBoost is a python package to estimate Random Utility models with Gradient Boosted Decision Trees. More specifically, each parameter in the traditional utility function is replaced by an ensemble of regression trees with appropriate constraints to: i) ensure the guarantee of marginal utilities monotonicity; ii) incorporate alternative-specific attributes; and iii) provide an intrinsically interpretable non-linear form of the utility function, directly learnt from the data.

Currently RUMBoost can estimate the following RUMs:

- MNL
- Nested Logit
- Cross-Nested Logit
- An equivalent of the Mixed Effect model

For more details, you can refer to our [paper](https://doi.org/10.1016/j.trc.2024.104897).

## Installation

To use the latest version of RUMBoost, we recommend cloning this repository and installing the requirements in a separate environment.

An oldest version of RUMBoost is also available on [pypi](https://pypi.org/project/rumboost/). All the resources (examples, docs and requirements) related to this version are in the [commit associated with the release](https://github.com/big-ucl/rumboost/tree/v1.0.2). You can install RUMBoost from PyPI with the following command:

`pip install rumboost`

We recommend to install rumboost in a separate environment with its dependencies.

## Documentation and examples
The full documentation can be found [here](https://rumboost.readthedocs.io/en/latest/). In addition, you can find several examples on how to use RUMBoost under the [example](https://github.com/NicoSlvd/rumboost/tree/main/examples) folder. Currently, there are seven example notebooks. We recommend using them in this order:

1. [simple_rumboost](https://github.com/NicoSlvd/rumboost/blob/main/examples/1_simple_rumboost.ipynb): how to train and plot parameters of a simple RUMBoost model
2. [feature_interaction](https://github.com/NicoSlvd/rumboost/blob/main/examples/2_feature_interaction.ipynb): how to include feature interactions for training and plotting
3. [shared_ensembles](https://github.com/NicoSlvd/rumboost/blob/main/examples/3_shared_ensembles.ipynb): how to train a RUMBoost model with one or more ensembles shared across alternatives
4. [functional_effect](https://github.com/NicoSlvd/rumboost/blob/main/examples/4_functional_effect.ipynb): how to train and plot a functional effect RUMBoost model
5. [nested](https://github.com/NicoSlvd/rumboost/blob/main/examples/5_nested.ipynb): how to train a nested logit RUMBoost model
6. [cross-nested](https://github.com/NicoSlvd/rumboost/blob/main/examples/6_cross-nested.ipynb): how to train a cross-nested logit RUMBoost model
7. [smoothing_and_vot](https://github.com/NicoSlvd/rumboost/blob/main/examples/7_smoothing_and_vot.ipynb): how to smooth a RUMBoost output and plot the smoothed version, as well as computing and plotting VoT
8. [bootstrap](https://github.com/NicoSlvd/rumboost/blob/main/examples/8_bootstrap.ipynb): how to test the model robustness
9. [GPU_and_batch_training](https://github.com/big-ucl/rumboost/blob/main/examples/9_GPU_and_batch_training.ipynb): how to train the model with batches and how to compute the gradients on the GPU

## Bug reports and feature requests
If you encounter any issues or have ideas for new features, please open an [issue](https://github.com/NicoSlvd/rumboost/issues). You can also contact us at nicolas.salvade.22@ucl.ac.uk

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Citation

If you found this repository useful, you can acknowledge the authors by citing:

* Salvadé, N., & Hillel, T. (2025). Rumboost: Gradient Boosted Random Utility Models. *Transportation
Research Part C: Emerging Technologies* 170, 104897. DOI: [10.1016/j.trc.2024.104897](https://doi.org/10.1016/j.trc.2024.104897)
