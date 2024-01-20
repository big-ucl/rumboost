<img src="logo/rumboost_logo.png" width="300">

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![PyPI version](https://badge.fury.io/py/mylibrary.svg)](https://badge.fury.io/py/rumboost)


## Description

RUMBoost is a python package to estimate Random Utility models with Gradient Boosted Decision Trees. More specifically, each parameter in the traditional utility function is replaced by an ensemble of regression trees with appropriate constraints to: i) ensure the guarantee of marginal utilities monotonicity; ii) incorporate alternative-specific attributes; and iii) provide an intrinsically interpretable non-linear form of the utility function, directly learnt from the data.

Currently RUMBoost can estimate the following RUMs:

- MNL
- Nested Logit
- Cross-Nested Logit
- An equivalent of the Mixed Effect model

For more details, you can refer to the [preprint](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4701222) of our paper.

## Installation

RUMBoost is launched on pypi. You can install it with the following command:

`pip install rumboost`

We recommend to install rumboost in a separate environment with its dependencies.

## Documentation and example
The full documentation can be found [here](https://rumboost.readthedocs.io/en/latest/). In addition, you can find a full tutorial on how to use RUMBoost under the example folder.

## Bug reports and feature requests
If you encounter any issues or have ideas for new features, please open an [issue](https://github.com/NicoSlvd/rumboost/issues). You can also contact us at nicolas.salvade.22@ucl.ac.uk

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Reference paper (preprint)

Salvadé, Nicolas and Hillel, Tim, Rumboost: Gradient Boosted Random Utility Models. Available at SSRN: https://ssrn.com/abstract=4701222 or http://dx.doi.org/10.2139/ssrn.4701222