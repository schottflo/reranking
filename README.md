# Reranking parse trees of low-resource languages with Gaussian processes

## Purpose

This repo contains the code for my master thesis project at RycoLab at ETH Zurich. Apart from the code that produces the results in the thesis,
it contains the code to more experiments that we have conducted during the project. For example there is a folder on constituency tree rerankers,
which are not mentioned in the thesis. The thesis is available [here](https://polybox.ethz.ch/index.php/s/36DyZXWt8nLUg4O).

## Installation

Make sure that Python 3.7 is installed. Moreover, our code relies on the library [spanningtrees](https://github.com/rycolab/spanningtrees). It needs to be installed manually by following the instructions in the repo. When installing spanningtrees, the authors had trouble installing the dependency "arsenal", but it is not necessary for our code. In case you experience difficulties as well, just remove it from the setup.py file from spanningtrees, before running pip install -e.

After spanningtrees is installed, run the following command to install the dependencies:

```
git clone https://github.com/schottflo/reranking
cd reranking
pip install -e .
```

## Reproduction of the results in the master thesis

In order to reproduce the results from the thesis, place an additional directory "data" inside "dep_tree_models".
You can find the directory at the following link with a password that should have been provided to you:

https://polybox.ethz.ch/index.php/s/opfMLVmB8rX5lgn

Otherwise, please maintain the structure of the directory as it is. Then run

```
cd parsing/dep_tree_models
python file_generator.py 
python oracle_stats.py
```

to create a pre-processed version of the data and all statistics that are not based on the computations of a model.
The other files are named after the baselines/models they reproduce

* **Baselines:** baseline_predictions.py (Experiment 2)
* **SVMs:** svms.py (Experiment 2)
* **GPs:** compute_gold_gps.py (Experiment 1), compute_gps.py (Experiment 2), conf_reranking.py (Experiment 3)

Note that conf_reranking.py requires that compute_gps.py was run before. When all the scripts were run successfully, use figures.py to reproduce the figures.

All GP computations were run on Euler, the high performance computing cluster of ETH. Reproducibility can only be ensured when rerunning the scripts there. The SVMs can be reproduced on your local machine.
