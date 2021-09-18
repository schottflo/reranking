# Reranking parse trees of low-resource languages with Gaussian processes

## Purpose

This repo contains the code for my master thesis project at RycoLab at ETH Zurich. Apart from the code that produces the results in the thesis,
it contains the code to more experiments that we have conducted during the project. For example there is a folder on constituency tree rerankers,
which are not mentioned in our work.

## Installation

Make sure that Python 3.7 is installed. Then run the following commands:

```
git clone https://github.com/schottflo/bayesian-nlp
cd parsing
pip install -e .
```


## Reproduction of the results in the master thesis

In order to reproduce the results from the thesis, place an additional directory "data" inside "dep_tree_models".
You can find the directory at the following link with a password that should have been provided to you:

https://polybox.ethz.ch/index.php/s/opfMLVmB8rX5lgn

Otherwise, please maintain the structure of the directory as it is. Then run

```
python file_generator.py 
python oracle_stats.py
```

to create a pre-processed version of the data and all statistics that are not based on the computations of a model.
The other files are named after the baselines/models they reproduce

* **Baselines:** baselines.py
* **SVMs:** svms.py (Experiment 2, Experiment 3)
* **GPs:** compute_gold_gps.py (Experiment 1), compute_gps.py (Experiment 2), conf_reranking.py (Experiment 3)

Note that conf_reranking.py requires that compute_gps.py was run before.
