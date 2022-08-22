# Adaptive Discretization for Reinforcement Learning in Metric Spaces

![discretization plot](discretization_figure.PNG)

This repository contains a reference implementation for the algorithms
appearing in the papers \[2\] for model-free Q learning in continuous spaces and \[3\] for model-based value iteration in continuous spaces.  We also include a fixed discretization implementation of model-free and model-based algorithms for discrete spaces from \[4\] and \[5\] respectively.

## Dependencies
The code has been tested in `Python 3.7.7` and depends on a number of Python
packages. For the core implementation, found under `src/` we include the following files:

* `environment.py`: defines an environment the agent interacts in
* `agent.py`: defines the agent
* `experiment.py`: defines an experiment and saves data

These implementations are adapted from TabulaRL developed by Ian Osband \[1\] extended to continuous state action spaces.  They serve as a test-bed for testing an RL algorithm where new algorithms are tested by implementing several key functions in the `agent` class.

For the remaining scripts which are aimed to reproduce some of the experimental
results found in the paper and can be found in the root directory of this repo,
the following packages are required:

* numpy 1.18.1
* matplotlib 3.1.3
* pandas 1.00.3
* seaborn 0.10.1


## Quick Tour

We offer implementations for four algorithms.  First, an adaptive discretization for model-free Q learning from \[2\] and its corresponding model-free epsilon net algorithm from \[4\].  We also include implementation of AdaMB from \[3\] and an epsilon net UCBVI algorithm \[5\].  All algorithms are implemented with a state space and action space of [0,1] in mind, but for an extension to higher-dimensional space please see the `multi_dimension` subfolder.

The following files implement the different algorithms:
* `adaptive_Agent.py`: implements `adaQL` \[1\]
* `adaptive_model_Agent.py`: implements `adaMB` \[2\]
* `eNet_Agent.py`: implements the discrete model free algorithm on the epsilon net \[4\]
* `data_Agent.py`: implements the heuristic algorithms discussed for the ambulance problem in \[1\]
* `eNet_model_Agent`: implements the discrete model based algorithm on an epsilon net \[5\]

These agents are imported and used in the different tests.  To run the experiments used in the papers the following two files can be used.
* `run_oil_experiments_save_data.py`
* `run_ambulance_experiments_save_data.py`

Each file has parameters at the top which can be changed in order to replicate the parameters considered for each experiment in the paper.  We also include a how-to jupyter notebook walking through the code and an example in `walkthrough.ipynb`.  Note that these use parallel processing and multiple CPU cores in order to speed up the run-time.

## Creating the Figures

The previous `run_....py` files are used to create a `.csv` files of the performance of each algorithm.  In order to create the plots used in the figures, see the jupyter notebooks `test_plot.ipynb` and `simulate_q_values_oil.ipynb`.  Due to storage limitations the final data files are omitted from this repo.


## Citing

If you use `adaMB` or `adaQL` in your work, please cite the accompanying [paper] for `adaMB`:

```bibtex
@misc{sinclair2020adaptive,
      title={Adaptive Discretization for Model-Based Reinforcement Learning}, 
      author={Sean R. Sinclair and Tianyu Wang and Gauri Jain and Siddhartha Banerjee and Christina Lee Yu},
      year={2020},
      eprint={2007.00717},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
and for `adaMB`:
```bibtex
@article{Sinclair_2019,
   title={Adaptive Discretization for Episodic Reinforcement Learning in Metric Spaces},
   volume={3},
   ISSN={2476-1249},
   url={http://dx.doi.org/10.1145/3366703},
   DOI={10.1145/3366703},
   number={3},
   journal={Proceedings of the ACM on Measurement and Analysis of Computing Systems},
   publisher={Association for Computing Machinery (ACM)},
   author={Sinclair, Sean R. and Banerjee, Siddhartha and Yu, Christina Lee},
   year={2019},
   month={Dec},
   pages={1–44}
}

}
```

## Bibliography

\[1\]: Ian Osband, TabulaRL (2017), Github Repository. https://github.com/iosband/TabulaRL

\[2\]: Sean R. Sinclair, Siddhartha Banerjee, Christina Lee Yu. *Adaptive Discretization for Episodic Reinforcement Learning in Metric Spaces.* Available
[here](https://arxiv.org/abs/1910.08151).

\[3\]: Sean R. Sinclair, Tianyu Wang, Gauri Jain, Siddhartha Banerjee, Christina Lee Yu. *Adaptive Discretization for Model Based Reinforcement Learning.* Available
[here](https://arxiv.org/abs/2007.00717).

\[4\]: Zhao Song, Wen Sun. *Efficient Model-free Reinforcement Learning in Metric Spaces.* Available [here](https://arxiv.org/abs/1905.00475).

\[5\]: Mohammad Gheshlaghi Azar, Ian Osband, and Remi Munos. *Minimax Regret Bounds for Reinforcement Learning.* Available [here](https://arxiv.org/abs/1703.05449).

## Contributing

If you'd like to contribute, or have any suggestions for these guidelines, you can contact us at `srs429 at cornell dot edu` or open an issue on this GitHub repository.

All contributions welcome! All content in this repository is licensed under the MIT license.