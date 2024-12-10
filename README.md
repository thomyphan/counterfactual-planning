# Counterfactual Planning

This software is based on the open-loop MCTS implementation from [1] as well as the original POMCP implementation from [2].

Featured algorithm: Counterfactual Open-loop Reasoning with Ad hoc Learning (CORAL) [3]

## Usage
The code requires the external libraries [`BOOST 1.81.0`](https://www.boost.org/) and [`CMake`](https://cmake.org) for building the code. 
    
After installing both libraries go to the `code` folder of this repository and run the following commands: 
```shell script
cmake -DCMAKE_BUILD_TYPE=RELEASE .
make
```

Run the code with:
```
./main <problem> <algorithm> <True/False> <Eta>
```

`problem` specifies the evaluation environment as used in the paper:
- `tag`
- `rocksample-11`
- `rocksample-15`
- `pocman`

`algorithm` specifies the planning algorithm as used in the paper:
- `MCTS` (closed-loop MCTS using Thompson Sampling)
- `POMCPOW` (closed-loop MCTS using Thompson Sampling and Progressive Widening)
- `POOLTS` (open-loop MCTS using Thompson Sampling)
- `CORAL` (open-loop MCTS using MABUCs - **our contribution**)
- `POSTS` (stack-based planning using Thompson Sampling)

`True/False` indicates, wheter human knowledge, i.e., preferred action sets are used for tree traversal. For `CORAL` this must be set to `True` to learn and query intent actions.

`Eta` specifies the training ratio and can be any numeric value between 0 and 1.

The command will output two CSV files containing **(1)** the average performance per simulation budget and **(2)** the number of nodes generated per trial.

## Important Modules

The algorithmic contributions of the paper can be found in `mabuc.h/cpp` (MABUC) and `causal_planner.h/cpp` (CORAL):

## Analysis

The output file `<problem>_<algorithm>_<True/False>_<Eta>.csv` is a CSV file containing the average discounted return and standard error per simulation budget which can be analyzed and plotted with any tool, e.g., Excel.

The node counts are outputted in a separate CSV file `<problem>_<algorithm>_<True/False>_<Eta>_nodeCount.csv`.

## References

- [1] T. Phan et al., *"Memory Bounded Open-Loop Planning in Large POMDPs Using Thompson Sampling"*, in AAAI 2019.
- [2] D. Silver and J. Veness, *"Monte-Carlo Planning in Large POMDPs"*, in NIPS 2010.
- [3] T. Phan et al., *"Counterfactual Online Learning for Open-Loop Monte-Carlo Planning"*, in AAAI 2025