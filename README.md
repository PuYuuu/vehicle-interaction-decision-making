# Vehicle Interaction Decision Making

The repository implements the decision-making of multiple vehicles at intersections based on level-k game, and uses MCTS to accelerate search. The code is fully implemented in C++ and Python respectively. Which programming language to use is up to you.

![](./img/sample.png)

## How to run🏃‍♂️

Clone the repository to your local path:

```shell
git clone git@github.com:PuYuuu/vehicle-interaction-decision-making.git
```

The following is the development and testing environment of this repository , for your information:

- **System:** Ubuntu 20.04 ( WSL2 )
- **Python 3.8.10:** numpy == 1.24.4  matplotlib == 3.7.4

### 🐍Run using Python

#### 1.1.1 Requirement

Make sure your python version is 3.6.12 or above. Then use the following instructions to install the required third-party libraries, or install them manually.

```shell
pip install -r scripts/requirements.txt
```

#### 1.1.2 Run it

Firstly, you can run it directly using the default parameters.

```shell
python scripts/run.py
```

Or manually specify parameters for example:

```
python scripts/run.py -r 5 --show -l 0
```

For specific parameter descriptions, please use `python scripts/run.py -h` to view.

### 🦏Run using C++

Coming soon ...

#### 1.2.1 Requirement



#### 1.2.2 Run it



### 🛠Configuration file usage

The configuration file of program running parameters is in `${Project}/config` and strictly uses the yaml file format.

## Experimental results📊



## Reference📝

1. *Game Theoretic Modeling of Vehicle Interactions at Unsignalized Intersections and Application to Autonomous Vehicle Control* [[link]](https://ieeexplore.ieee.org/abstract/document/8430842)
2. *Receding Horizon Motion Planning for Automated Lane Change and Merge Using Monte Carlo Tree Search and Level-K Game Theory*  [[link]](https://ieeexplore.ieee.org/document/9147369)
