### Value and Policy based methods algorithm using pytorch

<p align="center"> 
    <img src="Banana_Collector_Environment.gif">
</p>

## Installation:

Create and install dependencies:
```
pip install pipenv
pipenv install
```

Activate custom environment:
```
pipenv shell
```

In, `rl-vbm` install the minimal installation open ai gym with `classical_control` and `box2D` (to run in project root):
```
git clone https://github.com/openai/gym.git
pip install -e gym/
pip install -e 'gym/[box2d]'
pip install -e 'gym/[classic_control]'
```

This project use udacity deep-reinforcement learning repository as dependency (to run in project root):
```
git clone https://github.com/udacity/deep-reinforcement-learning.git
pip install deep-reinforcement-learning/python
```

1. Download the environment from udacity from the link below.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

2. Place the file in the `env/` folder, and decompress the file:
```
unzip Banana_Linux_NoVis.zip -d env/
```

if `unzip` is not installed:
```
sudo apt-get install unzip
```

### Instructions

To use jupyter notebooks, you may want to enable the environment with:
```
python -m ipykernel install --user --name=rl-vbm
```

```
jupyter notebook
```
