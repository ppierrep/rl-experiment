## Installation:

Create and activate custom environment
```
pip install pipenv
pipenv install --python=/usr/bin/python3.6
pipenv shell
```

To install the minimal installation open ai gym with `classical_control` and `box2D`:
```
git clone https://github.com/openai/gym.git
pip install -e gym/
pip install -e 'gym/[box2d]'
pip install -e 'gym/[classic_control]'
```

This project use udacity deep-reinforcement learning repository as dependency:
```
git clone https://github.com/udacity/deep-reinforcement-learning.git
pip install deep-reinforcement-learning/python
```

```
pip install mlagents==0.4.0
```

<!-- python -m ipykernel install --user --name drlnd --display-name "drlnd" -->

1. Download the environment from udacity from the link below.
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

2. Place the file in the `env/` folder, and decompress the file. 


### Instructions

Follow the instructions in `Navigation.ipynb` to get started with training your own agent!

```
jupyter notebook
```