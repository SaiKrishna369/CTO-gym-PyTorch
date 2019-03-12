# CTO-gym-PyTorch version
OpenAI environment for Cooperative Target Observation (CTO) domain is similar to [CTO-gym](https://github.com/SaiKrishna369/CTO-gym) expect, to explicitly run computations on GPU/CPU through PyTorch tensors.


## Installation

Install the [OpenAI gym](https://gym.openai.com/docs/).

Then install this package via

```
pip install -e .
```

## Usage

```
import gym
import gym_cto_pytorch

env = gym.make('TCTO-v0') or env = gym.make('TCTO-v1')
env.initialize() #compulsory
env.reset() #compulsory
```