# Interpolated Policy Gradients using PPO

The following project is a simple reproduction of the [Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning](https://arxiv.org/abs/1706.00387 "Named link title") of Gu, Shixiang and Lillicrap et al., using PPO rather than TRPO. The code is built upon the OpenAI Spinning Up PPO code.

### Main parameters

### Example of running experiment
python ppo_ipg.py --env=Ant-v3 --epochs=1000 --inter_nu=0.2 --beta=on_policy_sampling


### Packages versions:
python = '3.7.4'

pytorch = '1.7.0+cpu'

numpy = '1.20.2'

gym = '0.17.3'

mujoco = '2.0.2.13'

tensorboard = '1.15.0'
