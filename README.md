# Interpolated Policy Gradients using PPO

The following project is a reproduction of the [Interpolated Policy Gradient: Merging On-Policy and Off-Policy Gradient Estimation for Deep Reinforcement Learning](https://arxiv.org/abs/1706.00387 "Named link title") of Gu, Shixiang and Lillicrap et al., using Proximal Policy Optimization (PPO) rather than Trust Rregion Policy Optimization (TRPO). The code is built upon the OpenAI Spinning Up PPO code.

### Information:
The project interpolates between on-policy and off-policy data when updating the main policy. In addition to PPO, it includes a Q-function that is updated in the same manner 
as the Deep Deterministic Policy Gradient (DDPG) algorithm.
I had trouble implementing the taylor expansion control variate with pytorch, therefore the control variate implemented is the "Reparameterized Critic Control Variate" from appendix 11.1 of the IPG paper.  
Information about the parameters can be found in the [ppo_ipg.py](https://github.com/pavlosSkev/ipg_ppo/blob/main/ppo_ipg.py) file.  
Original code of IPG by the authors, implemented in the RLLAB framework with Tensorflow can be found [here](https://github.com/shaneshixiang/rllabplusplus).


### Example of experiment:
`python ppo_ipg.py --env=Ant-v3 --epochs=1000 --inter_nu=0.2 --beta=on_policy_sampling`

### Versions:
python = '3.7.4'  
pytorch = '1.7.0+cpu'  
numpy = '1.20.2'  
gym = '0.17.3'  
mujoco = '2.0.2.13'  
tensorboard = '1.15.0'

### Future Work:
- [ ] Include graphs from experimental results
- [ ] Implement PyBullet environments
- [ ] Fix Taylor Expansion control variate
- [ ] Include TD3 methods for possible improvement

