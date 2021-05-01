from copy import deepcopy

import numpy as np
import warnings

warnings.filterwarnings("ignore", message=r"Passing", category=FutureWarning)
from torch.optim import Adam
import gym
import time
import core as core
from utils.logx import EpochLogger
from utils.mpi_pytorch import setup_pytorch_for_mpi, sync_params, mpi_avg_grads
from utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs
from on_buffer import PPOBuffer
from off_buffer import DDPGBuffer

import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def _preproc_input(input_convention):
    # convert to tensor
    return torch.as_tensor(input_convention, dtype=torch.float32)


def evaluate(env, policy, max_timesteps, n_traj=1):
    print(f"Evaluating for {n_traj} episode(s), with {max_timesteps} time-steps.")
    rewards = []
    for j in range(n_traj):
        o, d, ep_ret, ep_len = env.reset(), False, 0, 0
        while not (d or (ep_len == max_timesteps)):
            a, _, _, _, _ = policy.step(torch.as_tensor(o, dtype=torch.float32))
            #use mu_net for evaluation??
            o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1
        rewards.append(ep_ret)
    print("Done evaluating")
    return sum(rewards) / len(rewards)

def ppo(env_fn, actor_critic=core.MLPActorCritic, q_func=core.QFunction, ac_kwargs=dict(), seed=0,
        num_episodes=50, steps_per_epoch=4000, epochs=50, batch_sample_size=64, gamma=0.99, clip_ratio=0.2,
        pi_lr=3e-4, vf_lr=1e-3, qf_lr=1e-3, inter_nu=0.2, train_pi_iters=80, train_v_iters=80, train_qf_iters=80,
        tau=0.001, num_cycles=1, lam=0.97, max_ep_len=1000, target_kl=0.01, logger_kwargs=dict(), save_freq=10,
        exp_qf_sample_size=5, use_cv=None, cv_type=None, goal_append=True, beta=None, off_buffer_size = 1e6):
    # Special function to avoid certain slowdowns from PyTorch + MPI combo.
    setup_pytorch_for_mpi()
    # Set up logger and save configuration
    logger = EpochLogger(**logger_kwargs)
    # TODO: check locals()
    logger.save_config(locals())

    # Random seed
    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Instantiate environment
    env = env_fn()

    steps_per_epoch = env._max_episode_steps
    max_ep_len = env._max_episode_steps

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape

    # Create actor-critic module and q function
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Sync params across processes
    sync_params(ac)

    # Count variables
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.v])
    logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n' % var_counts)


    # Set up experience buffers
    local_train_qf_iters = int(train_qf_iters / num_procs())
    print(f"local qf iterations: {local_train_qf_iters}")

    # Separates episode number per process. If 8 episodes per epoch with 4 processes, each process gets 2 simulations
    local_episodes_per_epoch = int(num_episodes / num_procs())
    local_steps_per_epoch = steps_per_epoch

    #initialize buffers
    buf_on = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch * local_episodes_per_epoch, gamma, lam)
    buf_off = DDPGBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(off_buffer_size))


    # Set up function for computing PPO policy loss
    def compute_loss_pi(data, plot=False):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']
        if plot:
            learning_signals = adv * (1 - inter_nu)
        else:
            if use_cv:
                print("Using Control Variate...")
                crit_based_adv = get_control_variate(data)  #returns q(s,t) - E[q(s,t)] for ~1
                learning_signals = (adv - crit_based_adv) * (1 - inter_nu)
            else:
                learning_signals = adv * (1 - inter_nu)  # line 10-12 IPG pseudocode

        # Policy loss
        pi, logp = ac.pi(obs, act)  # pi is a distribution
        ratio = torch.exp(logp - logp_old)  # same thing
        clip_adv = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * learning_signals  # ratio, min, max
        loss_pi = -(torch.min(ratio * learning_signals, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()  # with math, this is equal to dist division, like IS.
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)
        return loss_pi, pi_info

    # Set up function for computing value loss
    def compute_loss_v(data):
        obs, ret = data['obs'], data['ret']
        return ((ac.v(obs) - ret) ** 2).mean()

    def compute_loss_qf(data):
        obs, act, r, obs_next, dones = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        q_value_real = ac.qf(obs, _preproc_input(act))  # correct
        with torch.no_grad():
            # TODO: policy target?
            act_next, _, _, _, _ = ac.step(obs_next)
            q_next_value = ac_targ.qf(obs_next, _preproc_input(act_next))
            q_next_value = q_next_value.detach()  # detach tensor from graph
            # Bellman backup for Q function
            q_value_target = r + gamma * (1 - dones) * q_next_value
        return (q_value_target - q_value_real).pow(2).mean()

    def compute_loss_off_pi(data):
        obs = data['obs']
        mu = ac.pi.mu_net(obs)
        off_loss = ac.qf(obs, mu)
        return -(off_loss).mean()

    def get_expected_q(obs):
        mu, std = ac.pi._get_mean_sigma(obs)
        actions_noise = torch.normal(mean=0, std=1, size=mu.shape) * std + mu  # get_expected_q(mu, std)

        return ac.qf(obs, actions_noise)

    def get_control_variate(data): #11.1
        obs, act = data['obs'], data['act']

        if cv_type == 'reparam_critic_cv':
            # with torch.no_grad(): #makes it worse. It needs to be in the graph
            q_value_real = ac.qf(obs, act)
            q_value_expected = get_expected_q(obs)
            return q_value_real - q_value_expected  #from 3.1 of interpolated policy gradients paper

        #Taylor Expansion control variate is probably implemented wrong. Do not use, only fix.
        #TODO: Fix taylor expansion
        elif cv_type=='taylor_exp_cv':
            #second part of equation 8 in QPROP paper.
            mu0 = ac.pi.mu_net(obs)
            q_value_real = ac.qf(obs, mu0)
            q_prime = torch.autograd.grad(q_value_real.mean(), mu0, retain_graph=True)[0]
            deltas = act - mu0
            return q_value_real - (q_prime * deltas).sum()  #shape (4000,6)
        else:
            raise ValueError(f"Wrong value for parameter: cv_type, value: {cv_type} does not exist")

    # Set up optimizers for policy and value function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    vf_optimizer = Adam(ac.v.parameters(), lr=vf_lr)
    qf_optimizer = Adam(ac.qf.parameters(), lr=qf_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update():
        data_on = buf_on.get()  # with get we also fit baseline and compute value functions after E episodes. Line 4 in pseudocode
        data_off = buf_off.sample_batch(batch_sample_size)
        with torch.no_grad():
            pi_l_old, pi_info_old = compute_loss_pi(data_on, plot=True)
            pi_l_old = pi_l_old.item()
            v_l_old = compute_loss_v(data_on).item()
            qf_l_old = compute_loss_qf(data_off).item()
            pi_l_off_old = compute_loss_off_pi(data_off).item()
            inter_l_old = pi_l_old + inter_nu * pi_l_off_old
        eval_reward = evaluate(env, ac, steps_per_epoch)

        # write with tensorbard
        writer.add_scalar("Loss/train (policy)", pi_l_old, epoch)
        writer.add_scalar("Loss/train (value)", v_l_old, epoch)
        writer.add_scalar("Loss/train (q function)", qf_l_old, epoch)
        writer.add_scalar("Loss/train (off policy)", pi_l_off_old, epoch)
        writer.add_scalar("Loss/train (inter)", inter_l_old, epoch)
        writer.add_scalar("Return/Epoch", np.array(average_return).mean(), epoch)
        writer.add_scalar("Return/Epoch", eval_reward, epoch)

        # change parameters phi of QF
        for i in range(local_train_qf_iters):
            # sample batch
            data_off = buf_off.sample_batch(batch_sample_size)
            qf_optimizer.zero_grad()
            loss_qf = compute_loss_qf(data_off)
            loss_qf.backward()
            mpi_avg_grads(ac.qf)  # average grads across MPI processes
            qf_optimizer.step()
            soft_update()

        # freeze q network parameters.
        for param in ac.qf.parameters():
            param.requires_grad = False

        # update ppo policy
        # Train policy with multiple steps of gradient descent
        #samples same transitions as on-policy (batch is same size as the local collection steps)
        data_off = buf_off.sample_batch(local_train_qf_iters)
        for i in range(train_pi_iters):
            pi_optimizer.zero_grad()
            # update with previous loss only
            loss_pi, pi_info = compute_loss_pi(data_on)
            # choose between off policy and on policy sampling.

            if beta == 'off_policy_sampling':
                loss_off_pi = compute_loss_off_pi(data_off)
            elif beta == 'on_policy_sampling':
                loss_off_pi = compute_loss_off_pi(data_on)
            else:
                assert False, f"'{beta}' is no valid value for beta"
            # b value for multiplication with off policy loss, shown in algorithm 1 of IPG paper.
            if use_cv:
                b = 1
            else:
                b = inter_nu
            loss_pi_inter = loss_pi + b * loss_off_pi
            kl = mpi_avg(pi_info['kl'])
            # When completely off-policy (nu=1), it is better to avoid max kl break, as it deteriorates performance.
            if inter_nu<1.0:
                if kl > 1.5 * target_kl:
                    logger.log('Early stopping at step %d due to reaching max kl.' % i)
                    break
            loss_pi_inter.backward()  # computes gradients
            mpi_avg_grads(ac.pi)  # average grads across MPI processes
            pi_optimizer.step()

        logger.store(StopIter=i)

        # Value function learning
        for i in range(train_v_iters):
            vf_optimizer.zero_grad()
            loss_v = compute_loss_v(data_on)
            loss_v.backward()
            mpi_avg_grads(ac.v)  # average grads across MPI processes
            vf_optimizer.step()

        # unfreeze q net params for next iteration
        for param in ac.qf.parameters():
            param.requires_grad = True

        # Log changes from update
        kl, ent, cf = pi_info['kl'], pi_info_old['ent'], pi_info['cf']
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     LossQf=qf_l_old, LossIPG=inter_l_old,
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(loss_pi.item() - pi_l_old),
                     DeltaLossV=(loss_v.item() - v_l_old))

    def soft_update():
        with torch.no_grad():
            for p, p_targ in zip(ac.qf.parameters(), ac_targ.qf.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(0.999)
                p_targ.data.add_((1 - 0.999) * p.data)

    # Prepare for interaction with environment
    start_time = time.time()
    obs, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        average_return = []
        for _ in range(num_cycles):  # this is the "repeat" in pseudocode line 2 (or epochs, depending if I live this secont loop here)
            for episode in range(local_episodes_per_epoch):
                for t in range(local_steps_per_epoch):
                    a, v, logp, mean, std = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                    next_obs, r, d, _ = env.step(a)

                    ep_ret += r
                    ep_len += 1

                    #store data in buffers
                    buf_on.store(obs, a, r, v, logp)
                    buf_off.store(obs, a, r, next_obs, d)
                    logger.store(VVals=v)

                    obs = next_obs
                    timeout = ep_len == max_ep_len
                    terminal = d or timeout  # d or timeout for robotics, timeout or timeout for locomotion
                    epoch_ended = t == local_steps_per_epoch - 1
                    if terminal or epoch_ended:
                        if epoch_ended and not (terminal):
                            print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)

                        # if trajectory didn't reach terminal state, bootstrap value target
                        if timeout or epoch_ended:
                            _, v, _, _, _ = ac.step(torch.as_tensor(obs, dtype=torch.float32))
                        else:
                            v = 0
                        buf_on.finish_path(v)  # computes GAE after episode is done, we want this after all the gather
                        if terminal:
                            # only save EpRet / EpLen if trajectory finished
                            average_return.append(ep_ret)
                            logger.store(EpRet=ep_ret, EpLen=ep_len)
                        obs, ep_ret, ep_len = env.reset(), 0, 0
        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs - 1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch + 1) * steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('LossQf', average_only=True)
        logger.log_tabular('LossIPG', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time() - start_time)
        logger.dump_tabular()

    writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    # parser.add_argument('--env', type=str, default='Ant-v3')

    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--steps', type=int, default=None)  # set 50 for robotic tasks, 4000 for locomotion
    parser.add_argument('--epochs', type=int, default=250)  # 50 default, 200 works for robotic amnipulation
    parser.add_argument('--num_episodes', type=int,
                        default=4)  # set 50 for robotic tasks, 1 for locomotion, set to 4 if use default max timesteps for locomition
    parser.add_argument('--num_cycles', type=int, default=1)
    parser.add_argument('--use_cv', default=False)
    parser.add_argument('--cv_type', default='reparam_critic_cv', help='determines which cv to use. Possible vals: '
                                                                       '"reparam_critic_cv" and "taylor_exp_cv"')

    parser.add_argument('--beta', default='off_policy_sampling', help='determines sampling for off-policy loss. '
                                                                     'Possible values: "off_policy_sampling", "on_policy_sampling"')
    parser.add_argument('--train_pi_iters', type=int, default=80)  # default 80 for all of them
    parser.add_argument('--train_v_iters', type=int, default=80)
    parser.add_argument('--train_qf_iters', type=int, default=4000)  # same as batch size of on-policy collection
    parser.add_argument('--inter_nu', type=float, default=1.0)
    parser.add_argument('--exp_name', type=str, default='ppo3')

    args = parser.parse_args()

    # to avoid MPI error, steps%num_cpu has to be 0
    # args.steps = args.cpu * args.steps
    mpi_fork(args.cpu)  # run parallel code with mpi

    from utils.run_utils import setup_logger_kwargs

    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    # ------for tensorboard------
    experiment_folder = str(datetime.now()).replace(':', '-') #get current time and date
    path = './tensorboard_results/'
    tensorboard_path = path + f"{experiment_folder}"
    writer = SummaryWriter(log_dir=tensorboard_path)
    # --------------------------

    ppo(lambda: gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid] * args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, train_pi_iters=args.train_pi_iters,
        train_v_iters=args.train_v_iters, train_qf_iters=args.train_qf_iters, epochs=args.epochs, batch_sample_size=64,
        num_cycles=args.num_cycles, num_episodes=args.num_episodes, use_cv=args.use_cv, cv_type=args.cv_type, beta=args.beta,
        inter_nu=args.inter_nu, logger_kwargs=logger_kwargs)
