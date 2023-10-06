import argparse
import os
from pathlib import Path
import numpy as np
import tensorflow as tf
import pickle
from tqdm import tqdm as std_tqdm
from functools import partial
from utils import PPO2Agent, RandomAgent
tqdm = partial(std_tqdm, dynamic_ncols=True, disable=eval(os.environ.get("DISABLE_TQDM", 'False')))

import gym

import matplotlib
matplotlib.use('agg')
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
        'figure.figsize': (5, 4),
        'axes.labelsize': 'xx-large',
        'axes.titlesize':'xx-large',
        'xtick.labelsize':'xx-large',
        'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)
from matplotlib import pyplot as plt

from bc_mujoco import Policy
from utils import RandomAgent, gen_traj, gen_traj_with_gaussian_noise


class ActionNoise(object):
    def reset(self):
        pass

class NormalActionNoise(ActionNoise):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


############## New for iD-Rex ##################    
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise(ActionNoise):
    def __init__(self, mu, sigma, theta=.15, dt=0.008, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
##################################################

############## New for iD-Rex ##################    
class TimeDelayedActionNoise(NormalActionNoise):
    def __init__(self, mu, sigma, p):
        super().__init__(mu, sigma)
        self.p = p
        self.step = 0

    def __call__(self):
        if self.step / 100 < (1 - self.p):
            noise =  self.mu
        else:
            noise =  (self.step / 100 - (1 - self.p)) * np.random.normal(self.mu, self.sigma)

        self.step += 1
        return noise
    
    def reset(self):
        self.step = 0
##################################################


############## New for iD-Rex ##################    
class DegradingActionNoise(OrnsteinUhlenbeckActionNoise):
    def __init__(self, mu, sigma, theta=0.15, dt=1, x0=None):
        super().__init__(mu, sigma, theta, dt, x0)
        self.step = 0
        
    def __call__(self):
        noise = super().__call__() * (1 - (self.step / 1000))
        self.step += 1
        return noise
    
    def reset(self):
        self.step = 0
        super().reset()
##################################################

class NoiseInjectedPolicy(object):
    def __init__(self,env,policy,action_noise_type,noise_level):
        self.action_space = env.action_space
        self.policy = policy
        self.action_noise_type = action_noise_type

        if action_noise_type == 'normal':
            mu, std = np.zeros(self.action_space.shape), noise_level*np.ones(self.action_space.shape)
            self.action_noise = NormalActionNoise(mu=mu,sigma=std)
        elif action_noise_type == 'ou':
            mu, std = np.zeros(self.action_space.shape), noise_level*np.ones(self.action_space.shape)
            self.action_noise = OrnsteinUhlenbeckActionNoise(mu=mu,sigma=std)
        elif action_noise_type == 'epsilon':
            self.epsilon = noise_level
            self.scale = self.action_space.high[0]
            assert np.all(self.scale == self.action_space.high) and \
                np.all(self.scale == -1.*self.action_space.low)
        elif action_noise_type == "td":
            mu, std = np.zeros(self.action_space.shape), np.ones(self.action_space.shape)
            self.action_noise = TimeDelayedActionNoise(mu=mu, sigma=std, p=noise_level)
        elif action_noise_type == "da":
            mu, std = np.zeros(self.action_space.shape), noise_level*np.ones(self.action_space.shape)
            self.action_noise = DegradingActionNoise(mu=mu, sigma=std)
        else:
            assert False, "no such action noise type: %s"%(action_noise_type)

    def act(self, obs, reward, done):
        if self.action_noise_type == 'epsilon':
            if np.random.random() < self.epsilon:
                return np.random.uniform(-self.scale,self.scale,self.action_space.shape)
            else:
                act = self.policy.act(obs,reward,done)
        else:
            act = self.policy.act(obs,reward,done)
            act += self.action_noise()

        return np.clip(act,self.action_space.low,self.action_space.high)

    def reset(self):
        try:
            self.action_noise.reset()
        except AttributeError:
            pass

############## New for iD-Rex ##################
class ProbablisticInterpolatedPolicy(object):
    """
    A policy that "interpolates" between two other
    policies by probabilistically selecting actions from one
    of the two.
    """
    def __init__(self, env, policy_k, policy_km1, alpha):
        self.action_space = env.action_space
        self.policy_k = policy_k
        self.policy_km1 = policy_km1
        self.alpha = alpha # Probability of selecting actions from policy iterate k vs (k-1)

    def act(self, obs, reward, done):
        if np.random.random() < self.alpha:
            act = self.policy_k.act(obs, reward, done) 
        else:
            act = self.policy_km1.act(obs,reward,done)
        return np.clip(act,self.action_space.low,self.action_space.high)

#################################################


class PreferenceDataset(object):
    def __init__(self,env,max_steps=None,min_margin=None):
        self.env = env

        self.max_steps = max_steps
        self.min_margin = min_margin

        self.trajs = []

    def load_prebuilt(self,fname):
        if os.path.exists(fname):
            with open(fname,'rb') as f:
                self.trajs = pickle.load(f)
            return True
        else:
            return False

    def load_multiple_prebuilt(self, dir, fnames):
        for fname in fnames:
            combined_fname = dir/fname
            if os.path.exists(combined_fname):
                with open(combined_fname,'rb') as f:
                    self.trajs.extend(pickle.load(f))
                print(f"Loaded {combined_fname}")
            
            else:
                print(f"Error: {combined_fname} does not exist")
                return False

        return True

    def draw_fig(self,log_dir,demo_trajs):
        demo_returns = [np.sum(rewards) for _,_,rewards in demo_trajs]
        demo_ave, demo_std = np.mean(demo_returns), np.std(demo_returns)

        noise_levels = [noise for noise,_ in self.trajs]
        returns = np.array([[np.sum(rewards) for _,_,rewards in agent_trajs] for _,agent_trajs in self.trajs])

        random_agent = RandomAgent(self.env.action_space)
        random_returns = [np.sum(gen_traj(self.env,random_agent,-1)[2]) for _ in range(20)]
        random_ave, random_std = np.mean(random_returns), np.std(random_returns)

        from_to = [np.min(noise_levels), np.max(noise_levels)]

        plt.figure()
        plt.fill_between(from_to,
                         [demo_ave - demo_std, demo_ave - demo_std], [demo_ave + demo_std, demo_ave + demo_std], alpha = 0.3)
        plt.plot(from_to,[demo_ave, demo_ave], label='demos')

        plt.fill_between(noise_levels,
                         np.mean(returns, axis=1)-np.std(returns, axis=1), np.mean(returns, axis=1) + np.std(returns, axis=1), alpha = 0.3)
        plt.plot(noise_levels, np.mean(returns, axis = 1),'-.', label="bc")

        #plot the average of pure noise in dashed line for baseline
        plt.fill_between(from_to,
                         [random_ave - random_std, random_ave - random_std], [random_ave + random_std, random_ave + random_std], alpha = 0.3)
        plt.plot(from_to,[random_ave, random_ave], '--', label='random')

        plt.legend(loc="best")
        plt.xlabel("Epsilon")
        #plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir,"degredation_plot.pdf"))
        plt.close()


################################
        
class PPOTimestepPreferenceDataset(PreferenceDataset):
    """
    Ranks PPO-generated policies on the basis of the timestep in the training process
    from which they originate. (Not used in the final iD-REX.)
    """
    def prebuild(self, timestep_range, num_trajs, min_length, logdir, curr_iter):
        trajs = []
        for timestep in timestep_range: # Preferably 1 + 20, 40,..., 360
            str_form = '0000' + str(timestep) if timestep < 10 else ('000' + str(timestep) if timestep < 100 else '00' + str(timestep))
            for run in range(0, 1):
                rl_model_path = Path(logdir)/'rl'/f'run_{run}'/'checkpoints'/f"{str_form}"
                agent = PPO2Agent(self.env, 'mujoco', str(rl_model_path), stochastic=True)

                agent_trajs = []

                assert (num_trajs > 0 and min_length <= 0) or (min_length > 0 and num_trajs <= 0)
                while (min_length > 0 and np.sum([len(obs) for obs,_,_,_ in agent_trajs])  < min_length) or\
                        (num_trajs > 0 and len(agent_trajs) < num_trajs):
                    obs,actions,rewards = gen_traj(self.env,agent,-1)
                    agent_trajs.append((obs,actions,rewards))
                    # print(f"Generated trajectories for alpha {alpha}: GT reward {sum(rewards)}")

                # Format change: (alpha, trajectories) --> (alpha, trajectories, iteration)
                trajs.append((timestep, agent_trajs, curr_iter))

        self.trajs = trajs

        with open(os.path.join(logdir,f'prebuilt_ppo.pkl'),'wb') as f:
            pickle.dump(self.trajs,f)

    def draw_fig(self, log_dir):
        
        # Get returns for PPO trajectories
        self.load_prebuilt(os.path.join(log_dir, f'prebuilt_ppo.pkl'))
        timesteps = [t for t,_,_ in self.trajs]
        returns = np.array([[np.sum(rewards) for _,_,rewards in agent_trajs] for _,agent_trajs,_ in self.trajs])
        print("PPO timestep returns: ", np.mean(returns, axis = 1))

        plt.figure()
        plt.fill_between(timesteps,
                         np.mean(returns, axis=1)-np.std(returns, axis=1), np.mean(returns, axis=1) + np.std(returns, axis=1), alpha = 0.3)
        plt.plot(timesteps, np.mean(returns, axis = 1),'-.', label="interp")

        plt.legend(loc="best")
        plt.xlabel("Timesteps")
        #plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir,f"degredation_plot_ppo.pdf"))
        plt.close()



    def sample(self,num_samples,include_action=False):
        D = []

        for _ in tqdm(range(num_samples)):
            # Pick Two Noise Level Set
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)
            
            # Should be at least (40? 60?) timesteps apart
            while abs(self.trajs[x_idx][0] - self.trajs[y_idx][0] + self.trajs[y_idx][2]) < self.min_margin:
                x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            # Pick trajectory from each set
            x_traj = self.trajs[x_idx][1][np.random.choice(len(self.trajs[x_idx][1]))]
            y_traj = self.trajs[y_idx][1][np.random.choice(len(self.trajs[y_idx][1]))]

            # Subsampling from a trajectory
            if len(x_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(x_traj[0])-self.max_steps)
                x_slice = slice(ptr,ptr+self.max_steps)
            else:
                x_slice = slice(len(x_traj[1]))

            if len(y_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(y_traj[0])-self.max_steps)
                y_slice = slice(ptr,ptr+self.max_steps)
            else:
                y_slice = slice(len(y_traj[0]))

            # Done!
            if include_action:
                D.append((np.concatenate((x_traj[0][x_slice],x_traj[1][x_slice]),axis=1),
                          np.concatenate((y_traj[0][y_slice],y_traj[1][y_slice]),axis=1),
                          0 if self.trajs[x_idx][0] > self.trajs[y_idx][0] else 1) # if timestep is bigger, better traj.
                        )
            else:
                D.append((x_traj[0][x_slice],
                          y_traj[0][y_slice],
                          0 if self.trajs[x_idx][0] > self.trajs[y_idx][0] else 1)
                        )

        return D

################################
    
class ProbInterpPreferenceDataset(PreferenceDataset):
    """
    Create a dataset of trajectories 'interpolating' between two policies by selecting
    actions from them according to a probability 'alpha'. Rank trajectories on the basis
    of alpha; higher alpha trajectories should have higher returns and be ranked higher.
    """
    def prebuild(self, agent_k, agent_km1, alpha_range, num_trajs, min_length, logdir, curr_iter):
        trajs = []
        for alpha in tqdm(alpha_range):
            prob_policy = ProbablisticInterpolatedPolicy(self.env, agent_k, agent_km1, alpha)

            agent_trajs = []

            assert (num_trajs > 0 and min_length <= 0) or (min_length > 0 and num_trajs <= 0)
            while (min_length > 0 and np.sum([len(obs) for obs,_,_,_ in agent_trajs])  < min_length) or\
                    (num_trajs > 0 and len(agent_trajs) < num_trajs):
                obs,actions,rewards = gen_traj(self.env,prob_policy,-1)
                agent_trajs.append((obs,actions,rewards))

            # Format change: (alpha, trajectories) --> (alpha, trajectories, iteration)
            trajs.append((alpha, agent_trajs, curr_iter))

        self.trajs = trajs


        with open(os.path.join(logdir,f'prebuilt_{curr_iter}_{curr_iter-1}.pkl'),'wb') as f:
            pickle.dump(self.trajs,f)

        # Now generate an equivalent for the existing BC noisy trajectories.
        # Note that epsilon = 1 - alpha
        if curr_iter == 1:
            trajs = []
            bc_noisy_loaded = self.load_prebuilt(os.path.join(logdir, 'prebuilt.pkl'))
            assert bc_noisy_loaded

            for epsilon, agent_trajs in self.trajs:
                trajs.append((1-epsilon, agent_trajs, 0))
            self.trajs = trajs
            with open(os.path.join(logdir,'prebuilt_0_-1.pkl'),'wb') as f:
                pickle.dump(self.trajs,f)

        
        # Now do the same but with the non-interpolated policy
        trajs = []
        agent_trajs = []
        assert (num_trajs > 0 and min_length <= 0) or (min_length > 0 and num_trajs <= 0)
        while (min_length > 0 and np.sum([len(obs) for obs,_,_,_ in agent_trajs])  < min_length) or\
                (num_trajs > 0 and len(agent_trajs) < num_trajs):
            obs,actions,rewards = gen_traj(self.env, agent_k, -1)
            agent_trajs.append((obs,actions,rewards))

        
        trajs.append((1, agent_trajs, curr_iter))
        self.trajs = trajs
        with open(os.path.join(logdir,f'prebuilt_{curr_iter}.pkl'),'wb') as f:
            pickle.dump(self.trajs,f)

    def draw_fig(self, log_dir, curr_iter, ):
        # Get average return for policy k
        self.load_prebuilt(os.path.join(log_dir, f'prebuilt_{curr_iter}.pkl'))
        k_returns = np.array([[np.sum(rewards) for _,_,rewards in agent_trajs] for _,agent_trajs,_ in self.trajs])
        print("k_returns: ", k_returns)
        k_ave, k_std = np.mean(k_returns), np.std(k_returns)

        # Get returns for interpolated trajectories
        self.load_prebuilt(os.path.join(log_dir, f'prebuilt_{curr_iter}_{curr_iter - 1}.pkl'))
        alpha_levels = [alpha for alpha,_,_ in self.trajs]
        returns = np.array([[np.sum(rewards) for _,_,rewards in agent_trajs] for _,agent_trajs,_ in self.trajs])
        print("interpolated returns: ", np.mean(returns, axis = 1))

        # Get average return for policy (k-1)
        self.load_prebuilt(os.path.join(log_dir, f'prebuilt_{curr_iter - 1}.pkl'))
        km1_returns = np.array([[[np.sum(rewards) for _,_,rewards in agent_trajs] for _,agent_trajs,_ in self.trajs]])
        km1_ave, km1_std = np.mean(km1_returns), np.std(km1_returns)
        print("(k-1) returns: ", km1_returns)

        from_to = [np.min(alpha_levels), np.max(alpha_levels)]

        plt.figure()
        plt.fill_between(from_to,
                         [k_ave - k_std, k_ave - k_std], [k_ave + k_std, k_ave + k_std], alpha = 0.3)
        plt.plot(from_to,[k_ave, k_ave], label='iter k (avg.)')

        plt.fill_between(alpha_levels,
                         np.mean(returns, axis=1)-np.std(returns, axis=1), np.mean(returns, axis=1) + np.std(returns, axis=1), alpha = 0.3)
        plt.plot(alpha_levels, np.mean(returns, axis = 1),'-.', label="interp")

        #plot the average of pure noise in dashed line for baseline
        plt.fill_between(from_to,
                         [km1_ave - km1_std, km1_ave - km1_std], [km1_ave + km1_std, km1_ave + km1_std], alpha = 0.3)
        plt.plot(from_to,[km1_ave, km1_ave], '--', label='iter k-1 (avg.)')

        plt.legend(loc="best")
        plt.xlabel("Alpha")
        #plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir,f"degredation_plot_{curr_iter}_{curr_iter-1}.pdf"))
        plt.close()

    def sample(self,num_samples,include_action=False):
        D = []

        for _ in tqdm(range(num_samples)):
            # Pick Two Noise Level Set
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)
            # print("alpha: ", self.trajs[x_idx][0])
            # Trajectories are acceptably separated if they are from different iterations (random, BC, PPO_k) or
            # have much different alpha/error values
            # (Iteration2 + alpha2) - (Iteration1 + alpha1) >= self.min_margin for this loop to stop
            while abs((self.trajs[x_idx][0] + self.trajs[x_idx][2]) - (self.trajs[y_idx][0] + self.trajs[y_idx][2])) < self.min_margin:
                x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            # Pick trajectory from each set
            x_traj = self.trajs[x_idx][1][np.random.choice(len(self.trajs[x_idx][1]))]
            y_traj = self.trajs[y_idx][1][np.random.choice(len(self.trajs[y_idx][1]))]

            # Subsampling from a trajectory
            if len(x_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(x_traj[0])-self.max_steps)
                x_slice = slice(ptr,ptr+self.max_steps)
            else:
                x_slice = slice(len(x_traj[1]))

            if len(y_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(y_traj[0])-self.max_steps)
                y_slice = slice(ptr,ptr+self.max_steps)
            else:
                y_slice = slice(len(y_traj[0]))

            # Done!
            x_alpha = self.trajs[x_idx][0]
            y_alpha = self.trajs[y_idx][0]
            x_iter = self.trajs[x_idx][2]
            y_iter = self.trajs[y_idx][2]

            x_avg_reward = np.mean([max(t[2]) for t in self.trajs[x_idx][1]])
            y_avg_reward = np.mean([max(u[2]) for u in self.trajs[y_idx][1]])
            if x_avg_reward > y_avg_reward:
                ranking = 0
            elif x_avg_reward < y_avg_reward:
                ranking = 1
            elif x_iter > y_iter:
                ranking = 0
            elif x_iter < y_iter:
                ranking = 1
            elif x_alpha > y_alpha:
                ranking = 0
            else:
                ranking = 1
            if include_action:
                D.append((np.concatenate((x_traj[0][x_slice],x_traj[1][x_slice]),axis=1),
                          np.concatenate((y_traj[0][y_slice],y_traj[1][y_slice]),axis=1),
                          ranking) 
                        )
                        # Higher alpha = better trajectory
            else:
                D.append((x_traj[0][x_slice],
                          y_traj[0][y_slice],
                          ranking)
                        )

        return D

################################
class BCNoisePreferenceDataset(PreferenceDataset):
    def __init__(self,env,max_steps=None,min_margin=None):
        self.env = env

        self.max_steps = max_steps
        self.min_margin = min_margin

    def prebuild(self,agent,noise_range,num_trajs,min_length,logdir):
        trajs = []
        for noise_level in tqdm(noise_range):
            noisy_policy = NoiseInjectedPolicy(self.env,agent,'epsilon',noise_level)

            agent_trajs = []

            assert (num_trajs > 0 and min_length <= 0) or (min_length > 0 and num_trajs <= 0)
            while (min_length > 0 and np.sum([len(obs) for obs,_,_,_ in agent_trajs])  < min_length) or\
                    (num_trajs > 0 and len(agent_trajs) < num_trajs):
                obs,actions,rewards = gen_traj(self.env,noisy_policy,-1)
                agent_trajs.append((obs,actions,rewards))

            trajs.append((noise_level,agent_trajs))

        self.trajs = trajs

        with open(os.path.join(logdir,'prebuilt.pkl'),'wb') as f:
            pickle.dump(self.trajs,f)

    def sample(self,num_samples,include_action=False):
        D = []

        for _ in tqdm(range(num_samples)):
            # Pick Two Noise Level Set
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)
            
            while abs(self.trajs[x_idx][0] - self.trajs[y_idx][0]) < self.min_margin:
                x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            # Pick trajectory from each set
            x_traj = self.trajs[x_idx][1][np.random.choice(len(self.trajs[x_idx][1]))]
            y_traj = self.trajs[y_idx][1][np.random.choice(len(self.trajs[y_idx][1]))]

            # Subsampling from a trajectory
            if len(x_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(x_traj[0])-self.max_steps)
                x_slice = slice(ptr,ptr+self.max_steps)
            else:
                x_slice = slice(len(x_traj[1]))

            if len(y_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(y_traj[0])-self.max_steps)
                y_slice = slice(ptr,ptr+self.max_steps)
            else:
                y_slice = slice(len(y_traj[0]))

            # Done!
            if include_action:
                D.append((np.concatenate((x_traj[0][x_slice],x_traj[1][x_slice]),axis=1),
                          np.concatenate((y_traj[0][y_slice],y_traj[1][y_slice]),axis=1),
                          0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1) # if noise level is small, then it is better traj.
                        )
            else:
                D.append((x_traj[0][x_slice],
                          y_traj[0][y_slice],
                          0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1)
                        )

        return D

################################
class MultiBCNoisePreferenceDataset(BCNoisePreferenceDataset):
    """
    Build a preference dataset from multiple BC policies - one cloned from the demos
    and the rest cloned from learned PPO policies. Have each learned policy generate a
    trajectory, clone it via BC, then degrade each BC policy with random actions and rank
    actions within each degraded policy set by the amount of noise.
    """
    def __init__(self, env, max_steps=None, min_margin=None):
        super().__init__(env, max_steps, min_margin)
        self.tx = None
        self.ty = None
        self.trajs = []

    def prebuild(self,agent,noise_range,num_trajs,min_length,logdir,run, outfile='prebuilt.pkl'):
        trajs = []
        for noise_level in tqdm(noise_range):
            noisy_policy = NoiseInjectedPolicy(self.env,agent,'epsilon',noise_level)

            agent_trajs = []

            assert (num_trajs > 0 and min_length <= 0) or (min_length > 0 and num_trajs <= 0)
            while (min_length > 0 and np.sum([len(obs) for obs,_,_,_ in agent_trajs])  < min_length) or\
                    (num_trajs > 0 and len(agent_trajs) < num_trajs):
                obs,actions,rewards = gen_traj(self.env,noisy_policy,-1)
                agent_trajs.append((obs,actions,rewards))
            trajs.append((noise_level,agent_trajs, run))
            

        self.trajs = trajs

        with open(os.path.join(logdir,outfile),'wb') as f:
            pickle.dump(self.trajs,f)
    
    def sample(self,num_samples,include_action=False):
        D = []

        for _ in tqdm(range(num_samples)):
            # Pick Two Noise Level Set
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            if len(self.trajs[x_idx]) == 2:
                # BC - no run or iter number
                self.tx = (self.trajs[x_idx][0], self.trajs[x_idx][1], -1)
            else:
                self.tx = self.trajs[x_idx]
            if len(self.trajs[y_idx]) == 2:
                self.ty = (self.trajs[y_idx][0], self.trajs[y_idx][1], -1)
            else:
                self.ty = self.trajs[y_idx]

            
            # Don't compare trajectories unless they're from the same run
            while abs(self.tx[0] - self.ty[0]) < self.min_margin or self.tx[2] != self.ty[2]:
                x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)
                if len(self.trajs[x_idx]) == 2:
                    # BC - no run or iter number
                    self.tx = (self.trajs[x_idx][0], self.trajs[x_idx][1], -1)
                else:
                    self.tx = self.trajs[x_idx]
                if len(self.trajs[y_idx]) == 2:
                    self.ty = (self.trajs[y_idx][0], self.trajs[y_idx][1], -1)
                else:
                    self.ty = self.trajs[y_idx]

            # Pick trajectory from each set
            x_traj = self.trajs[x_idx][1][np.random.choice(len(self.trajs[x_idx][1]))]
            y_traj = self.trajs[y_idx][1][np.random.choice(len(self.trajs[y_idx][1]))]

            # Subsampling from a trajectory
            if len(x_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(x_traj[0])-self.max_steps)
                x_slice = slice(ptr,ptr+self.max_steps)
            else:
                x_slice = slice(len(x_traj[1]))

            if len(y_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(y_traj[0])-self.max_steps)
                y_slice = slice(ptr,ptr+self.max_steps)
            else:
                y_slice = slice(len(y_traj[0]))

            # Done!
            if include_action:
                D.append((np.concatenate((x_traj[0][x_slice],x_traj[1][x_slice]),axis=1),
                          np.concatenate((y_traj[0][y_slice],y_traj[1][y_slice]),axis=1),
                          0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1) # if noise level is small, then it is better traj.
                        )
            else:
                D.append((x_traj[0][x_slice],
                          y_traj[0][y_slice],
                          0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1)
                        )

        return D
    
    def draw_fig(self,log_dir,demo_trajs, iter):
        if iter == 0:
            demo_returns = [np.sum(rewards) for _,_,rewards in demo_trajs]
            demo_ave, demo_std = np.mean(demo_returns), np.std(demo_returns)

        else:
            demo_returns = [np.sum(rewards) for _,_,rewards in demo_trajs]
            demo_ave, demo_std = np.mean(demo_returns), np.std(demo_returns)

        noise_levels = [noise for noise,_,_ in self.trajs]
        returns = np.array([[np.sum(rewards) for _,_,rewards in agent_trajs] for _,agent_trajs,_ in self.trajs])

        from_to = [np.min(noise_levels), np.max(noise_levels)]

        plt.figure()
        plt.fill_between(from_to,
                         [demo_ave - demo_std, demo_ave - demo_std], [demo_ave + demo_std, demo_ave + demo_std], alpha = 0.3)
        plt.plot(from_to,[demo_ave, demo_ave], label='demos')

        plt.fill_between(noise_levels,
                         np.mean(returns, axis=1)-np.std(returns, axis=1), np.mean(returns, axis=1) + np.std(returns, axis=1), alpha = 0.3)
        plt.plot(noise_levels, np.mean(returns, axis = 1),'-.', label="bc")

        plt.legend(loc="best")
        plt.xlabel("Epsilon")
        #plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylabel("Return")
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir,f"degredation_plot_{iter}.pdf"))
        plt.close()

        
class PPONoisePreferenceDataset(BCNoisePreferenceDataset):
    """
    A dataset that adds Gaussian random noise to actions
    taken from a PPO policy. 
    """
    def __init__(self, env, max_steps=None, min_margin=None):
        super().__init__(env, max_steps, min_margin)
        self.tx = None
        self.ty = None
    def prebuild(self,agent,noise_range,num_trajs,min_length,logdir, run, iter):
        trajs = []
        for noise_level in tqdm(noise_range):
            noisy_policy = NoiseInjectedPolicy(self.env,agent,'epsilon',noise_level)

            agent_trajs = []

            assert (num_trajs > 0 and min_length <= 0) or (min_length > 0 and num_trajs <= 0)
            while (min_length > 0 and np.sum([len(obs) for obs,_,_,_ in agent_trajs])  < min_length) or\
                    (num_trajs > 0 and len(agent_trajs) < num_trajs):
                obs,actions,rewards = gen_traj(self.env,noisy_policy,-1)
                agent_trajs.append((obs,actions,rewards))
                noisy_policy.reset()

                

            trajs.append((noise_level,agent_trajs, run, iter))

        self.trajs = trajs

        with open(os.path.join(logdir,f'prebuilt_noisy_ppo_{run}.pkl'),'wb') as f:
            pickle.dump(self.trajs,f)

    def sample(self,num_samples,include_action=False):
        D = []

        for _ in tqdm(range(num_samples)):
            # Pick Two Noise Level Set
            x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

            if len(self.trajs[x_idx]) == 2:
                # BC - no run or iter number
                self.tx = (self.trajs[x_idx][0], self.trajs[x_idx][1], -1, -1)
            elif len(self.trajs[x_idx]) == 3:
                # PPO but no iter number
                self.tx = (self.trajs[x_idx][0], self.trajs[x_idx][1], self.trajs[x_idx][2], 1)
            else:
                self.tx = self.trajs[x_idx]
            if len(self.trajs[y_idx]) == 2:
                self.ty = (self.trajs[y_idx][0], self.trajs[y_idx][1], -1, -1)
            elif len(self.trajs[y_idx]) == 3:
                self.ty = (self.trajs[y_idx][0], self.trajs[y_idx][1], self.trajs[y_idx][2], 1)
            else:
                self.ty = self.trajs[y_idx]
            
            # Don't compare trajectories unless they're from the same run
            while abs(self.tx[0] - self.ty[0]) < self.min_margin or self.tx[2] != self.ty[2] or self.tx[3] != self.ty[3]:
                # print("noise difference: ", tx[0] - ty[0], "\t run numbers", f"{tx[2]}/{ty[2]}")
                x_idx,y_idx = np.random.choice(len(self.trajs),2,replace=False)

                if len(self.trajs[x_idx]) == 2:
                    # BC - no run or iter number
                    self.tx = (self.trajs[x_idx][0], self.trajs[x_idx][1], -1, -1)
                elif len(self.trajs[x_idx]) == 3:
                    # PPO but no iter number
                    self.tx = (self.trajs[x_idx][0], self.trajs[x_idx][1], self.trajs[x_idx][2], 1)
                else:
                    self.tx = self.trajs[x_idx]
                if len(self.trajs[y_idx]) == 2:
                    self.ty = (self.trajs[y_idx][0], self.trajs[y_idx][1], -1, -1)
                elif len(self.trajs[y_idx]) == 3:
                    self.ty = (self.trajs[y_idx][0], self.trajs[y_idx][1], self.trajs[y_idx][2], 1)
                else:
                    self.ty = self.trajs[y_idx]
            # Pick trajectory from each set
            x_traj = self.trajs[x_idx][1][np.random.choice(len(self.trajs[x_idx][1]))]
            y_traj = self.trajs[y_idx][1][np.random.choice(len(self.trajs[y_idx][1]))]

            # Subsampling from a trajectory
            if len(x_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(x_traj[0])-self.max_steps)
                x_slice = slice(ptr,ptr+self.max_steps)
            else:
                x_slice = slice(len(x_traj[0]))

            if len(y_traj[0]) > self.max_steps:
                ptr = np.random.randint(len(y_traj[0])-self.max_steps)
                y_slice = slice(ptr,ptr+self.max_steps)
            else:
                y_slice = slice(len(y_traj[0]))

            # Done!
            if include_action:
                D.append((np.concatenate((x_traj[0][x_slice],x_traj[1][x_slice]),axis=1),
                          np.concatenate((y_traj[0][y_slice],y_traj[1][y_slice]),axis=1),
                          0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1) # if noise level is small, then it is better traj.
                        )
            else:
                D.append((x_traj[0][x_slice],
                          y_traj[0][y_slice],
                          0 if self.trajs[x_idx][0] < self.trajs[y_idx][0] else 1)
                        )

        return D

    def draw_fig(self,log_dir, run):
        noise_levels = [noise for noise,_,_,_ in self.trajs]
        returns = np.array([[np.sum(rewards) for _,_,rewards in agent_trajs] for _,agent_trajs,_,_ in self.trajs])

        random_agent = RandomAgent(self.env.action_space)
        random_returns = [np.sum(gen_traj(self.env,random_agent,-1)[2]) for _ in range(20)]
        random_ave, random_std = np.mean(random_returns), np.std(random_returns)

        from_to = [np.min(noise_levels), np.max(noise_levels)]

        plt.figure()

        plt.fill_between(noise_levels,
                         np.mean(returns, axis=1)-np.std(returns, axis=1), np.mean(returns, axis=1) + np.std(returns, axis=1), alpha = 0.3)
        plt.plot(noise_levels, np.mean(returns, axis = 1),'-.', label="PPO")

        #plot the average of pure noise in dashed line for baseline
        plt.fill_between(from_to,
                         [random_ave - random_std, random_ave - random_std], [random_ave + random_std, random_ave + random_std], alpha = 0.3)
        plt.plot(from_to,[random_ave, random_ave], '--', label='random')

        plt.legend(loc="best")
        plt.xlabel("Epsilon")
        #plt.xticks([0.0, 0.25, 0.5, 0.75, 1.0])
        plt.ylabel("Return")
        plt.tight_layout()
        filename = f"degredation_plot_{run}.pdf"
        plt.savefig(os.path.join(log_dir, filename))
        plt.close()

################################


def main(args):
    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)

    # Generate a Noise Injected Trajectories
    env = gym.make(args.env_id)
    env.seed(args.seed)

    if args.use_prob_interp:
        dataset = ProbInterpPreferenceDataset(env)
        agent_k = PPO2Agent(env,'mujoco',str(args.agent_k_path),stochastic=True)

        # Special case for when the (k-1) model is the BC policy
        # It's not a PPO model so is saved and loaded differently
        if args.agent_km1_path is None:
            agent_km1 = RandomAgent(env.action_space)
        elif ".ckpt" in args.agent_km1_path:
            agent_km1 = Policy(env)
            agent_km1.load(args.agent_km1_path)
        else:
            agent_km1 = PPO2Agent(env, 'mujoco', str(args.agent_km1_path), stochastic=True)

        dataset.prebuild(agent_k, agent_km1, eval(args.alpha_range), args.num_trajs, args.min_length, args.log_dir, args.curr_iter)
        
        dataset.draw_fig(args.log_dir, args.curr_iter)
    
    elif args.use_noisy_ppo:
        for run in range(0, 3):
            agent_path = args.agent_k_path.format(run=run)
            agent= PPO2Agent(env,'mujoco',str(agent_path),stochastic=True)
            dataset=PPONoisePreferenceDataset(env)

            dataset.prebuild(agent, eval(args.noise_range), args.num_trajs, args.min_length, args.log_dir, run, 1)
            dataset.draw_fig(args.log_dir, run)
    elif args.use_multi_bc:
        for i, path in enumerate(eval(args.bc_agent_paths)):
            agent = Policy(env)
            agent.load(path)
            dataset = MultiBCNoisePreferenceDataset(env)
            outfile = f'prebuilt_bc_{i}.pkl'
            dataset.prebuild(agent, eval(args.noise_range), 1 if i > 0 else args.num_trajs, args.min_length, args.log_dir, i, outfile)
            
            env_short = "hopper" if "Hopper" in args.env_id else "halfcheetah"
            if i == 0:
                p = f'./demos/suboptimal_demos/{env_short}/dataset.pkl'
            else:
                p = f'./demos/suboptimal_demos/{env_short}/ppo_dataset_{i-1}.pkl'
            with open(p, 'rb') as f:
                demo_trajs = pickle.load(f)
            dataset.draw_fig(args.log_dir, demo_trajs, i)
    else:
        dataset = BCNoisePreferenceDataset(env)
        agent = Policy(env)
        agent.load(args.bc_agent)

        dataset.prebuild(agent,eval(args.noise_range),args.num_trajs,args.min_length,args.log_dir)

        with open(args.demo_trajs,'rb') as f:
            demo_trajs = pickle.load(f)

        dataset.draw_fig(args.log_dir,demo_trajs)

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int, help='seed for the experiments')
    parser.add_argument('--log_dir', required=True, help='log dir')
    parser.add_argument('--env_id', required=True, help='Select the environment to run')
    parser.add_argument('--demo_trajs',default="", help='suboptimal demo trajectories used for bc (used to generate a figure)')
    
    parser.add_argument('--use_prob_interp', action='store_true', help='Whether to use noise injection or probabilistic interpolation')
    # Either specify bc_agent OR (agent_k AND agent_km1)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--bc_agent')
    group.add_argument('--agent_k_path')
    parser.add_argument('--agent_km1_path')

    # Trajectory generation Hyperparams
    parser.add_argument('--num_trajs', default=5,type=int, help='number of trajectory generated by each agent')
    parser.add_argument('--min_length', default=0,type=int, help='minimum length of trajectory generated by each agent')
    # Noise Injection Hyperparams
    parser.add_argument('--noise_range', default='np.arange(0.,1.0,0.05)', help='decide upto what learner stage you want to give')
    # Probabilistic Interpolation Hyperparams
    parser.add_argument('--alpha_range', default='np.arange(1.0, 0., -0.05)', help='Decide range of probabilities for interpolating between policy iterates')
    parser.add_argument('--timestep_range', default='np.arange(20, 360, 20)', help='Decide range of probabilities for interpolating between policy iterates')

    # Other forms of generating the dataset
    parser.add_argument('--use_noisy_ppo', action="store_true", help="Use Gaussian/OU noise added to PPO policy")
    parser.add_argument('--use_multi_bc', action="store_true")
    args = parser.parse_args()

    main(args)
