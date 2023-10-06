import sys
import os
from pathlib import Path
import argparse
import pickle
from functools import partial
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm as std_tqdm
tqdm = partial(std_tqdm, dynamic_ncols=True, disable=eval(os.environ.get("DISABLE_TQDM", 'False')))

import gym

import bc_noise_dataset
from bc_noise_dataset import BCNoisePreferenceDataset, PPORankingPreferenceData, MultiBCNoisePreferenceDataset, PreferenceDataset, ProbInterpPreferenceDataset, PPOTimestepPreferenceDataset, PPONoisePreferenceDataset
from utils import RewardNet, Model

def train_reward(args):
    # set random seed
    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)

    log_dir = Path(args.log_dir)/'trex'
    log_dir.mkdir(parents=True,exist_ok=True)

    with open(str(log_dir/'args.txt'),'w') as f:
        f.write( str(args) )

    env = gym.make(args.env_id)
    env.seed(args.seed)

    ob_dims = env.observation_space.shape[-1]
    ac_dims = env.action_space.shape[-1]
    ############## New for iD-Rex ##################    
    if args.use_prob_interp:
        dataset = ProbInterpPreferenceDataset(env, args.max_steps, args.min_noise_margin)
        p = Path(args.log_dir)
        prebuilt_files = list(set([path.name for path in p.glob('prebuilt_*')]))
        loaded = dataset.load_multiple_prebuilt(p, prebuilt_files)
    elif args.use_noisy_ppo:
        dataset = PPONoisePreferenceDataset(env, args.max_steps, args.min_noise_margin)
        dataset.trajs = []
        loaded = dataset.load_prebuilt(args.noise_injected_trajs)
        assert loaded
        p = Path(args.log_dir)
        prebuilt_files = list(set([path.name for path in p.glob('prebuilt_noisy_ppo_*')]))
        loaded = dataset.load_multiple_prebuilt(p, prebuilt_files)
    elif args.use_multi_bc:
        dataset = MultiBCNoisePreferenceDataset(env, args.max_steps, args.min_noise_margin)
        loaded = dataset.load_prebuilt(args.noise_injected_trajs)
        assert loaded
        p = Path(args.log_dir)
        prebuilt_files = list(set([path.name for path in p.glob('prebuilt_bc_*')]))
        loaded = dataset.load_multiple_prebuilt(p, prebuilt_files)
    ##################################################

    else:
        dataset = BCNoisePreferenceDataset(env,args.max_steps,args.min_noise_margin)

        loaded = dataset.load_prebuilt(args.noise_injected_trajs)
    assert loaded

    models = []
    for i in range(args.num_models):
        with tf.compat.v1.variable_scope('model_%d'%i):
            net = RewardNet(args.include_action,ob_dims,ac_dims,num_layers=args.num_layers,embedding_dims=args.embedding_dims)
            model = Model(net,batch_size=64)
            models.append(model)

    ### Initialize Parameters
    init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                       tf.compat.v1.local_variables_initializer())
    # Training configuration
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.InteractiveSession()

    sess.run(init_op)

    for i,model in enumerate(models):
        D = dataset.sample(args.D,include_action=args.include_action)

        model.train(D,iter=args.iter,l2_reg=args.l2_reg,noise_level=args.noise,debug=True)

        model.saver.save(sess,os.path.join(str(log_dir),'model_%d.ckpt'%(i)),write_meta_graph=False)

    sess.close()

def eval_reward(args):
    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)

    env = gym.make(args.env_id)
    env.seed(args.seed)

    dataset = PreferenceDataset(env)
    ##################### New for iD-Rex ##################    
    if args.use_prob_interp:
        dataset = ProbInterpPreferenceDataset(env, args.max_steps, args.min_noise_margin)
        p = Path(args.log_dir)
        prebuilt_files = list(set([path.name for path in p.glob('prebuilt_*')]))
        loaded = dataset.load_multiple_prebuilt(p, prebuilt_files)
    elif args.use_noisy_ppo:
        dataset = PPONoisePreferenceDataset(env, args.max_steps, args.min_noise_margin)
        loaded = dataset.load_prebuilt(args.noise_injected_trajs)
        assert loaded
        p = Path(args.log_dir)
        prebuilt_files = list(set([path.name for path in p.glob('prebuilt_noisy_ppo_*')]))
        loaded = dataset.load_multiple_prebuilt(p, prebuilt_files)
    elif args.use_multi_bc:
        dataset = MultiBCNoisePreferenceDataset(env, args.max_steps, args.min_noise_margin)
        p = Path(args.log_dir)
        prebuilt_files = list(set([path.name for path in p.glob('prebuilt_bc_*')]))
        loaded = dataset.load_multiple_prebuilt(p, prebuilt_files)
    #########################################################
    else:
        loaded = dataset.load_prebuilt(args.noise_injected_trajs)
    assert loaded

    # Load Seen Trajs

    ##################### New for iD-Rex ##################    
    if args.use_prob_interp or args.use_multi_bc:
        seen_trajs = [
            (obs,actions,rewards, iter) for _,trajs,iter in dataset.trajs for obs,actions,rewards in trajs
        ]
    elif args.use_noisy_ppo:
        seen_trajs = []
        for traj in dataset.trajs:
            if len(traj) == 2:
                seen_trajs.extend([(obs, actions, rewards, 0) for obs, actions, rewards in traj[1]])
            else:
                seen_trajs.extend([(obs, actions, rewards, 1) for obs, actions, rewards in traj[1]])
    #########################################################
    else:
        seen_trajs = [
            (obs,actions,rewards, 0) for _,trajs in dataset.trajs for obs,actions,rewards in trajs
        ]

    # Load Unseen Trajectories
    if args.unseen_trajs:
        with open(args.unseen_trajs,'rb') as f:
            unseen_trajs = pickle.load(f)
    else:
        unseen_trajs = []

    # Load Demo Trajectories used for BC
    with open(args.bc_trajs,'rb') as f:
        bc_trajs = pickle.load(f)

    # Load T-REX Reward Model
    graph = tf.Graph()
    config = tf.compat.v1.ConfigProto() # Run on CPU
    config.gpu_options.allow_growth = True

    with graph.as_default():
        models = []
        for i in range(args.num_models):
            with tf.compat.v1.variable_scope('model_%d'%i):
                net = RewardNet(args.include_action,env.observation_space.shape[-1],env.action_space.shape[-1],num_layers=args.num_layers,embedding_dims=args.embedding_dims)

                model = Model(net,batch_size=1)
                models.append(model)

    sess = tf.compat.v1.Session(graph=graph,config=config)
    for i,model in enumerate(models):
        with sess.as_default():
            model.saver.restore(sess,os.path.join(args.log_dir,'trex','model_%d.ckpt'%i))

    # Calculate Predicted Returns
    def _get_return(obs,acs):
        with sess.as_default():
            return np.sum([model.get_reward(obs,acs) for model in models]) / len(models)

    seen = [1] * len(seen_trajs) + [0] * len(unseen_trajs) + [2] * len(bc_trajs)
    seen_iters = np.array([0] * len(seen_trajs + unseen_trajs + bc_trajs))
    for i, traj in enumerate(seen_trajs):
        seen_iters[i] = traj[3]

    print("seen: ", seen)
    print("seen_iters: ", seen_iters)

    gt_returns, pred_returns = [], []

    for obs,actions,rewards,_ in seen_trajs:
        gt_returns.append(np.sum(rewards))
        pred_returns.append(_get_return(obs,actions))
    for obs,actions,rewards in unseen_trajs+bc_trajs:
        gt_returns.append(np.sum(rewards))
        pred_returns.append(_get_return(obs,actions))

    sess.close()

    # Draw Result
    def _draw(gt_returns,pred_returns,seen,figname=False):
        """
        gt_returns: [N] length
        pred_returns: [N] length
        seen: [N] length
        """
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pylab
        from matplotlib import pyplot as plt
        from imgcat import imgcat

        matplotlib.rcParams.update(matplotlib.rcParamsDefault)
        plt.style.use('ggplot')
        params = {
            'text.color':'black',
            'axes.labelcolor':'black',
            'xtick.color':'black',
            'ytick.color':'black',
            'legend.fontsize': 'xx-large',
            #'figure.figsize': (6, 5),
            'axes.labelsize': 'xx-large',
            'axes.titlesize':'xx-large',
            'xtick.labelsize':'xx-large',
            'ytick.labelsize':'xx-large'}
        matplotlib.pylab.rcParams.update(params)

        def _convert_range(x,minimum, maximum,a,b):
            return (x - minimum)/(maximum - minimum) * (b - a) + a

        def _no_convert_range(x,minimum, maximum,a,b):
            return x

        convert_range = _convert_range
        #convert_range = _no_convert_range

        gt_max,gt_min = max(gt_returns),min(gt_returns)
        pred_max,pred_min = max(pred_returns),min(pred_returns)
        max_observed = np.max(gt_returns[np.where(seen!=1)])

        # Draw P
        fig,ax = plt.subplots()
        print("seen == 1: ", np.where(seen==1))
        print("seen iters == 1: ", np.where(seen_iters==1))
        ##################### New for iD-Rex ##################    
        ax.plot(gt_returns[np.where(seen==0)],
                [convert_range(p,pred_min,pred_max,gt_min,gt_max) for p in pred_returns[np.where(seen==0)]], 'go') # unseen trajs
        ax.plot(gt_returns[np.where(seen==1)],
                [convert_range(p,pred_min,pred_max,gt_min,gt_max) for p in pred_returns[np.where(seen==1)]], 'bo') # seen trajs for T-REX
        ax.plot(gt_returns[np.where(seen_iters==1)],
                [convert_range(p,pred_min,pred_max,gt_min,gt_max) for p in pred_returns[np.where(seen_iters==1)]], 'yo') # seen trajs for iD-REX
        ax.plot(gt_returns[np.where(seen==2)],
                [convert_range(p,pred_min,pred_max,gt_min,gt_max) for p in pred_returns[np.where(seen==2)]], 'ro') # seen trajs for BC
        #########################################################
        ax.plot([gt_min-5,gt_max+5],[gt_min-5,gt_max+5],'k--')
        #ax.plot([gt_min-5,max_observed],[gt_min-5,max_observed],'k-', linewidth=2)
        #ax.set_xlim([gt_min-5,gt_max+5])
        #ax.set_ylim([gt_min-5,gt_max+5])
        ax.set_xlabel("Ground Truth Returns")
        ax.set_ylabel("Predicted Returns (normalized)")
        fig.tight_layout()

        plt.savefig(figname)
        plt.close()

    save_path = os.path.join(args.log_dir,'gt_vs_pred_rewards.pdf')
    _draw(np.array(gt_returns),np.array(pred_returns),np.array(seen),save_path)

def train_rl(args):
    # Train an agent
    #import pynvml as N
    import subprocess, multiprocessing
    ncpu = multiprocessing.cpu_count()
    #N.nvmlInit()
    #ngpu = N.nvmlDeviceGetCount()

    log_dir = Path(args.log_dir)/'rl'
    log_dir.mkdir(parents=True,exist_ok=True)

    model_dir = os.path.join(args.log_dir,'trex')


    kwargs = {
        "model_dir":os.path.abspath(model_dir),
        "ctrl_coeff":args.ctrl_coeff,
        "alive_bonus": 0.
    }

    procs = []
    for i in range(args.rl_runs):
        # Prepare Command
        template = 'python3 -m baselines.run --alg=ppo2 --env={env} --save_path={save_path} --num_env={nenv} --num_timesteps={num_timesteps} --save_interval={save_interval} --custom_reward {custom_reward} --custom_reward_kwargs="{kwargs}" --gamma {gamma} --seed {seed}' # --load_path {load_path}
        
        gt_template = 'python3 -m baselines.run --alg=ppo2 --env={env} --save_path={save_path} --num_env={nenv} --num_timesteps={num_timesteps} --save_interval={save_interval} --gamma {gamma} --seed {seed}' # --load_path {load_path}
        # Prepare to save the learned PPO model
        rl_model_dir = Path(args.log_dir)/'rl'/f'run_{i}'/'ppo_models'
        rl_model_dir.mkdir(parents=True, exist_ok=True)
        rl_model_save_path = os.path.join(str(os.path.abspath(rl_model_dir)),f'model_{args.curr_iter or 1}')


        cmd = template.format(
            env=args.env_id,
            save_path=rl_model_save_path,
            nenv=1, #ncpu//ngpu,
            num_timesteps=args.num_timesteps,
            save_interval=args.save_interval,
            custom_reward='preference_normalized_v2',
            gamma=args.gamma,
            seed=i,
            kwargs=str(kwargs)
        )

        # Prepare Log settings through env variables
        env = os.environ.copy()
        env["OPENAI_LOGDIR"] = os.path.join(str(log_dir.resolve()),'run_%d'%i)
        if i == 0:
            env["OPENAI_LOG_FORMAT"] = 'stdout,log,csv,tensorboard'
            p = subprocess.Popen(cmd, cwd='./learner/baselines', stdout=subprocess.PIPE, env=env, shell=True)
        else:
            env["OPENAI_LOG_FORMAT"] = 'log,csv,tensorboard'
            p = subprocess.Popen(cmd, cwd='./learner/baselines', env=env, shell=True)

        # run process
        procs.append(p)

    for line in procs[0].stdout:
        print(line.decode(),end='')

    for p in procs[1:]:
        p.wait()

def eval_rl(args):
    np.random.seed(args.seed)
    tf.compat.v1.random.set_random_seed(args.seed)

    from utils import PPO2Agent, gen_traj

    env = gym.make(args.env_id)
    env.seed(args.seed)

    graph = tf.Graph()
    config = tf.compat.v1.ConfigProto() # Run on CPU
    config.gpu_options.allow_growth = True

    with graph.as_default():
        models = []
        for i in range(args.num_models):
            with tf.compat.v1.variable_scope('model_%d'%i):
                net = RewardNet(args.include_action,env.observation_space.shape[-1],env.action_space.shape[-1],num_layers=args.num_layers,embedding_dims=args.embedding_dims)

                model = Model(net,batch_size=1)
                models.append(model)

    sess = tf.compat.v1.Session(graph=graph,config=config)
    for i,model in enumerate(models):
        with sess.as_default():
            model.saver.restore(sess,os.path.join(args.log_dir,'trex','model_%d.ckpt'%i))

    # Calculate Predicted Returns
    def _get_return(obs,acs):
        with sess.as_default():
            return np.sum([model.get_reward(obs,acs) for model in models]) / len(models)



    def _get_perf(agent, num_eval=20):
        V = []
        for _ in range(num_eval):
            obs,actions,R = gen_traj(env,agent,-1)
            pred_return = _get_return(obs, actions)
            print("predicted vs GT return: ", pred_return, "/", np.sum(R))
            V.append(np.sum(R))
        return V

    with open(os.path.join(args.log_dir,'rl_results_clip_action.txt'),'w') as f:
        # Load T-REX learned agent
        agents_dir = Path(os.path.abspath(os.path.join(args.log_dir,'rl')))

        trained_steps = sorted(list(set([path.name for path in agents_dir.glob('run_*/checkpoints/?????')])))
        for step in trained_steps[::-1]:
            perfs = []
            for i in range(args.rl_runs):
                path = agents_dir/('run_%d'%i)/'checkpoints'/step

                if path.exists() == False:
                    continue

                agent = PPO2Agent(env,'mujoco',str(path),stochastic=True)
                agent_perfs = _get_perf(agent)
                print('[%s-%d] %f %f'%(step,i,np.mean(agent_perfs),np.std(agent_perfs)))
                print('[%s-%d] %f %f'%(step,i,np.mean(agent_perfs),np.std(agent_perfs)),file=f)

                perfs += agent_perfs
            print('[%s] %f %f %f %f'%(step,np.mean(perfs),np.std(perfs),np.max(perfs),np.min(perfs)))
            print('[%s] %f %f %f %f'%(step,np.mean(perfs),np.std(perfs),np.max(perfs),np.min(perfs)),file=f)

            f.flush()

######################### New for iD-Rex ######################
def train_idrex(args):
    # Assumption: You've run all the other steps up to and including PPO. Now it's time to iterate.

    num_iters = 1

    args.use_prob_interp = False # Use probabilistic interpolation and not noise injection between iterations
    args.use_noisy_ppo = False # Add Gaussian/OU noise to the PPO policy 
    args.use_multi_bc = True

    # Assumption: at this point, we'll have finished rebuilding the interpolated dataset. Now we must rebuild trajectories
    # and train
    for i in range(1, num_iters + 1):
        args.curr_iter = i
        # rebuild trajectories here. Use model from run 0
        args.agent_k_path = os.path.join(args.log_dir,'rl', 'run_0', 'checkpoints', '00140') # best model seen
        
        # Special case for first iD-REX iteration: agent (k-1) is the BC agent.
        if i == 1:
            args.agent_km1_path = os.path.join(args.log_dir, 'bc', 'model.ckpt')
        if args.use_prob_interp:
            args.alpha_range = 'np.arange(1.0, 0., -0.05)'
        elif args.use_noisy_ppo:
            args.noise_range = 'np.arange(0., 1., 0.05)'
        elif args.use_multi_bc:
            # Take advantage of *all* RL run policies
            args.agent_k_path = os.path.join(args.log_dir,'rl', 'run_{run}', 'checkpoints', '00140')
            args.noise_range = 'np.arange(0., 1., 0.05)'
            args.bc_agent_paths = str([
                os.path.join(args.log_dir,'bc', 'model.ckpt'),
                os.path.join(args.log_dir,'bc', 'bc_ppo_model_0.ckpt'),
                os.path.join(args.log_dir,'bc', 'bc_ppo_model_1.ckpt'),
                os.path.join(args.log_dir,'bc', 'bc_ppo_model_2.ckpt'),
            ])

        args.num_trajs=5
        args.min_length=0
        bc_noise_dataset.main(args)

        if "hopper" in str(args.log_dir):
            env_short = "hopper"
        else:
            env_short = "halfcheetah"
        args.noise_injected_trajs=os.path.join(args.log_dir, f'prebuilt.pkl')
        args.bc_trajs = f'./demos/suboptimal_demos/{env_short}/dataset.pkl'
        args.unseen_trajs= f'./demos/full_demos/{env_short}/trajs.pkl'
        args.min_noise_margin = 0.3 # Minimum difference between two alpha levels for sampling

        train_reward(args)
        tf.compat.v1.reset_default_graph()
        eval_reward(args)

        train_rl(args)
        eval_rl(args)
###############################################################    

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--seed', default=0, type=int, help='seed for the experiments')
    parser.add_argument('--log_dir', required=True)
    parser.add_argument('--env_id', required=True, help='Select the environment to run')
    parser.add_argument('--mode', default='train_reward',choices=['all','train_reward','eval_reward','train_rl','eval_rl', 'train_idrex'])
    
    # iD-REX args
    parser.add_argument('--use_prob_interp', action='store_true', help='Whether to use noise injection or probabilistic interpolation')
    parser.add_argument('--use_noisy_ppo', action='store_true', help='Whether to use Gaussian/OU noise added to PPO')
    parser.add_argument('--use_multi_bc', action="store_true")
    parser.add_argument('--curr_iter', type=int, default=None)


    # Args for T-REX
    ## Dataset setting
    parser.add_argument('--noise_injected_trajs', default='')
    parser.add_argument('--unseen_trajs', default='', help='used for evaluation only')
    parser.add_argument('--bc_trajs', default='', help='used for evaluation only')
    parser.add_argument('--D', default=5000, type=int, help='|D| in the preference paper')
    parser.add_argument('--max_steps', default=50, type=int, help='maximum length of subsampled trajecotry')
    parser.add_argument('--min_noise_margin', default=0.3, type=float, help='')
    parser.add_argument('--include_action', action='store_true', help='whether to include action for the model or not')
    ## Network setting
    parser.add_argument('--num_layers', default=2, type=int, help='number layers of the reward network')
    parser.add_argument('--embedding_dims', default=256, type=int, help='embedding dims')
    parser.add_argument('--num_models', default=3, type=int, help='number of models to ensemble')
    parser.add_argument('--l2_reg', default=0.01, type=float, help='l2 regularization size')
    parser.add_argument('--noise', default=0.0, type=float, help='noise level to add on training label (another regularization)')
    parser.add_argument('--iter', default=3000, type=int, help='# trainig iters')
    # Args for PPO
    parser.add_argument('--rl_runs', default=3, type=int)
    parser.add_argument('--num_timesteps', default=int(1e6), type=int)
    parser.add_argument('--save_interval', default=20, type=int)
    parser.add_argument('--ctrl_coeff', default=0.0, type=float)
    parser.add_argument('--gamma', default=0.99, type=float)
    args = parser.parse_args()

    if args.mode == 'train_reward':
        train_reward(args)
        tf.compat.v1.reset_default_graph()
        eval_reward(args)
    elif args.mode == 'eval_reward':
        eval_reward(args)
    elif args.mode =='train_rl':
        train_rl(args)
        tf.compat.v1.reset_default_graph()
        eval_rl(args)
    elif args.mode == 'eval_rl':
        eval_rl(args)
    elif args.mode == 'train_idrex':
        train_idrex(args)
    else:
        assert False

