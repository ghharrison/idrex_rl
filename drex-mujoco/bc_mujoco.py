import os
import argparse
import pickle
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm as std_tqdm
from functools import partial
tqdm = partial(std_tqdm, dynamic_ncols=True, disable=eval(os.environ.get("DISABLE_TQDM", 'False')))

from utils import PPO2Agent, gen_traj

import gym
from tf_commons.ops import Linear

class Policy(object):
    def __init__(self,env,num_layers=4,embed_size=256):
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.shape[0]

        self.graph = tf.Graph()

        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(graph=self.graph,config=config)

        with self.graph.as_default():
            self.inp = tf.compat.v1.placeholder(tf.float32,[None,ob_dim])
            self.l = tf.compat.v1.placeholder(tf.float32,[None,ac_dim])
            self.l2_reg = tf.compat.v1.placeholder(tf.float32,[])

            with tf.compat.v1.variable_scope('weights') as param_scope:
                self.param_scope = param_scope

                fcs = []
                last_dims = ob_dim
                for l in range(num_layers):
                    fcs.append(Linear('fc%d'%(l+1),last_dims,embed_size)) #(l+1) is gross, but for backward compatibility
                    last_dims = embed_size
                fcs.append(Linear('fc%d'%(num_layers+1),last_dims,ac_dim))

            # build graph
            def _build(x):
                for fc in fcs[:-1]:
                    x = tf.nn.relu(fc(x))
                pred_a = fcs[-1](x)
                return pred_a

            self.ac = _build(self.inp)

            loss = tf.reduce_sum((self.ac-self.l)**2,axis=1)
            self.loss = tf.reduce_mean(loss,axis=0)

            weight_decay = 0.
            for fc in fcs:
                weight_decay += tf.reduce_sum(fc.w**2)

            self.l2_loss = self.l2_reg * weight_decay

            self.optim = tf.compat.v1.train.AdamOptimizer(1e-4)
            self.update_op = self.optim.minimize(self.loss+self.l2_loss,var_list=self.parameters(train=True))

            self.saver = tf.compat.v1.train.Saver(var_list=self.parameters(train=False),max_to_keep=0)

            ################ Miscellaneous
            self.init_op = tf.group(tf.compat.v1.global_variables_initializer(),
                                    tf.compat.v1.local_variables_initializer())

        self.sess.run(self.init_op)

    def parameters(self,train=False):
        with self.graph.as_default():
            if train:
                return tf.compat.v1.trainable_variables(self.param_scope.name)
            else:
                return tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,self.param_scope.name)

    def train(self,D,batch_size,iter,l2_reg,debug=False):
        sess = self.sess

        obs,acs,_ = D

        idxes = np.random.permutation(len(obs)-1)
        train_idxes = idxes[:int(len(obs)*0.8)]
        valid_idxes = idxes[int(len(obs)*0.8):]

        def _batch(idx_list):
            if len(idx_list) > batch_size:
                idxes = np.random.choice(idx_list,batch_size,replace=False)
            else:
                idxes = idx_list

            batch = []
            for i in idxes:
                batch.append((obs[i],acs[i]))
            b_s,b_a = [np.array(e) for e in zip(*batch)]

            return b_s,b_a

        for it in tqdm(range(iter),dynamic_ncols=True):
            b_s,b_a = _batch(train_idxes)

            with self.graph.as_default():
                loss,l2_loss,_ = sess.run([self.loss,self.l2_loss,self.update_op],feed_dict={
                    self.inp:b_s,
                    self.l:b_a,
                    self.l2_reg:l2_reg,
                })

            if debug:
                if it % 100 == 0 or it < 10:
                    b_s,b_a= _batch(valid_idxes)
                    valid_loss = sess.run(self.loss,feed_dict={
                        self.inp:b_s,
                        self.l:b_a,
                    })
                    std_tqdm.write(('loss: %f (l2_loss: %f), valid_loss: %f'%(loss,l2_loss,valid_loss)))

    def act(self, observation, reward, done, clip=False):
        sess = self.sess

        with self.graph.as_default():
            ac = sess.run(self.ac,feed_dict={self.inp:observation[None]})[0]

        if clip:
            ac = np.clip(ac,self.action_space.low,self.action_space.high)

        return ac

    def save(self,path):
        with self.graph.as_default():
            prefix = self.saver.save(self.sess,path,write_meta_graph=False)
            print("Saved to prefix: ", prefix)

    def load(self,path):
        with self.graph.as_default():
            self.saver.restore(self.sess,path)

class Dataset(object):
    def __init__(self, trajs):
        with open(trajs,'rb') as f:
            self.trajs = pickle.load(f)

        V = [np.sum(rewards) for obs,acs,rewards in self.trajs]
        print(f'average demo performance: {np.mean(V)}')

    def getD(self):
        obs,actions,rewards = zip(*self.trajs)
        obs,actions,rewards = (np.concatenate(obs,axis=0),np.concatenate(actions,axis=0),np.concatenate(rewards,axis=0))

        return obs,actions,rewards

############## New for iD-Rex ##################    
class PPODataset(Dataset):
    def __init__(self, logdir, env, run=0):
        self.trajs = []
        rl_model_path = Path(logdir)/f'run_{run}'/'checkpoints'/"00140"
        agent = PPO2Agent(env, 'mujoco', str(rl_model_path), stochastic=True)
        obs,actions,rewards = gen_traj(env,agent,-1)
        self.trajs.append((obs, actions, rewards))

        V = [np.sum(rewards) for obs,acs,rewards in self.trajs]
        print(f'average PPO policy demo performance: {np.mean(V)}')

        env_short = "hopper" if "hopper" in str(logdir) else "halfcheetah"
        with open(os.path.join(f'./demos/suboptimal_demos/{env_short}', f'ppo_dataset_{run}.pkl'),'wb') as f:
            pickle.dump(self.trajs,f)
##################################################

def bc(args):
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    logdir = Path(args.log_path)
    logdir.mkdir(parents=True,exist_ok=True)

    with open(str(logdir/'args.txt'),'w') as f:
        f.write( str(args) )

    env = gym.make(args.env_id)
    env.seed(args.seed)

    policy = Policy(env,args.num_layers,args.embed_size)
    if args.use_ppo:
        dataset = PPODataset(args.ppo_dir, env, args.run)
    else:
        dataset = Dataset(args.demo_trajs)

    try:
        D = dataset.getD()
        policy.train(D,args.batch_size,args.num_iter,l2_reg=args.l2_reg,debug=True)
    except KeyboardInterrupt:
        pass
    if args.multi:
        fname = f'bc_ppo_model_{args.run}.ckpt'
    else:
        fname = 'model.ckpt'
    

    policy.save(os.path.join(logdir,fname))

if __name__ == "__main__":
    # Required Args (target envs & learners)
    parser = argparse.ArgumentParser(description=None)
    # Train Setting
    parser.add_argument('--seed', default=0, type=int, help='seed for the experiments')
    parser.add_argument('--log_path', required=True, help='log dir')
    parser.add_argument('--env_id', required=True, help='Select the environment to run')
    # Dataset
    parser.add_argument('--demo_trajs',required=True, help='suboptimal demo trajectories')
    # Network
    parser.add_argument('--num_layers', default=4,type=int)
    parser.add_argument('--embed_size', default=256,type=int)
    # Training
    parser.add_argument('--l2_reg', default=0.001, type=float, help='l2 regularization size')
    parser.add_argument('--num_iter',default=50000,type=int)
    parser.add_argument('--batch_size',default=128,type=int)

    # BC from PPO
    parser.add_argument('--use_ppo', action="store_true", help="Use PPO policy to generate demos")
    parser.add_argument('--ppo_dir', default="", help="dir for PPO runs")
    parser.add_argument('--run', default=0, help="which PPO run to use for BC")
    parser.add_argument('--multi', action="store_true", help="Use multiple PPO policies as the basis for the BC")

    args = parser.parse_args()

    if args.multi:
        for i in range(0, 3):
            args.run = i
            bc(args)
    else:
        bc(args)
