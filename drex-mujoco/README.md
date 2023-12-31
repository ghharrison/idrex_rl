# iD-REX mujoco

This is modified from the code for DREX provided [here](https://github.com/dsbrown1331/CoRL2019-DREX/tree/master/drex-mujoco), meant to accompany
the paper on iD-REX this was submitted with. The instructions below pertain to the multi-BC approach to iD-REX.

## Run Experiment

The below should all be run from the directory `drex-mujoco`.

1. Behavior Cloning (BC)

```
python3 bc_mujoco.py --env_id HalfCheetah-v2 --log_path ./log/drex/halfcheetah/bc --demo_trajs demos/suboptimal_demos/halfcheetah/dataset.pkl
python3 bc_mujoco.py --env_id Hopper-v2 --log_path ./log/drex/hopper/bc --demo_trajs demos/suboptimal_demos/hopper/dataset.pkl
```

2. Generate Noise Injected Trajectories

```
python3 bc_noise_dataset.py --log_dir ./log/drex/halfcheetah --env_id HalfCheetah-v2 --bc_agent ./log/drex/halfcheetah/bc/model.ckpt --demo_trajs ./demos/suboptimal_demos/halfcheetah/dataset.pkl
python3 bc_noise_dataset.py --log_dir ./log/drex/hopper --env_id Hopper-v2 --bc_agent ./log/drex/hopper/bc/model.ckpt --demo_trajs ./demos/suboptimal_demos/hopper/dataset.pkl
```

3. Run T-REX

```
python3 drex.py --log_dir ./log/drex/halfcheetah --env_id HalfCheetah-v2 --bc_trajs ./demos/suboptimal_demos/halfcheetah/dataset.pkl --unseen_trajs ./demos/full_demos/halfcheetah/trajs.pkl --noise_injected_trajs ./log/drex/halfcheetah/prebuilt.pkl
python3 drex.py --log_dir ./log/drex/hopper --env_id Hopper-v2 --bc_trajs ./demos/suboptimal_demos/hopper/dataset.pkl --unseen_trajs ./demos/full_demos/hopper/trajs.pkl --noise_injected_trajs ./log/drex/hopper/prebuilt.pkl
```

You can download pregenerated unseen trajectories from [here](https://github.com/dsbrown1331/CoRL2019-DREX/releases). Instead, you can just erase the `--unseeen_trajs` option. It is just used for generating the plot shown in the paper.

4. Run PPO

```
python3 drex.py --log_dir ./log/drex/halfcheetah --env_id HalfCheetah-v2 --mode train_rl --ctrl_coeff 0.1
python3 drex.py --log_dir ./log/drex/hopper --env_id Hopper-v2 --mode train_rl --ctrl_coeff 0.001
```

5. Run multi-BC
```
python3 bc_mujoco.py --env_id HalfCheetah-v2 --log_path ./log/drex/halfcheetah/bc --ppo_dir ./log/drex/halfcheetah/rl --use_ppo --demo_trajs ./demos/suboptimal_demos/halfcheetah/dataset.pkl --multi 
python3 bc_mujoco.py --env_id Hopper-v2 --log_path ./log/drex/hopper/bc --ppo_dir ./log/drex/hopper/rl --use_ppo --demo_trajs ./demos/suboptimal_demos/hopper/dataset.pkl --multi
```

6. Run iD-REX to train reward
```
python3 drex.py --log_dir ./log/drex/halfcheetah --env_id HalfCheetah-v2 --mode train_idrex --ctrl_coeff 0.1
python3 drex.py --log_dir ./log/drex/hopper --env_id Hopper-v2 --mode train_idrex --ctrl_coeff 0.001
```

7. Run PPO (again)

```
python3 drex.py --log_dir ./log/drex/halfcheetah --env_id HalfCheetah-v2 --mode train_rl --ctrl_coeff 0.1
python3 drex.py --log_dir ./log/drex/hopper --env_id Hopper-v2 --mode train_rl --ctrl_coeff 0.001
```