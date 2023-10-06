# idrex_rl
iD-REX: Iterated Disturbance-based Reward Extrapolation. This was the term project for CSC2626H: Imitation Learning for Robotics, taught at the University of Toronto by Professor Florian Shkurti. This project was a collaboration between Geoffrey Harrison, Jazib Ahmad, and Anh Tuan Tran. 

D-REX, or Disturbance-based Reward Extrapolation, is an algorithm for Inverse Reinforcement Learing (IRL) that begins by training a behavioral cloning (BC) policy and injecting increasing amounts of random noise into the policy's actions, obtaining a series of policies that have a roughly linear degradation of trajectory returns as more noise is added; essentially "interpolating" between the BC policy and the uniform random policy. By ranking trajectories generated by these policies according to the amount of noise injected into the trajectories' originating policy, D-REX learns to extrapolate from this pattern to learn a policy with superior returns to even the best demonstration initially used to train the BC policy. Training is done using a preference-based reward on top of Proximal Policy Optimization (PPO).

The motivation of this project was to make D-REX iterative by extending the generated series of policies by interpolating between the learned policy from D-REX and the BC policy, then repeating the process with subsequent iterations of learned policy. Three approaches were conceived of for this:

1. **Probabilistic policy iterpolation**: Creating a noisy policy that "interpolates" between policy T and policy T-1 by, at each step, selecting an action from policy T with probability _p_ and policy T-1 with probability _1-p_. As _p_ goes from 1 to 0, the resulting policy action distributions will more strongly resemble that of the older (and presumably worse) policy.
2. **Action noise degradation**: Forgoing the notion of policy interpolation, here we attempted to use similar methods as the D-REX paper by simply degrading the newly learned policy on each iteration with action noise. Contrary to initial expectations, unlike with the BC policy, degrading policies from later training iterations in this way does not result in a linear degradation in performance.
3. **Multi-BC approach**: We simply repeat the methodology of D-REX, but at each iteration, rather than using the demonstrations as the source of training data for the BC model, we use the trajectories generated by the previous iteration of training as the demonstrations. We then proceed to degrade these new BC policies using random action noise as in D-REX. As each iteration of training produces policies superior to both the initial demonstrations and the previous iterations of training, the resulting range of automatically-ranked policies is wider and therefore more representative of the true spectrum of policies.

Of these approaches, the multi-BC approach was demonstrated to achieve consistently higher returns than both behavioral cloning and D-REX.

This project was built upon the source code for D-REX, available at https://github.com/dsbrown1331/CoRL2019-DREX. Changes to the code are marked by sections with comments reading `############## New for iD-Rex ################## `.

# References
D. S. Brown, W. Goo, and S. Niekum. Better-than-demonstrator imitation learning via automatically-ranked demonstrations. In L. P. Kaelbling, D. Kragic, and K. Sugiura, editors, Proceedings of the Conference on Robot Learning, volume 100 of Proceedings of Machine Learning Research, pages 330–359. PMLR, 30 Oct–01 Nov 2020. URL https://proceedings.mlr.press/v100/brown20a.html.
