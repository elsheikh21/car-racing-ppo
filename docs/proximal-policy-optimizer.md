# Proximal Policy Optimizer [PPO]

It is a member of a new family of reinforcement learning methods known as [Policy Gradient methods](policy-optimization.md), which basically performs one policy update as per sample, however, PPO alternates between

1. Sampling data by interacting with the environment
2. Optimizing 'surrogate' objective function using stochastic gradient ascent.

Why PPO?

1. It has some benefits of Trust Region Policy Optimization [TRPO], but much simpler (in terms of implementation), more general, and have better sample complexity.
2. It outperforms other online policy gradient methods, and overall strikes a favorable balance between sample complexity and simplicity, and wall-time.

Why not the other RL methods?

- Q-Learning with function approximation fails on many simple problems
- Vanilla policy gradient methods have poor data efficiency and robustness
- TRPO is relatively complicated and not compatible with architectures that include noise (such as dropout) or parameter sharing (between the policy and value function, or with auxiliary tasks).

> So we are trying to modify TRPO, by implementing an algorithm that attains its efficiency and reliable performance, but using 1st order optimization, which is the PPO.

To optimize policies, we need to alternate between sampling data from policy, and perform several epochs of optimization on the sampled data.
