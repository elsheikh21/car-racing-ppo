# Proximal Policy Optimizer Implementation

**Goal**: Implement [PPO](docs/proximal-policy-optimizer.md) for Car Racing environment

---

## First things first

- [How to install openai gym and get started?](docs/how-to-get-started.md)

### For more information about

- [Policy Gradient](policy_gradient.md)
- [Natural Policy Gradient](natural_policy_gradient.md)
- [Trust Region Policy Optimization [TRPO]](trust-region-policy-optimization.md)
- [Proximal Policy Optimization [PPO]](proximal-policy-gradient.md)

---

## About the game

- Action Space is `Box(3, )`, which is `[Steering, gas, breaks]`

- Observation Space is `Box(96, 96, 3)`

- For random agent run in command prompt `python -u CarRacing-run_random_agent.py` or check it [here](CarRacing-run_random_agent.py)

- For implementation of PPO
  - Finalizing it

---

## End Result

- After training for 280,000 episodes (equivalent to +36 hours) on GPU - NIVIDIA GeForce GTX 950M -

![GIF for agent after training for 36+ hours](docs/image/carracing-ppo.gif)
