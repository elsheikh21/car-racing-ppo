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

## Pre-implementation analysis

- Environments are basically categorized for 2 parts

  1. Episodic
     - List of states `s`, actions `u`, rewards `r`, and of course new states `s'`
  2. Continuous
     - No terminal State

- Two ways of learning

  1. Monte Carlo Approach

     1. Collecting rewards after the end of episode
     2. Calculate max expected future reward

     - gets better by iteration
       - `V(s`<sup>`t`</sup>`) <- V(s`<sup>`t`</sup>`) + α(R(t) - V(s`<sub>`t`</sub>`))`
       - max expected future reward starting from this state <-- former estimation + learning rate \* (return - estimation of reward)
       - Problem with this approach: we calculate rewards at the end of every episode, we average all actions, even if some bad actions took place this will result in averaging them as good actions if the end result (as per episode) was good.
       - Every problem has a solution:
         - [Actor Critic](actor_critic.md): hybrid between policy based and value based methods
         - Proximal Policy Gradient: Ensures deviation from previous policy is relatively small

  2. Temporal Difference
     1. Estimate the reward at each step, gets better each step
     - `V(s`<sup>`t`</sup>`) <- V(s`<sup>`t`</sup>`) + α(R(t+1) + γV(S[t+1]) - V(s`<sub>`t`</sub>`))`

- However, we always need to balance the tradeoffs between exploration & exploitation.

- Basically we have 3 approaches to RL

  1. Value based (e.g. Q-Learning & DQN, Value Iteration)
     - Goal to optimize the value function `V(s)`
       - V(s) tells us the maximum expected future reward agent will get at each state
  2. Policy Based
     - Goal directly optimize policy function
       - action = π(state)
       - Policy `π` might be deterministic or stochastic
       - stochastic is better as it smooths the distribution of actions probability
  3. Model Based
     - model the environment and model its behavior

- In this project, I am implementing policy based approach

  - my motive

    1. I already explored Value Based approaches (Q-Learning)
    2. I know nothing about the model based approach
    3. Policy based approach is very good for continuous action space and more effective in high dimensional space of observations
    4. Convergence, value based has oscillations while training
    5. Policy based follows policy gradient trying to find the best parameters and smooth the update at each step
    6. Finally, it learns stochastic policies, so no need to implement explore exploit strategy.

  - There is no free lunch

    1. It might converge to a local maximum
    2. It takes more time to train compared to value based functions

- Policy based RL approach:
  - Instead of learning value function that tell us the expected sum of rewards given a state and action, we learn directly the policy function that maps the state to action thus selecting action without using the value function.
  - We have the value function V(s) helps us optimize the policy but it does not select an action for us.
  - Policy `pi` has parameters `theta` and basically policy based methods are viewed as optimization problems, and we are searching for best params to maximize the score function.
    - How?
      - Measure quality of policy with a policy score function (objective function, calculating expected rewards of policy)
      - Use policy gradient descent or ascent to find the best policy parameters improving our policy

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
