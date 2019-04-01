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

- For random agent run in command prompt `python -u CarRacing-run_random_agent.py` or check it [here](CarRacing-run_random_agent.py)

- Action Space is `Box(3, )`, which is `[Steering, gas, breaks]`

- For implementation of PPO [Based on A3C]

  1. Create command line interface

     ```{python}
     import argparse

     def parse_arg():
          parser = argparse.ArgumentParser(
               description="Trains an agent in a the CarRacing-v0     environment with proximal policy optimization")

          # Hyper-parameters
          parser.add_argument("--initial_lr", type=float, default=3e-4)
          parser.add_argument("--discount_factor", type=float, default=0.99)
          parser.add_argument("--gae_lambda", type=float, default=0.95)
          parser.add_argument("--ppo_epsilon", type=float, default=0.2)
          parser.add_argument("--value_scale", type=float, default=0.5)
          parser.add_argument("--entropy_scale", type=float, default=0.01)
          parser.add_argument("--horizon", type=int, default=128)
          parser.add_argument("--num_epochs", type=int, default=10)
          parser.add_argument("--batch_size", type=int, default=128)
          parser.add_argument("--num_envs", type=int, default=16)

          # Training vars
          parser.add_argument("--model_name", type=str, default='CarRacing-v0')
          parser.add_argument("--save_interval", type=int, default=1000)
          parser.add_argument("--eval_interval", type=int, default=200)
          parser.add_argument("--record_episodes", type=bool, default=True)
          parser.add_argument("-restart", action="store_true")

          params = vars(parser.parse_args())
          return params

     ```

  2. Input states here as mentioned before are of `Box(96, 96, 3)`, which we will need to remove color frames, as they add extra params for the computations, and then we will crop unnecessary pieces of information by cropping the frame, followed by down-sampling

     ```{python}
     def crop(frame):
       # Crop to 84x84
       return frame[:-12, 6:-6]

     def rgb2grayscale(frame):
       # change to grayscale
       return np.dot(frame[..., 0:3], [0.299, 0.587, 0.114])

     def normalize(frame):
       return frame / 255.0

     def preprocess_frame(frame):
       frame = crop(frame)
       frame = rgb2grayscale(frame)
       frame = normalize(frame)
       frame = frame * 2 - 1
       return frame

     ```

  3. Start the training part

     1. Create env
        ```{python}
         env = gym.make('carracing-v0')
        ```
     2. Set the training parameters (initial_lr, discount_factor, gae_lambda, ppo_epsilon, value_scale, entropy_scale, horizon, num_epochs, batch_size, num_envs)

        ```{python}
         # Traning parameters
         initial_lr = params["initial_lr"]
         discount_factor = params["discount_factor"]
         gae_lambda = params["gae_lambda"]
         ppo_epsilon = params["ppo_epsilon"]
         value_scale = params["value_scale"]
         entropy_scale = params["entropy_scale"]
         horizon = params["horizon"]
         num_epochs = params["num_epochs"]
         batch_size = params["batch_size"]
         num_envs = params["num_envs"]
        ```

     3. Set env constants (frame_stack_size, input_shape, num_actions, action_min, action_max)

        ```{python}
         # Environment constants
         frame_stack_size = 4
         input_shape = (84, 84, frame_stack_size)
         num_actions = test_env.action_space.shape[0]
         action_min = test_env.action_space.low
         action_max = test_env.action_space.high
        ```

     4. Create the model

        ```{python}
        model = PPO(input_shape, num_actions, action_min, action_max,
                epsilon=ppo_epsilon,
                value_scale=value_scale, entropy_scale=entropy_scale,
                model_name=model_name)
        ```

        1. Create policy gradient train function
           1. Create placeholders for returns and advantage.
        2. Calculate ratio

           1. π<sub>θ</sub>(a|s) / π<sub>θ<sub>old</sub></sub>(a|s)
              - r_t(θ) = π(a_t | s_t; θ) / π(a_t | s_t; θ_old)
              - r_t(θ) = exp( log ( π(a_t | s_t; θ) / π(a_t | s_t; θ_old) ) )
              - r_t(θ) = exp( log π(a_t | s_t; θ) - log π(a_t | s_t; θ_old) )

           ```{python}
            self.prob_ratio = tf.exp(
               self.policy.action_log_prob - self.policy_old.action_log_prob)
           ```

        3. Validate values
        4. Policy loss

           ```{python}
            adv = tf.expand_dims(self.advantage, axis=-1)
            self.policy_loss = tf.reduce_mean(tf.minimum(self.prob_ratio * adv,
                  tf.clip_by_value(self.prob_ratio, 1.0 - epsilon, 1.0 + epsilon) * adv))
           ```

        5. value loss

           ```{python}
            self.value_loss = tf.reduce_mean(tf.squared_difference(
            tf.squeeze(self.policy.value), self.returns)) * value_scale
           ```

        6. entropy loss

           ```{python}
            self.entropy_loss = tf.reduce_mean(tf.reduce_sum(
            self.policy.action_normal.entropy(), axis=-1)) * entropy_scale
           ```

        7. total loss

           ```{python}
            self.loss = -self.policy_loss + self.value_loss - self.entropy_loss
           ```

        8. policy parameters

           ```{python}
            policy_params = tf.get_collection(
               tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy/")
            policy_old_params = tf.get_collection(
               tf.GraphKeys.TRAINABLE_VARIABLES, scope="policy_old/")
           ```

        9. Minimize the loss

           ```{python}
            self.learning_rate = tf.placeholder(
                 shape=(), dtype=tf.float32, name="lr_placeholder")
            self.optimizer = tf.train.AdamOptimizer(
                 learning_rate=self.learning_rate)
            self.train_step = self.optimizer.minimize(
                 self.loss, var_list=policy_params)
           ```

        10. update network parameters

            ```{python}
               self.update_op = tf.group(
                  [dst.assign(src) for src, dst in zip(policy_params, policy_old_params)])
            ```

        11. Log
        - Check if there was earlier training data

     5. Create the agents
        1. Let every agent play the game for a number of steps (horizon)
           1. Predict and value action given state
              1. Get state of agents
              2. Predict each action, perform the action
           2. Sample action from a Gaussian distribution
           3. Store state, action, reward
           4. Get new state
        2. Calculate last values (bootstrap values)
        3. Flatten arrays
        4. Train for some number of epochs
           1. Update old policy
           2. Evaluate model
              1. make_video, based on flag
              2. Fetch the current state
              3. predict the action
              4. Compute returns
              5. value error
              6. log values
           3. Save model
           4. Sample mini-batch randomly
           5. Optimize network
              1. Optimize the learning rate

---

## End Result

- After training for 280,000 episodes (equivalent to +36 hours) on GPU - NIVIDIA GeForce GTX 950M -

![GIF for agent after training for 36+ hours](docs/image/carracing-ppo.gif)
