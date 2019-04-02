import os
import argparse
import gym

import cv2
import numpy as np
from utils import FrameStack, compute_gae, compute_returns

from ppo import PPO
from vec_env.subproc_vec_env import SubprocVecEnv

env_name = "CarRacing-v0"


def parse_arg():
    parser = argparse.ArgumentParser(
        description="Trains an agent in a the CarRacing-v0 environment with proximal policy optimization")

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


def make_env():
    return gym.make(env_name)


def evaluate(model, test_env, discount_factor, frame_stack_size,
             make_video=False):
    total_reward = 0
    test_env.seed(0)
    initial_frame = test_env.reset()
    frame_stack = FrameStack(
        initial_frame, stack_size=frame_stack_size,
        preprocess_fn=preprocess_frame)
    rendered_frame = test_env.render(mode="rgb_array")
    values, rewards, dones = [], [], []
    if make_video:
        video_writer = cv2.VideoWriter(os.path.join(model.video_dir, "step{}.avi".format(model.step_idx)),
                                       cv2.VideoWriter_fourcc(*"MPEG"), 30,
                                       (rendered_frame.shape[1], rendered_frame.shape[0]))
    while True:
        # Predict action given state: π(a_t | s_t; θ)
        state = frame_stack.get_state()
        action, value = model.predict(
            np.expand_dims(state, axis=0), greedy=False)
        frame, reward, done, _ = test_env.step(action[0])
        rendered_frame = test_env.render(mode="rgb_array")
        total_reward += reward
        dones.append(done)
        values.append(value)
        rewards.append(reward)
        frame_stack.add_frame(frame)
        if make_video:
            video_writer.write(cv2.cvtColor(rendered_frame, cv2.COLOR_RGB2BGR))
        if done:
            break
    if make_video:
        video_writer.release()
    returns = compute_returns(np.transpose([rewards], [1, 0]), [
                              0], np.transpose([dones], [1, 0]), discount_factor)
    value_error = np.mean(np.square(np.array(values) - returns))
    return total_reward, value_error


def train(params, model_name, save_interval=1000, eval_interval=200,
          record_episodes=True, restart=False):
    try:
        # Create test env
        print("[INFO] Creating test environment")
        test_env = gym.make(env_name)

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

        # Training parameters
        def lr_scheduler(step_idx): return initial_lr * \
            0.85 ** (step_idx // 10000)

        # Environment constants
        frame_stack_size = 4
        input_shape = (84, 84, frame_stack_size)
        num_actions = test_env.action_space.shape[0]
        action_min = test_env.action_space.low
        action_max = test_env.action_space.high

        # Create model
        print("[INFO] Creating model")
        model = PPO(input_shape, num_actions, action_min, action_max,
                    epsilon=ppo_epsilon,
                    value_scale=value_scale, entropy_scale=entropy_scale,
                    model_name=model_name)

        print("[INFO] Creating environments")
        envs = SubprocVecEnv([make_env for _ in range(num_envs)])

        initial_frames = envs.reset()
        envs.get_images()
        frame_stacks = [FrameStack(initial_frames[i], stack_size=frame_stack_size,
                                   preprocess_fn=preprocess_frame) for i in range(num_envs)]

        print("[INFO] Training loop")
        while True:
            # While there are running environments
            states, taken_actions, values, rewards, dones = [], [], [], [], []

            # Simulate game for some number of steps
            for _ in range(horizon):
                # Predict and value action given state
                # π(a_t | s_t; θ_old)
                states_t = [frame_stacks[i].get_state()
                            for i in range(num_envs)]
                actions_t, values_t = model.predict(states_t)

                # Sample action from a Gaussian distribution
                envs.step_async(actions_t)
                frames, rewards_t, dones_t, _ = envs.step_wait()
                envs.get_images()  # render

                # Store state, action and reward
                # [T, N, 84, 84, 4]
                states.append(states_t)
                taken_actions.append(actions_t)              # [T, N, 3]
                values.append(np.squeeze(values_t, axis=-1))  # [T, N]
                rewards.append(rewards_t)                    # [T, N]
                dones.append(dones_t)                        # [T, N]

                # Get new state
                for i in range(num_envs):
                    # Reset environment's frame stack if done
                    if dones_t[i]:
                        for _ in range(frame_stack_size):
                            frame_stacks[i].add_frame(frames[i])
                    else:
                        frame_stacks[i].add_frame(frames[i])

            # Calculate last values (bootstrap values)
            states_last = [frame_stacks[i].get_state()
                           for i in range(num_envs)]
            last_values = np.squeeze(model.predict(
                states_last)[1], axis=-1)  # [N]

            advantages = compute_gae(
                rewards, values, last_values, dones, discount_factor, gae_lambda)
            advantages = (advantages - advantages.mean()) / \
                (advantages.std() + 1e-8)  # Move down one line?
            returns = advantages + values
            # Flatten arrays
            states = np.array(states).reshape(
                (-1, *input_shape))       # [T x N, 84, 84, 4]
            taken_actions = np.array(taken_actions).reshape(
                (-1, num_actions))  # [T x N, 3]
            # [T x N]
            returns = returns.flatten()
            # [T X N]
            advantages = advantages.flatten()

            T = len(rewards)
            N = num_envs
            assert states.shape == (
                T * N, input_shape[0], input_shape[1], frame_stack_size)
            assert taken_actions.shape == (T * N, num_actions)
            assert returns.shape == (T * N,)
            assert advantages.shape == (T * N,)

            # Train for some number of epochs
            model.update_old_policy()  # θ_old <- θ
            for _ in range(num_epochs):
                num_samples = len(states)
                indices = np.arange(num_samples)
                np.random.shuffle(indices)
                for i in range(int(np.ceil(num_samples / batch_size))):
                    # Evaluate model
                    if model.step_idx % eval_interval == 0:
                        print("[INFO] Running evaluation...")
                        avg_reward, value_error = evaluate(
                            model, test_env, discount_factor, frame_stack_size, make_video=True)
                        model.write_to_summary("eval_avg_reward", avg_reward)
                        model.write_to_summary("eval_value_error", value_error)

                    # Save model
                    if model.step_idx % save_interval == 0:
                        model.save()

                    # Sample mini-batch randomly
                    begin = i * batch_size
                    end = begin + batch_size
                    if end > num_samples:
                        end = None
                    mb_idx = indices[begin:end]

                    # Optimize network
                    model.train(states[mb_idx], taken_actions[mb_idx],
                                returns[mb_idx], advantages[mb_idx])
    except KeyboardInterrupt:
        model.save()


if __name__ == "__main__":
    # Silence the logs of TF
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    # Read the params from the cmd prmpt
    params = parse_arg()

    # Remove non-hyperparameters
    model_name = params["model_name"]
    del params["model_name"]
    save_interval = params["save_interval"]
    del params["save_interval"]
    eval_interval = params["eval_interval"]
    del params["eval_interval"]
    record_episodes = params["record_episodes"]
    del params["record_episodes"]
    restart = params["restart"]
    del params["restart"]

    train(params, model_name, save_interval=save_interval,
          eval_interval=eval_interval, record_episodes=record_episodes,
          restart=restart)

    #    - A3C performs better on `Atari` and provide really good results for
    #       continuous control compared to DQN in terms of how fast it converges
    #       this is due to little bit built in of exploration over
    #       multiple machines.
    #    - GAE params are
    #      1. Discounting factor: `γ`
    #      2. Exponential decay: `λ`
    #         - brings a little bit of Temporal difference and reduces variance,
    #           but when we rely on TD a lot, model gets biased.
