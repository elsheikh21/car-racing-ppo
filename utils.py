from collections import deque
import numpy as np
import scipy.signal


class FrameStack():
    def __init__(self, initial_frame, stack_size=4, preprocess_fn=None):
        # Setup initial state
        self.frame_stack = deque(maxlen=stack_size)
        initial_frame = preprocess_fn(
            initial_frame) if preprocess_fn else initial_frame
        for _ in range(stack_size):
            self.frame_stack.append(initial_frame)
        self.state = np.stack(self.frame_stack, axis=-1)
        self.preprocess_fn = preprocess_fn

    def add_frame(self, frame):
        self.frame_stack.append(self.preprocess_fn(frame))
        self.state = np.stack(self.frame_stack, axis=-1)

    def get_state(self):
        return self.state


def discount(x, gamma):
    return scipy.signal.lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]


def compute_v_and_adv(rewards, values, bootstrapped_value, gamma, lam=1.0):
    rewards = np.array(rewards)
    values = np.array(list(values) + [bootstrapped_value])
    v = discount(np.array(list(rewards) + [bootstrapped_value]), gamma)[:-1]
    delta = rewards + gamma * values[1:] - values[:-1]
    adv = discount(delta, gamma * lam)
    return v, adv


def compute_returns(rewards, bootstrap_value, terminals, gamma):
    returns = []
    R = bootstrap_value
    for i in reversed(range(len(rewards))):
        R = rewards[i] + (1.0 - terminals[i]) * gamma * R
        returns.append(R)
    returns = reversed(returns)
    return np.array(list(returns))


def compute_gae(rewards, values, bootstrap_values, terminals, gamma, lam):
    values = np.vstack((values, [bootstrap_values]))

    # Compute delta
    deltas = []
    for i in reversed(range(len(rewards))):
        V = rewards[i] + (1.0 - terminals[i]) * gamma * values[i + 1]
        delta = V - values[i]
        deltas.append(delta)
    deltas = np.array(list(reversed(deltas)))

    # Compute gae
    A = deltas[-1, :]
    advantages = [A]
    for i in reversed(range(len(deltas) - 1)):
        A = deltas[i] + (1.0 - terminals[i]) * gamma * lam * A
        advantages.append(A)
    advantages = reversed(advantages)

    return np.array(list(advantages))
