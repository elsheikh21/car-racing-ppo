# Policy Gradient methods

## Introduction

Most of RL approaches have been action-value methods; they learned the values
of actions & then selected the action based on their estimated action-value
estimates.

Policy Gradient methods are methods that instead learn a parameterized
policy that can select actions without consulting a value function.

However, a value function may still be used to learn the policy parameter,
but not required for action selection.

θ policy parameter vector, where θ ∈ ℝ.

π(a |s, θ) = Pr{A[t] = a | S[t] = s, θ[t] = θ}
for the probability that action 'a' is taken at time 't'
given that environment is at state 's' at same 't' with param θ.

If this method learned a value function as well, then value's function
weight vector will be denoted by 'w' as in v_hat(s,w).

To learn the policy parameter based on a gradient of some scalar performance
measure J(θ) w.r.t the policy parameter. Thus, we seek maximizing the
performance. So their updates approximate gradient ascent in J

θ[t + 1] = θ[t] + α \* ∇J(θ[t]).

∇J(θ[t]): stochastic estimate whose expectation approximates the gradient of
the performance measure w.r.t its argument θ[t].

All methods that follow this general schema are called
`Policy Gradient Methods`, whether or not they learn an approximate
value function.

Methods that learn approximations for `policy & value functions` are known as `Actor-Critic methods`, where `actor` is a reference to the learned policy and `critic` is a reference to the learned value function.
