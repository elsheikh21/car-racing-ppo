# Policy Optimization Methods

## Policy Gradient methods

[Policy Gradient methods](policy-gradient-methods.md) are a type of [RL](https://github.com/elsheikh21/pacman-game/blob/master/docs/RL_Intro.md) techniques that rely upon **optimizing parametrized policies** w.r.t the long-term cumulative reward (expected return) by gradient descent.

Why?

They do not suffer from many of the problems that have been marrying traditional RL approaches such as:

- the lack of guarantees of a value function
- the intractability problem resulting from uncertain state information and the complexity arising from continuous states & actions.

> Systems that contain uncertain state information, are considered partially observable, thus are approached by Partially Observable Markov Decision Process [POMDP](partially-observable-markov-decision-process.md), and those systems most often results in excessive computational demands. Read about [MDPs](https://github.com/elsheikh21/pacman-game/blob/master/docs/Markov_Decision_Process.md).
>
> Continuous states and actions are not treated by most of the off-shelf RL approaches.

However, Policy Gradient methods does not suffer the same way

- State uncertainty results in performance degradation (without state estimator), however, no need to change the optimization techniques.
- Continuous states and actions can be dealt with exactly the same way as discrete ones, while, in addition, the learning performance is often increased. Convergence at least to a local minima is guaranteed.

One of the most important of PO advantages, that its policy representation can be chosen so that it is meaningful for the task and can incorporate domain knowledge, resulting in less parameters needed in the learning process than in value-function based approaches.

Policy Gradient methods are not the holly grail of RL approaches. They are by definition on-policy and they need to forget data very fast; in order not to avoid the introduction of bias to gradient estimator. Hence, the use of sampled data is not very.

Value-function based approaches are guaranteed to reach global maximum, while policy gradients only converge to local maximum, and there may be many maxima in discrete problems.

Policy gradient methods are often quite demanding to apply, mainly because one has to have considerable knowledge about the system one wants to control to make reasonable policy definitions. Finally, policy gradient methods always have an open parameter, the learning rate, which may decide over the order of magnitude of the speed of convergence, these have led to new approaches inspired by expectation-maximization.
