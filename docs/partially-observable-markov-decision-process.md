# Partially Observable Markov Decision Process

POMDP are basically Markov Decision Process in terms of

- Finite number of discrete states.
- probabilistic transitions between states and controllable actions.
- Next state is determined only by the current state and current action.
- HOWEVER, we are unsure which state we are in, the current state emits observations

|     Markov Models      |      |                 Do you have control Over the state transitions?                  |                                                                                              |
| :--------------------: | :--: | :------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
|                        |      |                                       No?                                        |                                             Yes?                                             |  |
|     Are the states     | Yes? |                                  Markov Chains                                   | [MDP](https://github.com/elsheikh21/pacman-game/blob/master/docs/Markov_Decision_Process.md) |  |
| completely observable? | No?  | [HMM](https://github.com/elsheikh21/weather-inference-hmm/blob/master/readme.md) |                                            POMDP                                             |  |

| MDP                                 | POMDP                                        |
| ----------------------------------- | -------------------------------------------- |
| +Tractable to solve                 | +Treats all sources of uncertainty uniformly |
| +Relatively easy to specify         | +Allows for information gathering actions    |
| -Assumes perfect knowledge of state | -Hugely intractable to solve optimally       |

---

## Formalism O'clock

- POMDP model is made up of

  1. Finite set of states `s[1:n] ∈ S`
  2. Finite set of Actions `a[1:m] ∈ A`
  3. Probabilistic state-action transitions `P(s[i]| a, s[j])`
  4. Reward for each state/action pair `r(s, a)`
  5. Conditional Observation probabilities `P(o|s)`

- Belief state:

  1. probability distribution over world states `b(s) = p(s)`
  2. Action update rule `b'(s) = Sigma(p(s|a, s') * b(s')) over s'∈ S`
  3. Observation update rule `b'(s) = p(o|s) * b(s)/k`

---

## POMDP Solving Algorithms

1. Value Iteration

2. Policy Iteration

3. Witness algorithm, HSVI

4. Greedy solutions
