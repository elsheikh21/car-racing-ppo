# Proximal Policy Optimizer [PPO]

## Motivation

- The problem with TRPO is that it uses conjugate gradient to solve the policy update problem
- Same performance as TRPO but with first order optimization
- TRPO is not compatible with architecture that produces multiple outputs
- TRPO performs poorly when integrated with CNNs and RNNs

## Introduction

- We are working on policy updating for best rewards same as TRPO but without conjugate gradient
- it has 2 forms:

  - KL Penalty version

    ![ppo kl penalty](image/ppo-penalty-version.png)

  - Clipping objective function version

    - it is formed from two parts, the Importance sampling surrogate loss function
      ![Importance sampling loss function](image/importance-sampling-loss-fn.png)
    - and lower bound theory used in TRPO, forming lower -pessimistic- bound which can be optimized by SGD
      ![Clipped version to form lower bound](image/clipping-ppo.png)

    - PPO Algorithm

      - implement vanilla policy gradient and use the clipped surrogate objective loss function for many steps of Gradient Descent instead of just one (better sample efficiency). As we deduce more info from our data and doing update that achieves kindly a reliable amount of KL divergence between `π`<sub>`θ`<sub>`old`</sub></sub> and `π`<sub>`θ`</sub>

      ![PPO pseudo code](image/ppo-algorithm.png)

---

## WRAP UP

![Meme](image/policy-gradient-meme.png)
