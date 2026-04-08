# Advanced Reinforcement Learning: Custom Grid Worlds & Policy Architectures

## Overview

Engineered and evaluated advanced Reinforcement Learning algorithms—specifically SARSA and N-Step Double Q-Learning—to autonomously navigate a custom probabilistic grid-world. This project addresses the exploration-exploitation dilemma and multi-step reward attribution, proving the agent's ability to efficiently convergence on optimal policies while mitigating maximization bias.

## Key Concepts

- **Reinforcement Learning**: Temporal difference learning and action-value evaluations.
- **Algorithms**: SARSA (On-Policy), Double Q-Learning (Off-Policy), and N-Step bootstrapping.
- **Hyperparameter Optimization**: Epsilon decay, discount factors (Gamma), and learning rates (Alpha).

## Architecture / Approach

1. **Custom Environment Engine**: Constructed `HappinessEnv` (a 5x5 grid map) utilizing the `Gymnasium` API, embedding complex reward/penalty mappings (+50 max goal, +20 alternate goal, negative penalties).
2. **RL Agents**:
   - **SARSA**: Iteratively updates expected returns adhering strictly to the exploratory policy.
   - **N-Step Double Q-Learning**: Resolves overestimation bias utilizing dual Q-tables (`Q_A` & `Q_B`) and utilizes multi-step trajectories (tuned from n=1 to 5) to accelerate reward propagation.
3. **Training & Evaluation**: Executed 1000-episode runs, analyzing Q-table convergence and tracking total reward variance under greedy policies.

## Dataset

- **Source**: Custom modeled `HappinessEnv` Gymnasium Environment.
- **State/Action Space**: 25 Discrete States (5x5 grid), 4 Discrete Actions (Up, Down, Left, Right).

## Implementation Details

- **Libraries**: `Gymnasium` (OpenAI Gym), `NumPy`, `Matplotlib`.
- **Design Choices**: Built custom Gym classes with rendering overrides via `matplotlib.image` to visualize specific state states (Happy/Sad nodes).
- **Optimization Strategy**: Employed rigorous grid search for finding optimal alpha, gamma, and epsilon-decay configurations.

## Results & Metrics

- Achieved **fastest convergence** utilizing the optimized N-Step Double Q-Learning approach with carefully tuned hyperparameters (Alpha=0.1, Gamma=0.95, Epsilon Decay=0.995).
- N-step bootstrapping significantly minimized the required steps to discover the sparse optimal (+50) reward compared to baseline 1-step logic.
- Double Q-Learning effectively suppressed the over-optimistic Q-value inflations occasionally observed in standard Q-Learning.

## Key Learnings

- **Tradeoffs in Horizon (N-Step)**: Observed that while higher N-steps accelerate learning by propagating rewards backward faster, exceptionally high N can increase variance.
- **Decoupled Value Estimation**: Tangibly demonstrated how Double Q-learning prevents sub-optimal loop-traps caused by maximization bias.

## Improvements / Future Work

- **Continuous Environments**: Migrate the Q-Learning engines into Deep Q-Networks (DQN) to handle continuous state spaces.
- **Stochastic Transitions**: Implement probabilistic transition dynamics (slippery grid) to test policy robustness.

## How to Run

1. Install dependencies: `pip install gymnasium numpy matplotlib`
2. Launch the desired notebook (e.g., `SARSA.ipynb` or `NStep DoubleQ Learning.ipynb`).
3. View the custom visualizations and agent progress directly via Matplotlib inline charts.
