# Fair-MARL: Fair Multi-Agent Reinforcement Learning

This repository implements a Fair Multi-Agent Reinforcement Learning (Fair-MARL) framework that performs preference-aware and fairness-constrained goal assignments in decentralized environments. The framework integrates Graph Neural Networks, local observations, and centralized or decentralized training paradigms.

## 📦 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/yourusername/fair-marl.git
cd fair-marl
pip install -r requirements.txt
```

## 🧬 Dependencies

- Python 3.9+
- PyTorch
- OpenAI Gym
- Multi-Agent Particle Environment (MPE)

## 🔐 Gurobi License (for central planner with budgeted optimization)

If using Gurobi-based assignment solvers (e.g., Eisenberg-Gale):

```bash
cp /Users/yaoliu/Desktop/Fair-MARL/gurobi.lic /Users/yaoliu/gurobi.lic
export GRB_LICENSE_FILE="/Users/yaoliu/Desktop/Fair-MARL/gurobi.lic"
```

## 🚀 Training

Training uses [Weights & Biases (WandB)](https://wandb.ai/) for logging and visualization. Please ensure you are logged in to WandB or set your API key.

> ⚠️ Note: Training is typically run in a **Jupyter Notebook** for experiment management and real-time tracking.

Example command:

```bash
python -u onpolicy/scripts/train_mpe.py \
  --project_name "test" \
  --env_name "GraphMPE" \
  --algorithm_name "rmappo" \
  --seed 2 \
  --experiment_name "test123" \
  --scenario_name "nav_fairassign_nofairrew_formation_graph" \
  --num_env_steps 1000000 \
  --num_agents 2 \
  --num_landmarks 2 \
  --num_obstacles 2
```

Alternative with fairness-based reward shaping:

```bash
python -u onpolicy/scripts/train_mpe.py \
  --project_name "test" \
  --env_name "GraphMPE" \
  --algorithm_name "rmappo" \
  --seed 2 \
  --experiment_name "test123" \
  --scenario_name "nav_fairassign_fairrew_formation_graph" \
  --num_env_steps 1000000 \
  --num_agents 2 \
  --num_landmarks 2 \
  --num_obstacles 2
```

## 🎥 Evaluation and Rendering

Rendering also works best in a **Jupyter Notebook** for visualization and GIF saving.

Example evaluation command:

```bash
python onpolicy/scripts/eval_mpe.py \
  --env_name "GraphMPE" \
  --scenario_name "nav_fairassign_nofairrew_formation_graph" \
  --algorithm_name "rmappo" \
  --model_dir "/Users/yaoliu/Desktop/Fair-MARL/onpolicy/results/GraphMPE/nav_fairassign_nofairrew_formation_graph/rmappo/test123/wandb/run-20250725_001808-2ur8kb5w/files" \
  --model_name "FA_FR" \
  --assignment_type fair \
  --use_fairness_reward \
  --num_agents 2 \
  --num_landmarks 2 \
  --num_obstacles 2 \
  --world_size 3 \
  --episode_length 500 \
  --render_episodes 5 \
  --save_gifs \
  --use_render
```

## 📜 Notes

- This project supports both fairness-agnostic and fairness-aware reward shaping.
- Agent-goal assignments can be learned (via RL) or computed via centralized optimization (e.g., Eisenberg-Gale).
- The default scenario used is `nav_fairassign_*_formation_graph`, but you may adapt to your own.

---

For questions, please contact the maintainer or file an issue.

