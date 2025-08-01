import numpy as np
import cvxpy as cp

# Agent types: 0 (red), 1 (green)
# Goal types: 0 (landmark 0), 1 (landmark 1)

preference_matrix = np.array([
    [3.0, 1.0],  # Agent 0 (red)
    [1.0, 2.0]   # Agent 1 (green)
])

# Assign agent types
agent_types = [0, 1]
goal_types = [0, 1]
num_agents = 2
num_goals = 2

# Estimated distances visually:
distances = np.array([
    [0.5, 1.2],  # Agent 0
    [3.6, 4]  # Agent 1
])

# Compute utility[goal, agent] = preference - weighted distance
distance_weight = 1
utility = np.zeros((num_goals, num_agents))
for i in range(num_agents):
    for j in range(num_goals):
        pref = preference_matrix[agent_types[i], goal_types[j]]
        utility[j, i] = pref - distance_weight * distances[i, j]

budgets = np.ones(num_goals)  # one per goal

x = cp.Variable((num_goals, num_agents), nonneg=True)  # shape: goal x agent


# Add penalty to discourage assigning multiple agents to the same goal
penalty_weight = 0.1  # Adjust as needed

# Log utility term
log_utility = cp.sum([
    budgets[j] * cp.log(cp.sum(cp.multiply(utility[j], x[j])) + 1e-6)
    for j in range(num_goals)
])

# Penalty to discourage assigning more than one agent to the same goal
penalty = cp.sum_squares(cp.sum(x, axis=1) - 1)

objective = cp.Maximize(log_utility - penalty_weight * penalty)



constraints = [
    cp.sum(x[:, i]) <= 1  # each agent is assigned exactly once
    for i in range(num_agents)
] + [
    cp.sum(x[j, :]) <= 1  # each goal is assigned to exactly one agent
    for j in range(num_goals)
]

problem = cp.Problem(objective, constraints)
problem.solve()

x_val = x.value
x_onehot = np.zeros_like(x_val)
x_onehot[np.arange(num_goals), np.argmax(x_val, axis=1)] = 1

print("Assignment matrix (goal × agent):")
print(x_val)
print("One-hot assignment (goal × agent):")
print(x_onehot)