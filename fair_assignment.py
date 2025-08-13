import numpy as np
import cvxpy as cp
import pyomo.environ as pe
import scipy.spatial.distance as dist

def build_base_model(costs):
    assert np.ndim(costs) == 2
    n, nj = costs.shape
    m = pe.ConcreteModel()
    m.ns = pe.RangeSet(0,n-1)
    m.ms = pe.RangeSet(0,nj-1)
    m.x = pe.Var(m.ns, m.ms, domain=pe.Binary)
    m.coverage = pe.Constraint(m.ms, rule=lambda m,j: sum(m.x[i,j] for i in m.ns) == 1)   # eacg task must be performed by exactly one agent
    m.assignment = pe.Constraint(m.ns, rule=lambda m,i: sum(m.x[i,j] for j in m.ms) == 1) # each agent must perform exactly one task
    return m

def solve_fair_assignment(costs):
    '''Solves lexifair assignment.

    costs - matrix of costs where entry [i,j] is the cost for agent i to perform task j
    '''
    import logging
    n, nj = costs.shape
    cost_helper = np.copy(costs)
    m = build_base_model(costs)
    m.z = pe.Var()
    m.aux = pe.Constraint(m.ns, m.ms, rule=lambda m,i,j: cost_helper[i,j]*m.x[i,j] <= m.z)  # maximum cost of each agent
    m.assigned = pe.ConstraintList()
    m.obj = pe.Objective(expr=m.z, sense=pe.minimize)

    solver = pe.SolverFactory('gurobi_persistent')
    solver.set_instance(m)
    objs = []

    for iter in range(n):
        results = solver.solve(options={'TimeLimit': 60}, save_results=True)
        # Logging solver status and termination condition
        print(f"[solve_fair_assignment] Iter {iter+1}/{n}:")
        print("  Costs matrix:\n", costs)
        print("  Solver status:", results.solver.status)
        print("  Termination condition:", results.solver.termination_condition)

        # Check for infeasibility or unboundedness
        if (results.solver.termination_condition == pe.TerminationCondition.infeasible) or \
           (results.solver.termination_condition == pe.TerminationCondition.infeasibleOrUnbounded) or \
           (results.solver.termination_condition == pe.TerminationCondition.unbounded):
            raise RuntimeError(
                f"Fair assignment infeasible or unbounded! "
                f"Costs:\n{costs}\n"
                f"Status: {results.solver.status}, "
                f"Termination: {results.solver.termination_condition}"
            )

        x_val = pe.value(m.x[:,:])
        if x_val is None:
            raise RuntimeError(
                f"Solver did not return a solution (NoneType). Costs:\n{costs}\n"
                f"Status: {results.solver.status}, "
                f"Termination: {results.solver.termination_condition}"
            )

        x = np.array(x_val).reshape((n, nj)).astype(int)
        obj = pe.value(m.obj)

        objs.append(obj)
        r,c = np.unravel_index(np.argmin(np.abs(costs - obj)), (n,nj)) # find the agent/task that incurred max cost

        # update costs + constraints
        cost_helper[r,c] = 0
        for constr in m.aux.values():
            solver.remove_constraint(constr)
        m.del_component(m.aux)
        
        m.aux = pe.Constraint(m.ns, m.ms, rule=lambda m,i,j: cost_helper[i,j]*m.x[i,j] <= m.z)
        for constr in m.aux.values():
            solver.add_constraint(constr)
        for j in m.ms:
            m.assigned.add(expr=m.x[r,j] == x[r,j])
            solver.add_constraint(m.assigned[iter*nj + j + 1])

    objs = np.sort(np.sum(costs*x, axis=1))[::-1]
    return x, objs

def solve_eg_assignment(preference, cost, agent_types, goal_types, budgets=None, distance_weight=1.0, verbose=False):
    
    n_agents, n_goals = cost.shape
    '''
    Solves the Eisenberg-Gale (EG) goal assignment problem using cvxpy.

    Args:
        preference: (agent_types x goal_types) preference matrix, e.g., preference[a_type, g_type]
        cost: (n_agents x n_goals) matrix, e.g., distance or cost between each agent and goal
        agent_types: (n_agents,) array-like, agent type for each agent (integer index)
        goal_types: (n_goals,) array-like, goal type for each goal (integer index)
        budgets: (n_agents,) array-like, optional, budget for each agent (default: all ones)
        distance_weight: float, penalty weight for cost
        verbose: print assignment & utilities

    Returns:
        x: assignment matrix (n_agents x n_goals), binary
        utilities: array of each agent's achieved utility

        
    '''

    preference = preference.T
    utility = np.zeros((n_agents, n_goals))
    for j in range(n_goals):
        for i in range(n_agents):
            agt_type = agent_types[i]
            g_type = goal_types[j]
            if agt_type is None or g_type is None:
                raise ValueError(f"agent_types[{i}] or goal_types[{j}] is None. agent_types: {agent_types}, goal_types: {goal_types}")
            pref_val = preference[int(g_type), int(agt_type)]
            if np.isscalar(pref_val):
                is_zero = (pref_val == 0)
            else:
                is_zero = np.all(pref_val == 0)
            if is_zero:
                utility[j, i] = 0
            else:
                utility[j, i] = pref_val - distance_weight * cost[i, j]
        if budgets is None:
            budgets = np.ones(n_agents)

    x = cp.Variable((n_goals, n_agents), nonneg=True)  # match original shape (m x n)

    penalty = cp.sum_squares(cp.sum(x, axis=0) - 1)  # Encourage each agent to be fully used by exactly one buyer

    objective = cp.Maximize(
        cp.sum([
            budgets[goal_types[j]] * cp.log(cp.sum(cp.multiply(utility[j], x[j])) + 1e-6)
            for j in range(n_goals)
        ])-0.8*penalty
    )

    constraints = [cp.sum(x[:, t]) == 1 for t in range(n_agents)]  # agent caps
    constraints += [cp.sum(x[j, :]) == 1 for j in range(n_goals)]  # goal caps
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    if result is None or prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
        print(f"Social problem solver failed. Status: {prob.status}")
        return

    if x.value is None:
        raise RuntimeError("CVXPY solver failed. x.value is None.")

    x_val = np.asarray(x.value)
    print("x_val shape:", x_val)

    # Safe reshape if needed
    if x_val.ndim == 1:
        try:
            x_val = x_val.reshape((n_agents, n_goals))
            print("Reshaped x_val to:", x_val.shape)
        except Exception as e:
            raise RuntimeError(f"Failed to reshape x_val: {e}, original shape: {x_val.shape}")

    x_onehot = np.zeros_like(x_val)
    x_onehot[np.arange(n_agents), np.argmax(x_val, axis=1)] = 1
    print("one_hot shape:", x_onehot)

    lambda_t = np.array([constraints[t].dual_value for t in range(n_agents)])
    print("Dual variable values (prices):", lambda_t)
    x_onehot = x_onehot.T  # Transpose to match (n_goals, n_agents) shape

    return x_onehot, lambda_t

if __name__=='__main__':
    # n = 10
    # rng = np.random.default_rng(seed=667143)
    # goals = rng.random((n,2))
    # agents = rng.random((n,2))

    goals = np.array([[0.,-0.5],[0.45,-0.5],[0.9,-0.5]])
    agents = np.array([[-0.9,-0.9],[-0.9,0.],[-0.9,0.9]])
    print(goals.shape)
    # goals = # FILL ME (n-by-2 np.ndarray)
    # agents = # FILL ME (n-by-2 np.ndarray)

    costs = dist.cdist(agents, goals)
    x, objs = solve_fair_assignment(costs)
    # print(x)
    # print(objs)
    