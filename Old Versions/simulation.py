# Full code to reproduce the plots for the polarization simulation

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Simulation parameters
np.random.seed(42)
num_users = 100
num_content = 100
time_steps = 10

# User and content distributions (randomly generated for simplicity)
user_distribution = [np.random.dirichlet(np.ones(num_users)) for _ in range(time_steps)]
content_distribution = [np.random.dirichlet(np.ones(num_content)) for _ in range(time_steps)]

# Engagement function (randomized for simulation)
engagement = [np.random.rand(num_users, num_content) for _ in range(time_steps)]

# Polarization scores for content (randomized for simulation)
polarization = [np.random.rand(num_content) for _ in range(time_steps)]

# Constraints
engagement_min = [50 for _ in range(time_steps)]  # Minimum engagement threshold
polarization_max = [0.3 for _ in range(time_steps)]  # Maximum polarization threshold

# Regularization parameter for temporal smoothing
gamma = 0.1

# Adjust weights for engagement and polarization over time
alpha_t = np.linspace(1, 0.5, time_steps)  # Engagement weight decreases
lambda_t = np.linspace(1, 2, time_steps)  # Polarization penalty increases

# Initialize a highly polarized transport plan for t = 0
initial_transport_plan = np.zeros((num_users, num_content))
polarized_indices = np.argsort(-polarization[0])[:num_users]
for i, idx in enumerate(polarized_indices):
    initial_transport_plan[i, idx] = 1.0

# Normalize the initial transport plan to match the distributions
initial_transport_plan /= initial_transport_plan.sum(axis=1, keepdims=True)

# Initialize storage for transport plans and polarization levels
transport_plans = [initial_transport_plan]
polarization_levels = [np.sum(initial_transport_plan * polarization[0])]

# Simulation over time steps
for t in range(1, time_steps):
    # Start with the previous transport plan for temporal smoothing
    prev_plan = transport_plans[-1].flatten()
    
    # Compute the cost matrix with updated weights
    c = (-alpha_t[t] * engagement[t] + lambda_t[t] * polarization[t][np.newaxis, :]).flatten()
    
    # Build the equality constraints for user and content distributions
    A_eq = np.zeros((num_users + num_content, num_users * num_content))
    for i in range(num_users):
        A_eq[i, i * num_content : (i + 1) * num_content] = 1
    for j in range(num_content):
        A_eq[num_users + j, j::num_content] = 1
    b_eq = np.hstack([user_distribution[t], content_distribution[t]])
    
    # Build the inequality constraint for polarization
    A_ineq = np.zeros((1, num_users * num_content))
    for i in range(num_users):
        A_ineq[0, i * num_content : (i + 1) * num_content] = polarization[t]
    b_ineq = [polarization_max[t]]
    
    # Combine all constraints
    A_combined = np.vstack([A_eq, A_ineq])
    b_combined = np.hstack([b_eq, b_ineq])
    
    # Bounds for the transport plan
    bounds = [(0, None) for _ in range(num_users * num_content)]
    
    # Solve the linear program
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    
    if not res.success:
        raise RuntimeError(f"Optimization failed at time step {t}.")
    
    # Reshape the solution to transport plan
    transport_plan = res.x.reshape((num_users, num_content))
    
    # Apply temporal smoothing explicitly
    transport_plan = (1 - gamma) * transport_plan + gamma * prev_plan.reshape(num_users, num_content)
    
    # Compute the polarization level for the current transport plan
    current_polarization = np.sum(transport_plan * polarization[t])
    polarization_levels.append(current_polarization)
    
    # Store the transport plan for the next iteration
    transport_plans.append(transport_plan)

# Plot polarization levels over time
plt.figure(figsize=(10, 6))
plt.plot(range(time_steps), polarization_levels, marker="o")
plt.title("Polarization Levels Over Time")
plt.xlabel("Time Step")
plt.ylabel("Total Polarization")
plt.axhline(y=polarization_max[0], color="r", linestyle="--", label=f"P_max = {polarization_max[0]}")
plt.legend()
plt.grid()
plt.show()

# Plot the final transport plan
plt.figure(figsize=(10, 6))
plt.imshow(transport_plans[-1], aspect="auto", cmap="viridis")
plt.colorbar(label="Transport Plan Probability")
plt.title("Final Transport Plan (Steady State)")
plt.xlabel("Content")
plt.ylabel("Users")
plt.show()
