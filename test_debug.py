import pickle
import os
import glob

# Find latest results dir
dirs = sorted(glob.glob("results/development_*"))
if not dirs:
    print("No results found")
    exit()

latest_dir = dirs[-1]
agg_path = os.path.join(latest_dir, "aggregation_history.pkl")

with open(agg_path, "rb") as f:
    history = pickle.load(f)

last_step = history[-1]
root_node = max(last_step.keys())
pareto = last_step[root_node]

print(f"Total solutions in root pareto front: {len(pareto)}")
for i, ind in enumerate(pareto):
    fit = ind['fitness']
    sol = ind['solution']
    print(f"Sol {i}: Fit={fit}, shape={sol.shape}, sum={sol.sum()}")
