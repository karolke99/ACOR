import numpy as np
from mealpy.swarm_based.ACOR import OriginalACOR

def fitness_function(solution):
    return np.sum(solution**2)

problem_dict1 = {
    "fit_func": fitness_function,
    "lb": [-10, -15, -4, -2, -8],
    "ub": [10, 15, 12, 8, 20],
    "minmax": "min",
}

epoch = 1000
pop_size = 50
sample_count = 25
intent_factor = 0.5
zeta = 1.0
model = OriginalACOR(epoch, pop_size, sample_count, intent_factor, zeta)
best_position, best_fitness = model.solve(problem_dict1)
print(f"Solution: {best_position}, Fitness: {best_fitness}")
