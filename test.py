import numpy as np
from matplotlib import pyplot as plt
from mealpy.swarm_based.ACOR import OriginalACOR


def fitness_function(params):
    return -20 * np.exp(-0.2 * np.sqrt((1 / 2) * (params[0] ** 2 + params[1] ** 2))) - np.exp(
            (1 / 2) * (np.cos(2 * np.pi * params[0]) + np.cos(2 * np.pi * params[1]))) + 20 + np.exp(1)


problem_dict1_Grid = {
    "fit_func": fitness_function,
    "lb": [-32.768, -32.768],
    "ub": [32.768, 32.768],
    "minmax": "min",
    "log_to": None,
}


def generate_best_value_plot(best_value, iter):
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(range(len(best_value)), best_value)
    plt.xlabel("epoch")
    plt.ylabel("best value")
    fig.savefig(f'best_value-{iter}.png')  # save the figure to file
    plt.close(fig)


epoch = 100
pop_size = 50
best = list()


# for sample_count, intent_factor, zeta, iter in [[25, 0.5, 1.0, 1], [15, 0.1, 4.0, 2], [100, 0.9, 0.5, 3]]:
#     model = OriginalACOR(epoch, pop_size, sample_count, intent_factor, zeta)
#     best.append(model.solve(problem_dict1_Grid))
#     generate_best_value_plot(model.history.list_global_best_fit, iter)


results = list()
for sample_count in range(2, 100):
    for intent_factor in np.arange(0.1, 1, 0.1):
        for zeta in np.arange(0.1, 5, 0.2):
            model = OriginalACOR(epoch, pop_size, sample_count, float(intent_factor), float(zeta))
            results.append([model.solve(problem_dict1_Grid), [sample_count, intent_factor, zeta]])
    print(f"{sample_count}/100")

sorted_data = sorted(results, key=lambda x: x[0][1])
print(sorted_data[0])

# 29, 0.1, 0.6
# model = OriginalACOR(epoch, pop_size, 29, float(0.1), float(0.6))
# print(model.solve(problem_dict1_Grid))
# generate_best_value_plot(model.history.list_global_best_fit, 4)

