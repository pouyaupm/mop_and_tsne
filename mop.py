import numpy as np
import pandas as pd
import os

from pymoo.problems import get_problem
from pymoo.optimize import minimize

# Import algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# Import operators
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Import performance indicators
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD

# For visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Create directories to save results and plots
results_dir = 'results'
plots_dir = 'plots'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)

# List of benchmark problems to test
benchmark_problems = [
    'zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6',
    'dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7',
    'wfg1', 'wfg2', 'wfg3', 'wfg4', 'wfg5', 'wfg6', 'wfg7', 'wfg8', 'wfg9'
]

# Define scaling factors and normalization methods to test
scaling_factor_sets = [
    [1, 1],
    [1, 1000],
    [1000, 1],
    [1, 0.001],
    [0.001, 1],
    [1, 1, 1],  # For problems with more than 2 objectives
    [1000, 1, 1],
    [1, 1000, 1],
    [1, 1, 1000]
]

normalization_methods = [None, 'min-max', 'z-score', 'dynamic']

# List of algorithms to test
algorithms = {
    'nsga2': NSGA2,
    'nsga3': NSGA3,
    'spea2': SPEA2,
    'moead': MOEAD,
    'rvea': RVEA
}

# Initialize performance records
performance_records = []

# Function to apply scaling and normalization
def scale_and_normalize(F, scaling_factors, normalization, gen=None, max_gen=None):
    # Apply scaling factors
    F_scaled = F * scaling_factors

    # Dynamic normalization adjustment based on generation
    if normalization == 'dynamic' and gen is not None and max_gen is not None:
        # Linearly adjust normalization extent based on generation
        alpha = gen / max_gen  # Normalization factor increases over generations
        # Estimate ideal and nadir points dynamically
        ideal_point = np.min(F_scaled, axis=0)
        nadir_point = np.max(F_scaled, axis=0)
        # Avoid division by zero
        range_F = nadir_point - ideal_point + 1e-9
        F_normalized = (F_scaled - ideal_point) / range_F
        # Apply dynamic adjustment
        F_normalized = alpha * F_normalized + (1 - alpha) * F_scaled
        return F_normalized
    elif normalization == 'min-max':
        # Estimate ideal and nadir points
        ideal_point = np.min(F_scaled, axis=0)
        nadir_point = np.max(F_scaled, axis=0)
        # Avoid division by zero
        range_F = nadir_point - ideal_point + 1e-9
        F_normalized = (F_scaled - ideal_point) / range_F
        return F_normalized
    elif normalization == 'z-score':
        F_mean = np.mean(F_scaled, axis=0)
        F_std = np.std(F_scaled, axis=0) + 1e-9
        F_normalized = (F_scaled - F_mean) / F_std
        return F_normalized
    else:
        return F_scaled

# Function to identify extreme points
def identify_extreme_points(F):
    # Identify extreme points along each objective
    extreme_points = []
    for i in range(F.shape[1]):
        idx = np.argmin(F[:, i])
        extreme_points.append(F[idx, :])
    return np.array(extreme_points)

# Function to compute performance indicators
def compute_performance_metrics(F, problem, n_obj):
    # Get Pareto front for performance evaluation
    if hasattr(problem, 'pareto_front'):
        pareto_front = problem.pareto_front()
        if pareto_front is None:
            pareto_front = F  # Use obtained solutions if Pareto front is not available
    else:
        pareto_front = F  # Use obtained solutions if Pareto front is not available

    # Compute Hypervolume
    # Define reference point (assuming maximization problems have been converted to minimization)
    ref_point = np.max(pareto_front, axis=0) + 1
    hv_indicator = HV(ref_point=ref_point)
    hv = hv_indicator(F)

    # Compute Inverted Generational Distance (IGD)
    igd_indicator = IGD(pareto_front)
    igd = igd_indicator(F)

    # Return performance metrics
    return hv, igd

# Function to filter nondominated solutions

def filter_nondominated(F):
    if NonDominatedSorting is not None:
        nds = NonDominatedSorting().do(F, only_non_dominated_front=True)
        return F[nds]
    else:
        # Use custom function
        n_points = F.shape[0]
        is_dominated = np.zeros(n_points, dtype=bool)
        for i in range(n_points):
            for j in range(n_points):
                if all(F[j] <= F[i]) and any(F[j] < F[i]):
                    is_dominated[i] = True
                    break
        return F[~is_dominated]
# Loop over benchmark problems
for problem_name in benchmark_problems:
    # Determine number of objectives
    if 'dtlz' in problem_name or 'wfg' in problem_name:
        n_obj = 3  # Adjust n_obj as needed
    else:
        # ZDT problems are bi-objective
        n_obj = 2

    # Calculate number of variables (n_var)
    if 'dtlz' in problem_name:
        k = 10  # Position-related parameters
        n_var = n_obj + k - 1
    elif 'wfg' in problem_name:
        k = 2 * (n_obj - 1)  # Position-related parameters
        l = 20  # Distance-related parameters
        n_var = k + l
    else:
        # For ZDT problems, n_var is typically 30
        n_var = 30

    # Get the problem instance
    if 'dtlz' in problem_name or 'wfg' in problem_name:
        # DTLZ and WFG problems require n_obj and n_var
        problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)
    else:
        # ZDT problems only require n_var
        problem = get_problem(problem_name, n_var=n_var)

    # Adjust scaling factors to match number of objectives
    for scaling_factors in scaling_factor_sets:
        # Skip scaling factors that don't match the number of objectives
        if len(scaling_factors) != n_obj:
            continue

        for normalization in normalization_methods:
            for alg_name, alg_class in algorithms.items():
                # Define the algorithm
                if alg_name == 'nsga2':
                    algorithm = alg_class(
                        pop_size=100,
                        sampling=FloatRandomSampling(),
                        crossover=SBX(prob=0.9, eta=15),
                        mutation=PM(eta=20),
                        eliminate_duplicates=True
                    )
                elif alg_name == 'nsga3':
                    ref_dirs = None
                    if n_obj > 2:
                        from pymoo.util.ref_dirs import get_reference_directions
                        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
                    algorithm = alg_class(
                        pop_size=100 if ref_dirs is None else len(ref_dirs),
                        ref_dirs=ref_dirs,
                        sampling=FloatRandomSampling(),
                        crossover=SBX(prob=0.9, eta=15),
                        mutation=PM(eta=20),
                        eliminate_duplicates=True
                    )
                elif alg_name == 'spea2':
                    algorithm = alg_class(
                        pop_size=100,
                        sampling=FloatRandomSampling(),
                        crossover=SBX(prob=0.9, eta=15),
                        mutation=PM(eta=20),
                        eliminate_duplicates=True
                    )
                elif alg_name == 'moead':
                    ref_dirs = None
                    if n_obj > 2:
                        from pymoo.util.ref_dirs import get_reference_directions
                        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
                    algorithm = alg_class(
                        ref_dirs=ref_dirs,
                        n_neighbors=15,
                        decomposition='tchebi',
                        prob_neighbor_mating=0.7,
                        sampling=FloatRandomSampling(),
                        crossover=SBX(prob=1.0, eta=20),
                        mutation=PM(eta=20),
                        eliminate_duplicates=True
                    )
                elif alg_name == 'rvea':
                    ref_dirs = None
                    if n_obj > 2:
                        from pymoo.util.ref_dirs import get_reference_directions
                        ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
                    algorithm = alg_class(
                        ref_dirs=ref_dirs,
                        sampling=FloatRandomSampling(),
                        crossover=SBX(prob=0.9, eta=15),
                        mutation=PM(eta=20),
                        eliminate_duplicates=True
                    )
                else:
                    continue  # Skip unknown algorithms

                # Maximum generations
                max_gen = 200

                # Store objectives over generations for dynamic normalization
                F_all_gens = []

                # Callback function to store F at each generation
                def store_F(algorithm):
                    F_all_gens.append(algorithm.pop.get('F'))

                # Run the optimization with callback
                res = minimize(problem,
                               algorithm,
                               ('n_gen', max_gen),
                               seed=1,
                               verbose=False,
                               callback=store_F)

                # Extract the results
                X = res.X  # Decision variables
                F = res.F  # Objective values

                # Apply scaling and normalization
                scaling_factors_array = np.array(scaling_factors)
                if normalization == 'dynamic':
                    # Apply dynamic normalization over generations
                    F_processed = []
                    for gen, F_gen in enumerate(F_all_gens, start=1):
                        F_gen_processed = scale_and_normalize(F_gen, scaling_factors_array, normalization, gen, max_gen)
                        F_processed.append(F_gen_processed)
                    # Flatten the list
                    F_processed = np.vstack(F_processed)
                else:
                    F_processed = scale_and_normalize(F, scaling_factors_array, normalization)

                # Filter nondominated solutions
                F_nondom = filter_nondominated(F_processed)

                # Identify extreme points (for analysis)
                extreme_points = identify_extreme_points(F_nondom)

                # Compute performance metrics
                hv, igd = compute_performance_metrics(F_nondom, problem, n_obj)

                # Record performance metrics
                performance_records.append({
                    'Problem': problem_name,
                    'Algorithm': alg_name,
                    'Scaling_Factors': scaling_factors,
                    'Normalization': normalization,
                    'Hypervolume': hv,
                    'IGD': igd
                })

                # Create a DataFrame to save the results
                df_X = pd.DataFrame(X, columns=[f'x{i+1}' for i in range(problem.n_var)])
                df_F = pd.DataFrame(F_nondom, columns=[f'f{i+1}' for i in range(n_obj)])
                df_extreme = pd.DataFrame(extreme_points, columns=[f'f{i+1}' for i in range(n_obj)])
                df_extreme['type'] = 'extreme'

                # Generate a filename based on problem, scaling factors, normalization, and algorithm
                scaling_str = '_'.join(map(str, scaling_factors))
                normalization_str = normalization if normalization is not None else 'none'
                filename = f'{problem_name}_{alg_name}_scaling_{scaling_str}_norm_{normalization_str}.csv'
                filepath = os.path.join(results_dir, filename)

                # Save the results to CSV
                df = pd.concat([df_X, df_F], axis=1)
                df.to_csv(filepath, index=False)

                print(f'Results saved to {filepath}')

                # Visualization
                if n_obj == 2:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(F_nondom[:, 0], F_nondom[:, 1], c='blue', label='Nondominated Solutions', alpha=0.7)
                    plt.scatter(extreme_points[:, 0], extreme_points[:, 1], color='red', label='Extreme Points', s=100, marker='X')
                    plt.xlabel('Objective 1')
                    plt.ylabel('Objective 2')
                    plt.title(f'{problem_name.upper()} | {alg_name.upper()} | Scaling: {scaling_str} | Norm: {normalization_str}')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plot_filename = f'{problem_name}_{alg_name}_{scaling_str}_{normalization_str}.png'
                    plt.savefig(os.path.join(plots_dir, plot_filename), dpi=300)
                    plt.close()
                elif n_obj == 3:
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(F_nondom[:, 0], F_nondom[:, 1], F_nondom[:, 2], c='blue', label='Nondominated Solutions', alpha=0.7)
                    ax.scatter(extreme_points[:, 0], extreme_points[:, 1], extreme_points[:, 2], color='red', label='Extreme Points', s=100, marker='X')
                    ax.set_xlabel('Objective 1')
                    ax.set_ylabel('Objective 2')
                    ax.set_zlabel('Objective 3')
                    plt.title(f'{problem_name.upper()} | {alg_name.upper()} | Scaling: {scaling_str} | Norm: {normalization_str}')
                    ax.legend()
                    plt.tight_layout()
                    plot_filename = f'{problem_name}_{alg_name}_{scaling_str}_{normalization_str}.png'
                    plt.savefig(os.path.join(plots_dir, plot_filename), dpi=300)
                    plt.close()

# Save performance metrics to CSV
performance_df = pd.DataFrame(performance_records)
performance_df.to_csv(os.path.join(results_dir, 'performance_metrics.csv'), index=False)
print(f'Performance metrics saved to {os.path.join(results_dir, "performance_metrics.csv")}')
