import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pymoo.problems import get_problem
from pymoo.optimize import minimize

# Import algorithms
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.rvea import RVEA

# Import operators
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

# Import performance indicators
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.decomposition.tchebicheff import Tchebicheff

# Handle NonDominatedSorting import
try:
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
except ImportError:
    # Fallback to custom function if import fails
    NonDominatedSorting = None

# Create directories to save results and plots
results_dir = 'results'
plots_dir = 'plots'
os.makedirs(results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Define the number of objectives to test
objective_counts = [2, 3, 5, 7]

# Define benchmark problems for each number of objectives
benchmark_problems_dict = {
    2: ['zdt1', 'zdt2', 'zdt3', 'zdt4', 'zdt6'],
    3: ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'],
    5: ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7'],
    7: ['dtlz1', 'dtlz2', 'dtlz3', 'dtlz4', 'dtlz5', 'dtlz6', 'dtlz7']
    # Note: You can add WFG problems here if they support 5 and 7 objectives
}

# Define scaling factors and normalization methods to test
scaling_factor_sets = {
    2: [
        [1, 1],
        [1, 1000],
        [1000, 1],
        [1, 0.001],
        [0.001, 1]
    ],
    3: [
        [1, 1, 1],
        [1000, 1, 1],
        [1, 1000, 1],
        [1, 1, 1000],
        [1000, 1000, 1],
        [1, 1000, 1000],
        [1000, 1, 1000]
    ],
    5: [
        [1, 1, 1, 1, 1],
        [1000, 1, 1, 1, 1],
        [1, 1000, 1, 1, 1],
        [1, 1, 1000, 1, 1],
        [1, 1, 1, 1000, 1],
        [1, 1, 1, 1, 1000],
        [1000, 1000, 1, 1, 1],
        [1, 1000, 1000, 1, 1],
        [1, 1, 1000, 1000, 1],
        [1, 1, 1, 1000, 1000],
        [1000, 1, 1000, 1, 1],
        [1000, 1, 1, 1000, 1],
        [1000, 1, 1, 1, 1000],
        [1, 1000, 1, 1000, 1],
        [1, 1000, 1, 1, 1000],
        [1, 1, 1000, 1, 1000]
    ],
    7: [
        [1, 1, 1, 1, 1, 1, 1],
        [1000, 1, 1, 1, 1, 1, 1],
        [1, 1000, 1, 1, 1, 1, 1],
        [1, 1, 1000, 1, 1, 1, 1],
        [1, 1, 1, 1000, 1, 1, 1],
        [1, 1, 1, 1, 1000, 1, 1],
        [1, 1, 1, 1, 1, 1000, 1],
        [1, 1, 1, 1, 1, 1, 1000],
        [1000, 1000, 1, 1, 1, 1, 1],
        [1000, 1, 1000, 1, 1, 1, 1],
        [1000, 1, 1, 1000, 1, 1, 1],
        [1000, 1, 1, 1, 1000, 1, 1],
        [1000, 1, 1, 1, 1, 1000, 1],
        [1000, 1, 1, 1, 1, 1, 1000],
        [1, 1000, 1000, 1, 1, 1, 1],
        [1, 1000, 1, 1000, 1, 1, 1],
        [1, 1000, 1, 1, 1000, 1, 1],
        [1, 1000, 1, 1, 1, 1000, 1],
        [1, 1000, 1, 1, 1, 1, 1000],
        [1, 1, 1000, 1000, 1, 1, 1],
        [1, 1, 1000, 1, 1000, 1, 1],
        [1, 1, 1000, 1, 1, 1000, 1],
        [1, 1, 1000, 1, 1, 1, 1000],
        [1, 1, 1, 1000, 1000, 1, 1],
        [1, 1, 1, 1000, 1, 1000, 1],
        [1, 1, 1, 1000, 1, 1, 1000],
        [1, 1, 1, 1, 1000, 1000, 1],
        [1, 1, 1, 1, 1000, 1, 1000],
        [1, 1, 1, 1, 1, 1000, 1000]
    ]
}

# Define normalization methods to test
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
    try:
        pareto_front = problem.pareto_front()
    except AttributeError:
        pareto_front = None

    if pareto_front is None:
        pareto_front = F  # Use obtained solutions if Pareto front is not available

    # Compute Hypervolume
    # Define reference point slightly worse than the worst point in F
    ref_point = np.max(F, axis=0) + 1
    hv_indicator = HV(ref_point=ref_point)
    hv = hv_indicator(F)

    # Compute Inverted Generational Distance (IGD)
    if pareto_front is not None and pareto_front.shape[0] > 0:
        igd_indicator = IGD(pareto_front)
        igd = igd_indicator(F)
    else:
        igd = np.nan  # Not available

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

# Main loop over number of objectives
for n_obj in objective_counts:
    print(f'\n=== Number of Objectives: {n_obj} ===')
    problems = benchmark_problems_dict.get(n_obj, [])
    
    if not problems:
        print(f'No benchmark problems defined for {n_obj} objectives. Skipping.')
        continue
    
    for problem_name in problems:
        print(f'\nProcessing Problem: {problem_name}')
        
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
        try:
            if 'dtlz' in problem_name or 'wfg' in problem_name:
                problem = get_problem(problem_name, n_var=n_var, n_obj=n_obj)
            else:
                problem = get_problem(problem_name, n_var=n_var)
        except Exception as e:
            print(f'Error initializing problem {problem_name}: {e}')
            continue

        # Get scaling factors for the current number of objectives
        scaling_factors_list = scaling_factor_sets.get(n_obj, [])
        
        for scaling_factors in scaling_factors_list:
            # Ensure scaling_factors matches the number of objectives
            if len(scaling_factors) != n_obj:
                continue  # Skip invalid scaling factors

            for normalization in normalization_methods:
                for alg_name, alg_class in algorithms.items():
                    print(f'\nRunning Algorithm: {alg_name.upper()} | Scaling: {scaling_factors} | Normalization: {normalization}')
                    
                    # Initialize the algorithm with appropriate parameters
                    try:
                        if alg_name == 'nsga2':
                            # NSGA-II supports eliminate_duplicates
                            algorithm = alg_class(
                                pop_size=100,
                                sampling=FloatRandomSampling(),
                                crossover=SBX(prob=0.9, eta=15),
                                mutation=PM(eta=20),
                                eliminate_duplicates=True
                            )
                        elif alg_name == 'nsga3':
                            from pymoo.util.ref_dirs import get_reference_directions

                            # Set n_partitions based on the number of objectives
                            if n_obj == 2:
                                n_partitions = 100
                            elif n_obj == 3:
                                n_partitions = 12
                            elif n_obj == 5:
                                n_partitions = 6
                            elif n_obj == 7:
                                n_partitions = 4
                            else:
                                n_partitions = 6  # Default value

                            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)

                            algorithm = alg_class(
                                pop_size=len(ref_dirs),
                                ref_dirs=ref_dirs,
                                sampling=FloatRandomSampling(),
                                crossover=SBX(prob=0.9, eta=15),
                                mutation=PM(eta=20),
                                eliminate_duplicates=True  # NSGA-III supports eliminate_duplicates
                            )
                        elif alg_name == 'spea2':
                            # SPEA2 does NOT support eliminate_duplicates
                            algorithm = alg_class(
                                pop_size=100,
                                sampling=FloatRandomSampling(),
                                crossover=SBX(prob=0.9, eta=15),
                                mutation=PM(eta=20)
                                # eliminate_duplicates parameter is omitted
                            )
                        elif alg_name == 'moead':
                            from pymoo.util.ref_dirs import get_reference_directions

                            if n_obj == 2:
                                n_partitions = 100
                            elif n_obj == 3:
                                n_partitions = 12
                            elif n_obj == 5:
                                n_partitions = 6
                            elif n_obj == 7:
                                n_partitions = 4
                            else:
                                n_partitions = 6  # Default value

                            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)

                            algorithm = alg_class(
                                ref_dirs=ref_dirs,
                                n_neighbors=15,
                                decomposition=Tchebicheff(),  # Corrected: Use instance of Tchebicheff
                                prob_neighbor_mating=0.7,
                                sampling=FloatRandomSampling(),
                                crossover=SBX(prob=1.0, eta=20),
                                mutation=PM(eta=20)
                                # eliminate_duplicates parameter is omitted
                            )
                        elif alg_name == 'rvea':
                            from pymoo.util.ref_dirs import get_reference_directions

                            if n_obj == 2:
                                n_partitions = 100
                            elif n_obj == 3:
                                n_partitions = 12
                            elif n_obj == 5:
                                n_partitions = 6
                            elif n_obj == 7:
                                n_partitions = 4
                            else:
                                n_partitions = 6  # Default value

                            ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=n_partitions)

                            algorithm = alg_class(
                                ref_dirs=ref_dirs,
                                sampling=FloatRandomSampling(),
                                crossover=SBX(prob=0.9, eta=15),
                                mutation=PM(eta=20)
                                # eliminate_duplicates parameter is omitted
                            )
                        else:
                            print(f'Unknown algorithm: {alg_name}')
                            continue  # Skip unknown algorithms
                    except Exception as e:
                        print(f'Error initializing algorithm {alg_name.upper()}: {e}')
                        continue

                    # Maximum generations
                    max_gen = 200

                    # Store objectives over generations for dynamic normalization
                    F_all_gens = []

                    # Callback function to store F at each generation
                    def store_F(algorithm_instance):
                        F_all_gens.append(algorithm_instance.pop.get('F'))

                    # Run the optimization with callback
                    try:
                        res = minimize(problem,
                                       algorithm,
                                       ('n_gen', max_gen),
                                       seed=1,
                                       verbose=False,
                                       callback=store_F)
                    except Exception as e:
                        print(f'Error during optimization with {alg_name.upper()}: {e}')
                        continue

                    # Extract the final population's decision variables and objective values
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
                        'Number_of_Objectives': n_obj,
                        'Algorithm': alg_name.upper(),
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
                    filename = f'{problem_name}_{alg_name.upper()}_scaling_{scaling_str}_norm_{normalization_str}.csv'
                    filepath = os.path.join(results_dir, filename)

                    # Save the results to CSV
                    try:
                        df = pd.concat([df_X, df_F], axis=1)
                        df.to_csv(filepath, index=False)
                        print(f'Results saved to {filepath}')
                    except Exception as e:
                        print(f'Error saving results to {filepath}: {e}')

                    # Visualization
                    try:
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
                            plot_filename = f'{problem_name}_{alg_name.upper()}_{scaling_str}_{normalization_str}.png'
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
                            plot_filename = f'{problem_name}_{alg_name.upper()}_{scaling_str}_{normalization_str}.png'
                            plt.savefig(os.path.join(plots_dir, plot_filename), dpi=300)
                            plt.close()
                        elif n_obj in [5, 7]:
                            # For higher objectives, visualize pairwise plots for the first three objectives
                            if F_nondom.shape[1] >= 3:
                                plt.figure(figsize=(18, 12))
                                objective_indices = range(3)  # First three objectives
                                labels = [f'Objective {i+1}' for i in objective_indices]
                                subplot_idx = 1
                                for i in objective_indices:
                                    for j in objective_indices:
                                        if i < j:
                                            plt.subplot(3, 3, subplot_idx)
                                            plt.scatter(F_nondom[:, i], F_nondom[:, j], c='blue', alpha=0.7, label='Nondominated Solutions')
                                            plt.scatter(extreme_points[:, i], extreme_points[:, j], color='red', s=100, marker='X', label='Extreme Points')
                                            plt.xlabel(labels[i])
                                            plt.ylabel(labels[j])
                                            plt.title(f'{labels[i]} vs {labels[j]}')
                                            plt.legend()
                                            subplot_idx += 1
                                plt.suptitle(f'{problem_name.upper()} | {alg_name.upper()} | Scaling: {scaling_str} | Norm: {normalization_str}', fontsize=16)
                                plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                                plot_filename = f'{problem_name}_{alg_name.upper()}_{scaling_str}_{normalization_str}_pairwise.png'
                                plt.savefig(os.path.join(plots_dir, plot_filename), dpi=300)
                                plt.close()
                            else:
                                print('Not enough objectives for pairwise plotting.')
                        else:
                            print(f'Visualization for {n_obj} objectives is not implemented.')
                    except Exception as e:
                        print(f'Error during visualization: {e}')

# Save performance metrics to CSV
try:
    performance_df = pd.DataFrame(performance_records)
    performance_metrics_filepath = os.path.join(results_dir, 'performance_metrics.csv')
    performance_df.to_csv(performance_metrics_filepath, index=False)
    print(f'\nPerformance metrics saved to {performance_metrics_filepath}')
except Exception as e:
    print(f'Error saving performance metrics: {e}')
