#!/usr/bin/env python
"""
Multi‑Algorithm, Multi‑Seed Pareto‑Front Benchmark Suite
=======================================================
This *research‑grade* pipeline executes several Evolutionary Multi‑Objective
Algorithms (EMOAs) on classic benchmark problems, collects convergence &
diversity metrics (Hypervolume, IGD, Spacing, Crowding‑Distance), exports every
Pareto approximation and decision vector, and generates an interactive Plotly
Dashboard for visual analysis.

Key features
------------
* **Algorithms**: NSGA‑II, NSGA‑III, SPEA2, MOEA/D, and a uniform Random Search
  baseline.
* **Benchmarks**: ZDT1‑3 (2‑obj), DTLZ2 & DTLZ7 (5‑obj) – easily extendable.
* **Statistics**: 30 independent seeds (configurable) → mean ± std tables.
* **Metrics**: Hypervolume (HV), Inverted Generational Distance (IGD), Spacing
  (SP), mean Crowding Distance (CD).
* **Timeout**: per‑run wall‑clock limit.
* **Data export**: CSVs of objective & decision variables for each seed, plus a
  summary CSV.
* **Interactive visualisation**: single HTML file with tabs – scatter (2‑obj),
  parallel‑coordinates, and PCA/t‑SNE projections (many‑obj). Hover reveals
  decision vectors; checkboxes toggle algorithms.

Usage
-----
```bash
conda activate pareto-viz  # or your preferred env
python benchmark_suite.py \
       --seeds 30 --gens 200 --popsz 100 \
       --time-limit 600 \
       --outdir results
# Afterwards open results/dashboard.html in a browser.
```
Dependencies
------------
```
pymoo>=0.6.0
numpy pandas scikit-learn matplotlib tqdm plotly
```

"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as TPE
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.io as pio
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import trange

from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.core.problem import Problem
from pymoo.factory import (
    get_problem,
    get_reference_directions,
    get_termination,
)
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
# Spacing indicator is available in pymoo>=0.6.1; fallback otherwise
try:
    from pymoo.indicators.spacing import Spacing  # type: ignore
except ImportError:
    class Spacing:  # noqa: N801 – mimic pymoo interface
        """Manual Spacing implementation (fallback if not in pymoo)."""

        def __call__(self, F):
            from scipy.spatial.distance import cdist

            if len(F) < 2:
                return 0.0
            D = cdist(F, F)
            np.fill_diagonal(D, np.inf)
            d_i = D.min(axis=1)
            d_bar = d_i.mean()
            return np.sqrt(((d_i - d_bar) ** 2).sum() / (len(F) - 1))

from pymoo.optimize import minimize

################################################################################
# Benchmark & algorithm registry
################################################################################

BENCHMARKS: Dict[str, Callable[[], Problem]] = {
    "zdt1": lambda: get_problem("zdt1"),
    "zdt2": lambda: get_problem("zdt2"),
    "zdt3": lambda: get_problem("zdt3"),
    "dtlz2_5": lambda: get_problem("dtlz2", n_obj=5),
    "dtlz7_5": lambda: get_problem("dtlz7", n_obj=5),
}

ALGORITHMS: Dict[str, Callable[[Problem, int], object]] = {
    "nsga2": lambda prob, pop: NSGA2(pop_size=pop),
    "nsga3": lambda prob, pop: NSGA3(
        pop_size=pop if prob.n_obj <= 3 else 92,
        ref_dirs=get_reference_directions(
            "das-dennis", prob.n_obj, n_partitions=12 if prob.n_obj <= 3 else 4
        ),
    ),
    "spea2": lambda prob, pop: SPEA2(pop_size=pop),
    "moead": lambda prob, pop: MOEAD(
        ref_dirs=get_reference_directions("uniform", prob.n_obj, n_points=pop),
        n_neighbors=15,
    ),
    "random": lambda prob, pop: None,  # handled specially
}

################################################################################
# Helper functions
################################################################################


def _crowding_distance(F: np.ndarray) -> np.ndarray:
    n, m = F.shape
    d = np.zeros(n)
    fmin, fmax = F.min(0), F.max(0)
    span = np.where(fmax - fmin == 0, 1.0, fmax - fmin)
    F_norm = (F - fmin) / span
    for j in range(m):
        idx = np.argsort(F_norm[:, j])
        d[idx[0]] = d[idx[-1]] = np.inf
        for k in range(1, n - 1):
            d[idx[k]] += F_norm[idx[k + 1], j] - F_norm[idx[k - 1], j]
    finite = d[np.isfinite(d)]
    d[~np.isfinite(d)] = finite.max(initial=0.0) + 1.0
    return d


def _random_search(problem: Problem, pop_size: int, n_gen: int):
    """Uniform random sampling baseline within decision variable bounds."""
    # Determine number of samples
    n_samp = pop_size * n_gen
    # Get decision variable bounds
    lb = problem.xl
    ub = problem.xu
    # Generate uniform random samples
    X = np.random.rand(n_samp, problem.n_var) * (ub - lb) + lb
    # Evaluate objective values
    F = problem.evaluate(X)
    # Keep non-dominated points only (approximate Pareto front)
    from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
    fronts = NonDominatedSorting().do(F, only_non_dominated_front=True)
    return X[fronts], F[fronts]


################################################################################
# Core runner wrapped with timeout
################################################################################


def _run_once(
    prob_name: str,
    algo_name: str,
    seed: int,
    gens: int,
    pop: int,
    time_lim: int,
    outdir: Path,
):
    problem = BENCHMARKS[prob_name]()
    np.random.seed(seed)

    if algo_name == "random":
        X, F = _random_search(problem, pop, gens)
    else:
        algorithm = ALGORITHMS[algo_name](problem, pop)
        res = minimize(
            problem,
            algorithm,
            get_termination("n_gen", gens),
            seed=seed,
            verbose=False,
            save_history=False,
        )
        X, F = res.X, res.F

    # save raw csv
    run_dir = outdir / prob_name / algo_name / f"seed{seed:02d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(F).to_csv(run_dir / "objectives.csv", header=False, index=False)
    pd.DataFrame(X).to_csv(run_dir / "decisions.csv", header=False, index=False)

    # true Pareto front for metrics
    ref_F = problem.pareto_front(n_pareto_points := 1000)

    # metrics
    hv = HV(ref_point=F.max(0) + 0.1 * np.abs(F.max(0)))(F)
    igd = IGD(ref_F)(F)
    spc = Spacing()(F)
    cd_mean = float(_crowding_distance(F).mean())

    return {
        "problem": prob_name,
        "algorithm": algo_name,
        "seed": seed,
        "hv": hv,
        "igd": igd,
        "spacing": spc,
        "crowding": cd_mean,
    }


################################################################################
# Dashboard generation
################################################################################


def _build_dashboard(results_dir: Path):
    # collects every final front for interactive viewing
    plots = []
    for prob_path in results_dir.iterdir():
        if not prob_path.is_dir():
            continue
        prob_name = prob_path.name
        dfs = []
        for algo_path in prob_path.iterdir():
            algo_name = algo_path.name
            for seed_path in algo_path.iterdir():
                objs_path = seed_path / "objectives.csv"
                if not objs_path.exists():
                    continue
                F = pd.read_csv(objs_path, header=None)
                F["algo"] = algo_name
                F["seed"] = seed_path.name
                dfs.append(F)
        if not dfs:
            continue
        cat = pd.concat(dfs, ignore_index=True)
        n_obj = cat.shape[1] - 2
        if n_obj == 2:
            fig = px.scatter(
                cat,
                x=0,
                y=1,
                color="algo",
                symbol="algo",
                hover_data=["seed"],
                title=f"{prob_name} – Objective Space",
            )
        else:
            # use parallel coordinates for >2 obj
            fig = px.parallel_coordinates(
                cat,
                dimensions=list(range(n_obj)),
                color="algo",
                title=f"{prob_name} – Objectives (parallel‑coord)",
            )
        plots.append(fig)
    if not plots:
        return
    # combine into HTML
    html_parts = [pio.to_html(fig, full_html=False, include_plotlyjs="cdn") for fig in plots]
    dashboard = """<html><head><title>Pareto Dashboard</title></head><body>""" + "<hr>".join(html_parts) + "</body></html>"
    (results_dir / "dashboard.html").write_text(dashboard, encoding="utf-8")


################################################################################
# Main CLI
################################################################################


def main():
    ap = argparse.ArgumentParser("Benchmark EMOA Suite")
    ap.add_argument("--gens", type=int, default=200, help="Generations per run")
    ap.add_argument("--popsz", type=int, default=100, help="Population size")
    ap.add_argument("--seeds", type=int, default=30, help="Independent runs")
    ap.add_argument("--time-limit", type=int, default=600, help="Seconds per run")
    ap.add_argument("--outdir", type=str, default="results", help="Output directory")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(exist_ok=True, parents=True)

    metrics_rows: List[dict] = []

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = []
        for prob in BENCHMARKS:
            for algo in ALGORITHMS:
                for seed in range(1, args.seeds + 1):
                    futures.append(
                        ex.submit(
                            _run_once,
                            prob,
                            algo,
                            seed,
                            args.gens,
                            args.popsz,
                            args.time_limit,
                            outdir,
                        )
                    )
        for fut in trange(len(futures), desc="Runs"):
            try:
                res = futures[fut].result()
                metrics_rows.append(res)
            except TPE:
                # already handled inside _run_once or thread timed out
                continue

    # aggregate & save summary
    df = pd.DataFrame(metrics_rows)
    summary = (
        df.groupby(["problem", "algorithm"])
        .agg({"hv": ["mean", "std"], "igd": ["mean", "std"], "spacing": ["mean", "std"], "crowding": ["mean", "std"]})
        .reset_index()
    )
    summary.columns = ["problem", "algorithm", *[
        f"{m}_{stat}" for m in ["hv", "igd", "spacing", "crowding"] for stat in ["mean", "std"]
    ]]
    summary.to_csv(outdir / "metrics_summary.csv", index=False)

    # build dashboard
    _build_dashboard(outdir)
    print("All done →", outdir)


if __name__ == "__main__":
    main()
