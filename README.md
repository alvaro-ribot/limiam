# LiMIAM: Causal discovery of linear mean-independent acyclic models

This repository contains a self-contained Python implementation of
DirectLiMIAM and the scripts needed to reproduce the empirical figures and
tables from the paper.

DirectLiMIAM estimates a causal order and a weighted adjacency matrix for a
linear structural equation model when the structural errors are allowed to be
mean-independent rather than fully independent. The implementation also includes
a DirectLiNGAM-style score as a baseline, using the same source-removal
algorithmic skeleton.

## Citation

If you use this code, please cite the paper:

```bibtex
@misc{limiam,
      title={Causal discovery under mean independence and linearity}, 
      author={Geert Mesters and Alvaro Ribot and Anna Seigal and Piotr Zwiernik},
      year={2026},
      eprint={2605.04381},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2605.04381}, 
}
```

## Acknowledgements

Parts of our code are adapted from the LiNGAM Python project:
<https://github.com/cdt15/lingam>.

## Installation

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Graphviz is required for the oil-market DAG plots. On Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y graphviz
```

On macOS with Homebrew:

```bash
brew install graphviz
```

## Quick Start

Use `DirectLiMIAM` directly from Python:

```python
import numpy as np
from direct_limiam import DirectLiMIAM

# X has shape (n_samples, n_variables)
X = np.asarray(your_data, dtype=float)

model = DirectLiMIAM(score="kernel_cv").fit(X)

causal_order = model.causal_order_
B = model.adjacency_matrix_
```

The adjacency matrix convention is:

```text
B[target, source] = coefficient for source -> target
```

For example, if `B[3, 1]` is nonzero, variable `1` points to variable `3`.

The default estimator is:

```python
DirectLiMIAM(
    score="kernel_cv",
    adjacency_estimator="ols",
    cv_folds=5,
)
```

## Source Scores

The `score` argument controls how candidate source variables are selected:

- `score="kernel_cv"`: local-linear kernel predictive score with cross-validation. This is the main DirectLiMIAM implementation used for the paper's oil-market application.
- `score="sieve_cv"`: cubic B-spline predictive score with cross-validation.
- `score="moment"`: moment score using powers such as `x^2` and `x^3`.
- `score="pwling"`: DirectLiNGAM-style pairwise likelihood/entropy score. This is used for the DirectLiNGAM baseline.

Example:

```python
model = DirectLiMIAM(score="sieve_cv", adjacency_estimator="ols").fit(X)
```

## Estimating the Adjacency Matrix

DirectLiMIAM first estimates a causal order. Given that order, the code offers
two choices for estimating the weighted adjacency matrix `B`.

### 1. Full OLS, the default

```python
model = DirectLiMIAM(adjacency_estimator="ols").fit(X)
```

For each target variable, this regresses the target on all variables that appear
earlier in the estimated causal order.

Use `ols` when:

- you want the paper's default behavior;
- the number of variables is small or moderate;
- you want a complete weighted triangular matrix conditional on the estimated order;
- you plan to inspect or threshold coefficients yourself.

Full OLS can produce dense graphs. Coefficients are also scale-dependent, so if
you want to compare edge magnitudes across variables, standardize the variables
before fitting or interpret the coefficients in their original units.

### 2. Adaptive LASSO plus OLS refit

```python
model = DirectLiMIAM(adjacency_estimator="adaptive_lasso").fit(X)
```

This first performs adaptive-LASSO parent selection among the earlier variables
in the estimated order, then refits OLS on the selected parents.

Use `adaptive_lasso` when:

- you want a sparser graph;
- the number of variables is larger;
- you prefer automatic parent pruning after the order has been estimated.

Adaptive LASSO may drop weak but scientifically meaningful effects. For the
paper results, use the default `ols`.

## Reproducing the Paper Figures

The commands below use the paper defaults:

- final adjacency estimation uses full OLS;
- the oil-market application standardizes VAR residuals before DirectLiNGAM and DirectLiMIAM;
- the oil-market DAGs use the same display threshold for both methods:
  `0.001`.

### Figures 3 and 4

Run the full Monte Carlo simulation:

```bash
python run_simulations.py --jobs 16
```

This writes:

- `results/simulations/figure3_order_success.pdf`
- `results/simulations/figure4_shd.pdf`
- `results/simulations/summary_results.csv`
- `results/simulations/raw_results.csv`

If `summary_results.csv` already exists and you only want to rebuild the plots:

```bash
python run_simulations.py \
  --plot-only-summary results/simulations/summary_results.csv \
  --output-dir results/simulations
```

### Figure 5

Run the oil-market application:

```bash
python oil_market.py
```

This writes:

- `results/oil_market/figure5_oil_market_dags.pdf`
- `results/oil_market/figure5_oil_market_dags.png`
- `results/oil_market/figure5_directlingam_dag.pdf`
- `results/oil_market/figure5_directlimiam_dag.pdf`
- `results/oil_market/bhat_directlingam.csv`
- `results/oil_market/bhat_directlimiam_kernel.csv`
- `results/oil_market/specification_tests.csv`

The oil-market DirectLiMIAM adjacency matrix with fixed-order bootstrap
standard errors is saved here:

- CSV: `results/oil_market/kernel_b_matrix_with_fixed_order_bootstrap_ses.csv`
- LaTeX: `results/oil_market/kernel_b_table_causal_order.tex`

The file `results/oil_market/oil_variable_scales.csv` records the raw variable
standard deviations and the VAR residual standard deviations used for
standardization.

### Figure 2c

```bash
python figure2c.py
```

This writes:

- `results/figure2c/figure2c_scatter_e1_e2.pdf`
- `results/figure2c/figure2c_scatter_e1_e2.png`

## Useful Command-Line Options

Synthetic simulations:

```bash
python run_simulations.py --help
```

Common options:

- `--jobs 16`: run simulation tasks in parallel.
- `--mc 100`: number of Monte Carlo replications per simulation cell.
- `--adjacency-estimator ols`: default adjacency estimator.
- `--adjacency-estimator adaptive_lasso`: sparse sensitivity analysis.
- `--plot-only-summary PATH`: rebuild plots from an existing summary CSV.

Oil-market application:

```bash
python oil_market.py --help
```

Common options:

- `--raw-residuals`: use raw VAR residuals instead of standardized residuals.
- `--adjacency-estimator adaptive_lasso`: sparse sensitivity analysis.
- `--edge-tol-direct 0.001`: display threshold for DirectLiNGAM DAG edges.
- `--edge-tol-limiam 0.001`: display threshold for DirectLiMIAM DAG edges.
- `--bootstraps 199`: number of fixed-order bootstrap draws.
- `--permutations 499`: number of permutations in the diagnostic tests.

## Output Conventions

Rows are observations and columns are variables.

The adjacency matrix convention is always:

```text
B[target, source] = source -> target
```

The reported causal order is a list of variable indices, from earliest cause to
latest effect. For the oil-market scripts, CSV files with variable names are
also written for readability.

## Notes For Practitioners

1. Start with `score="kernel_cv"` and `adjacency_estimator="ols"`.
2. Standardize your data first if you want edge magnitudes to be comparable
   across variables.
3. Use `adaptive_lasso` as a robustness check when you want a sparse graph.
4. Treat very small coefficients cautiously. The weighted adjacency matrix is
   conditional on the estimated causal order and on the scale of the input data.
5. For drawing DAGs, tune the display thresholds rather than
   changing the fitted adjacency matrix.
