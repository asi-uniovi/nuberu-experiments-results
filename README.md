# Nuberu experiments

This repository is a companion to the paper: "Modular and Reproducible Simulator Architecture for Composable Cloud Systems"

It contains two notebooks which show statistics and plots for the results of the 16 experiments described in the paper. The input of these notebooks is a large .parquet file containing metadata about every request injected in the simulator for each of the 16 experiments. This parquet file can be downloaded from the releases section of this repository.

You can read the [Static Notebook](https://github.com/asi-uniovi/nuberu-experiments-results/blob/main/Static-Notebook.ipynb) directly in GithHub, since it contains some pre-generated plots. To explore other plots you can use the [Interactive Notebook](https://github.com/asi-uniovi/nuberu-experiments-results/blob/main/Interactive-Notebook.ipynb) but you need to run it in a Jupyter environment.

## Instructions to run the notebooks

1. Install [`uv`](https://docs.astral.sh/uv/) in your machine.
2. Clone this repository.
3. Run `uv sync` in the root of the repository to install all required dependencies.
4. Run `uv run --with jupyter jupyter lab` and open the given URL in your browser.

