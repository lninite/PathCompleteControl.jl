# PathCompleteControl.jl

This repository contains all the code necessary to reproduce the experiments presented in

> **Robust Optimal Control of Arbitrarily Switched Systems: A Path-Complete Framework**

**Authors:**  
Léa Ninite, Adrien Banse, Guillaume O. Berger, Raphaël M. Jungers

<!-- Optionally add -->
**Paper:** [Preprint](link)

---

# Repository Structure

The package is organized as follows.

### `src/`

The `src/PathCompleteControl.jl` module is composed of the following files:

- `complete_graph_policy_synthesis.jl`  
  Implements the synthesis of a policy together with an upper bound on the closed-loop value function when using a *complete* path-complete graph. It consists of solving a single SDP; see Section 5.2 of the paper.

- `general_graph_policy_synthesis.jl`  
  Implements the synthesis of a policy together with an upper bound on the closed-loop value function when using a *general* path-complete graph (i.e., not necessarily complete). It consists of an alternating optimization procedure involving a sequence of SDPs; see Algorithm 1 of the paper.

- `mpc.jl`  
  Implements the robust Model Predictive Control (MPC) algorithms (with and without a terminal cost) used for comparison with our method in Section 6.

- `simulate.jl`  
  Simulation routines for closed-loop switched systems.

- `utils.jl`  
  Utility functions for data saving and result formatting.

---

### `experiments/`

Contains all the scripts used to reproduce the numerical experiments presented in the paper.

These scripts generate the data and figures reported in the manuscript.

- `2D_value_function.jl`  
  Reproduces Figure 5 of the paper, illustrating the computed upper bounds obtained with our approach for a two-dimensional system.

- `temperature_regulation.jl`  
  Reproduces Table 1 of the paper, comparing our approach (complete and co-complete graphs) with MPC (with and without a terminal cost) in terms of performance and computation time.

- `plot_trajectories.jl`  
  Reproduces Figure 7 of the paper, comparing the closed-loop trajectories obtained with several controllers (path-complete controllers and MPC).

---

### `data/`

Stores the generated experimental data.
