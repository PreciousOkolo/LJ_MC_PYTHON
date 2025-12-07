# Lennard-Jones Monte Carlo (NVT) Simulation

This repository contains a clean and modular implementation of a Lennard-Jones (LJ) fluid Monte Carlo simulation in the NVT ensemble.  
The code is designed for clarity, reproducibility, and teaching/learning molecular simulation methods.

---

## File Overview

**Main script**

- `LJ_Monte_Carlo_Simulation.py` – main driver script and all core functions.

**Core functions**

- `write_xyz()` – write XYZ trajectory frames.
- `log_thermo_header()` – initialize the thermodynamic CSV log file.
- `log_thermo_row()` – append one thermodynamic data row to the CSV file.
- `lattice_displace()` – build the initial simple-cubic (cell-centered) configuration.
- `ener()` – compute Lennard-Jones pair energy and virial contribution.
- `minimum_image()` – apply the minimum-image convention (MIC) under periodic boundary conditions.
- `potential()` – compute energy \(U_i\) and virial \(W_i\) for a single particle.
- `tail_energy()` – analytic Lennard-Jones tail correction for energy.
- `tail_pressure()` – analytic Lennard-Jones tail correction for pressure.
- `total_energy()` – compute full truncated potential energy and virial.
- `mc_move()` – perform a single-particle Metropolis Monte Carlo move.
- `monte_carlo()` – perform a sequence of trial moves and return acceptance statistics.
- `adjust_step()` – tune displacement step size toward a target acceptance ratio (e.g. 50%).
- `compute_observables()` – compute energy and pressure (truncated and tail-corrected).
- `main()` – top-level driver: sets parameters, runs equilibration and production, writes outputs.

---

## Outputs

### `logs/thermo.csv`

Cycle-averaged thermodynamic data, including:

- Energy per particle (truncated and corrected)
- Pressure (truncated and corrected)
- Tail correction terms
- Acceptance ratio
- MC step size

### `traj/frame_XXXXXXXXX.xyz`

XYZ trajectory snapshots for visualization in tools such as:

- OVITO  
- VMD  
- PyMOL  
- ASE viewer

---

## Author

**Precious C. Okolo**  
PhD Student — Chemical Engineering  
University of Notre Dame
