# Import libraries 
import numpy as np
import math
import os, csv

# ------------------------
# Logging & Trajectory I/O
# ------------------------

THERMO_CSV = "logs/thermo.csv"   # thermodynamic properties log
XYZ_PREFIX = "traj/frame_"       # XYZ trajectory prefix
XYZ_EVERY_CYCLES = 500           # dump XYZ every 500 cycles


def write_xyz(positions, L, moves):
    """
    Write one XYZ snapshot with box length in the comment line

    Arguments:
        positions: array-like of shape (N, 3) with particle coordinates
        L: box length
        moves: current total number of MC moves (for filename)
    """
    path = f"{XYZ_PREFIX}{moves:09d}.xyz"  # traj/frame_000010000.xyz
    with open(path, "w") as xyz_file:
        xyz_file.write(f"{len(positions)}\nL={L:.8f}\n")
        for x, y, z in positions:
            xyz_file.write(f"Ar {x:.8f} {y:.8f} {z:.8f}\n")


def log_thermo_header(path):
    """Write header row for thermodynamic CSV log."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "moves", "cycle",
            "U_trunc_perN", "U_tail_perN", "U_corr_perN",
            "P_trunc", "P_tail", "P_corr",
            "acc_ratio", "step", "L"
        ])


def log_thermo_row(path, moves, cycle_index, observables, acc_ratio, step, L):
    """
    Append a single thermodynamic data row to the CSV file.

    observables is a dict from compute_observables().
    """
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            moves, cycle_index,
            f"{observables['U_trunc_perN']:.8f}",
            f"{observables['U_tail_perN']:.8f}",
            f"{observables['U_corr_perN']:.8f}",
            f"{observables['P_trunc']:.8f}",
            f"{observables['P_tail']:.8f}",
            f"{observables['P_corr']:.8f}",
            f"{acc_ratio:.4f}",
            f"{step:.4f}",
            f"{L:.8f}",
        ])


# -------------------------------------------------------------
# Lattice initializer (simple cubic, cell-centered variant)
# -------------------------------------------------------------
def lattice_displace(box_size: float, num_particles: int) -> np.ndarray:
    """
    Place `num_particles` points on a cubic lattice inside a box of length `box_size`,
    using cell-centered coordinates (delta/2, 3*delta/2, ...). 
    So each atom sits in the middle of a small cube cell.
    """
    if num_particles <= 0:
        return np.zeros((0, 3), dtype=float)
    
    n = math.ceil(num_particles ** (1.0 / 3.0))
    delta = box_size / n

    positions = np.zeros((num_particles, 3), dtype=float)
    atom_placed_counter = 0
    coords = np.arange(delta / 2.0, box_size + 1e-12, delta)
    for x in coords:
        for y in coords:
            for z in coords:
                if atom_placed_counter >= num_particles:
                    return positions
                positions[atom_placed_counter] = (x, y, z)
                atom_placed_counter += 1
    return positions


# --------------------------------------------------------
# LJ pair energy and virial (r·f term) & distance with MIC
# --------------------------------------------------------
def ener(cutoff_sq: float, distance_sq: float, sigma_sq: float, epsilon: float, shift: bool):
    """
    Lennard-Jones pair energy (optionally shifted) and virial term r·f. 
    """
    if distance_sq < cutoff_sq:
        s2_over_r2 = sigma_sq / distance_sq
        s6_over_r6 = s2_over_r2 ** 3
        s12_over_r12 = s6_over_r6 ** 2

        energy = 4.0 * epsilon * (s12_over_r12 - s6_over_r6)

        # potential shift at rc if set to True
        if shift:
            s2_over_rc2 = sigma_sq / cutoff_sq
            s6_over_rc6 = s2_over_rc2 ** 3
            s12_over_rc12 = s6_over_rc6 ** 2
            energy -= 4.0 * epsilon * (s12_over_rc12 - s6_over_rc6)

        # virial r·f = 24ε(2(σ/r)^12 - (σ/r)^6)  == 48ε[(σ/r)^12 - 0.5(σ/r)^6]
        virial = 48.0 * epsilon * (s12_over_r12 - 0.5 * s6_over_r6)
    else:
        energy = 0.0
        virial = 0.0

    return energy, virial


def minimum_image(disp: np.ndarray, L: float) -> np.ndarray:
    """
    Apply the minimum-image convention (MIC) component-wise.
    Only pairwise interactions need MIC to compute distance correctly under periodic boundary conditions.

    Given a displacement vector disp = r_i - r_j in a cubic box of length L,
    wrap it so that each component lies in [-L/2, L/2).

    Mathematically:
        disp <- disp - L * round(disp / L)

    """
    return disp - L * np.rint(disp / L)


def potential(posi,
              particle_id: int,
              start_particle: int = 1,
              box_params: tuple = None):
    """
    Accumulate LJ energy and virial for one particle i against a range of partners,
    with MIC and optional potential shift.

    Arguments:
        posi: array-like of shape (N, 3) with particle coordinates
        particle_id: 1-based index of the particle for which to compute energy/virial
        start_particle: 1-based index of the first partner particle to consider
        box_params: tuple (boxSize, cutoffSquare, sigmaSquare, epsilon, shift)
    Returns:
        energy: total LJ energy contribution for particle_id
        vir: total LJ virial contribution for particle_id
    """
    if box_params is None:
        raise ValueError("box_params must be a tuple: (boxSize, cutoffSquare, sigmaSquare, epsilon, shift)")

    box_size, cutoff_sq, sigma_sq, epsilon, shift = box_params

    pos = np.asarray(posi, dtype=float)
    N = pos.shape[0]

    i = int(particle_id) - 1
    start_idx = int(start_particle) - 1
    if not (0 <= i < N):
        raise IndexError("particle_id out of range for positions array")

    energy = 0.0
    vir = 0.0
    L = box_size

    ri = pos[i]

    for part in range(start_idx, N):
        if part == i:
            continue
        rj = pos[part]

        disp = ri - rj
        disp = minimum_image(disp, L)
        dist_sq = float(np.dot(disp, disp))

        e_pair, v_pair = ener(cutoff_sq, dist_sq, sigma_sq, epsilon, shift)
        energy += e_pair
        vir    += v_pair

    return energy, vir



# ----------------------
# Tail correction terms
# ----------------------
def tail_pressure(rc: float, sigma: float, epsilon: float, rho: float) -> float:
    """
    LJ tail correction to pressure (virial route), P_tail.

    Arguments:
        rc: cutoff distance
        sigma: LJ sigma
        epsilon: LJ epsilon
        rho: number density (N/V)
    Returns:
        correct_pressure: tail correction to pressure
    """
    sr = sigma / rc  # σ/rc
    correct_pressure = (16.0 / 3.0) * np.pi * epsilon * (rho ** 2) * (sigma ** 3) * \
                       ((2.0 / 3.0) * sr**9 - sr**3)
    return correct_pressure


def tail_energy(rc: float, sigma: float, epsilon: float, rho: float) -> float:
    """
    LJ tail correction to potential energy per particle, U_tail/N.

    Arguments:
        rc: cutoff distance
        sigma: LJ sigma
        epsilon: LJ epsilon
        rho: number density (N/V)
    Returns:
        correct_energy: tail correction to potential energy per particle
    """
    sr = sigma / rc  # σ/rc
    correct_energy = (8.0 / 3.0) * np.pi * epsilon * rho * (sigma ** 3) * \
                     ((1.0 / 3.0) * sr**9 - sr**3)
    return correct_energy




# -------------------------
# Total energy & virial sum
# -------------------------
def total_energy(posi,
                 tail_cor: bool,
                 box_params: tuple):
    """
    Sum over unique pairs by calling potential() with 1-based indices.
    Returns total truncated U and total virial W (r·f).
    If tail_cor=True, adds N * U_tail_per_particle to the returned energy.
    """
    box_size, cutoff_sq, sigma_sq, epsilon, shift = box_params
    pos = np.asarray(posi, dtype=float)
    N = pos.shape[0]

    total_E = 0.0
    total_Vir = 0.0

    for particle in range(1, N + 1):
        e_i, v_i = potential(pos, particle, particle, box_params)
        total_E  += e_i
        total_Vir += v_i

    if tail_cor:
        rho = N / (box_size ** 3)
        rc = math.sqrt(cutoff_sq)
        sigma = math.sqrt(sigma_sq)
        total_E += N * tail_energy(rc, sigma, epsilon, rho)


    return total_E, total_Vir



# ---------------------
# Single-particle move
# ---------------------
def mc_move(posi, dr, beta, box_params, rng, reject_cutoff: float = 50.0):
    """
    Propose a single-particle displacement with PBC; return accept counter,
    ΔU, ΔW and possibly updated configuration.

    rng: np.random.Generator (required)
    reject_cutoff: threshold for early reject when βΔU is very large.
    """
    box_size, cutoff_sq, sigma_sq, epsilon, shift = box_params

    config = np.array(posi, dtype=float, copy=True)
    N = config.shape[0]

    # pick particle (1-based)
    pick_random_particle = int(rng.integers(1, N + 1))
    # old contribution
    old_pot, old_vir = potential(config, pick_random_particle, 1, box_params)

    # save(copy), then attempt move
    i = pick_random_particle - 1
    old_conf = config[i].copy()
    config[i] = config[i] + dr * (rng.random(3) - 0.5)

    # PBC wrap
    config[i] = np.mod(config[i], box_size)

    # new contribution
    new_pot, new_vir = potential(config, pick_random_particle, 1, box_params)
    dU = new_pot - old_pot
    dW = new_vir - old_vir

    # --------------------------------------------------------------------
    # Overflow/underflow-safe Metropolis (a safe early-reject/accept path)
    # Accept with probability min(1, exp(-βΔU))
    # --------------------------------------------------------------------
    accept = False

    if dU <= 0.0:
        # 1) Energy decreased or stayed the same -> accept with prob 1
        accept = True
    else:
        beta_dU = beta * dU
        if beta_dU >= reject_cutoff:
            # 2) Very large positive ΔU -> exp(-βΔU) ~ 0, reject without exp()
            accept = False
        else:
            # 3) Normal case -> accept with probability exp(-βΔU)
            accept = (rng.random() < math.exp(-beta_dU))

    if accept:
        num_acc = 1
        potential_diff = dU
        vir_diff = dW
        # config already has the new position
    else:
        # revert to old configuration
        config[i] = old_conf
        num_acc = 0
        potential_diff = 0.0
        vir_diff = 0.0

    return num_acc, potential_diff, vir_diff, config



# --------------------------
# Monte Carlo move sequence
# --------------------------
def monte_carlo(posi, attemp, step_space, temp, box_params, rng):
    """
    Perform `attemp` single-particle attempts; return acceptance,
    total ΔU, total ΔW, and final configuration.
    """

    beta = 1.0 / float(temp)
    configuration = np.array(posi, dtype=float, copy=True)

    num_accept = 0
    dE_total = 0.0
    dW_total = 0.0

    for _ in range(int(attemp)):
        acc, dU, dW, configuration = mc_move(configuration, step_space, beta, box_params, rng)
        num_accept += acc
        dE_total += dU
        dW_total += dW

    acc_ratio = num_accept / float(attemp)
    return acc_ratio, dE_total, dW_total, configuration


def adjust_step(frac, step_space, target_frac=0.5, box_size=None):
    """
    Adjust the displacement to steer acceptance toward target_frac.
    """
    step_origin = step_space
    step_new = step_space * (frac / target_frac)

    # caps
    if step_new / step_origin > 1.5:
        step_new = 1.5 * step_origin
    if step_new / step_origin < 0.5:
        step_new = 0.5 * step_origin
    if box_size is not None and step_new > 0.5 * box_size:
        step_new = 0.5 * box_size

    return step_new



# -------------------------
# Physics-level observables
# -------------------------
def compute_observables(pos, N, L, T, rc, sigma, epsilon, shift):
    """
    Given a configuration, compute truncated and corrected
    energy per particle and pressure.

    Arguments:
        rc: cutoff distance
        sigma: LJ sigma
        epsilon: LJ epsilon
        shift: whether LJ is shifted at rc (for pair potential)
    """
    cutoff_sq = rc**2
    sigma_sq  = sigma**2
    box_params_energy = (L, cutoff_sq, sigma_sq, epsilon, shift)

    # Total truncated energy & virial
    U_trunc_tot, W_trunc_tot = total_energy(pos, tail_cor=False, box_params=box_params_energy)

    rho_now = N / (L ** 3)
    U_tail_perN = tail_energy(rc, sigma, epsilon, rho_now)
    P_tail      = tail_pressure(rc, sigma, epsilon, rho_now)

    U_trunc_perN = U_trunc_tot / N
    P_trunc      = rho_now * T + W_trunc_tot / (3.0 * L ** 3)

    U_corr_perN  = U_trunc_perN + U_tail_perN
    P_corr       = P_trunc + P_tail

    return {
        "U_trunc_perN": U_trunc_perN,
        "U_tail_perN":  U_tail_perN,
        "U_corr_perN":  U_corr_perN,
        "P_trunc":      P_trunc,
        "P_tail":       P_tail,
        "P_corr":       P_corr,
    }



# --------------------
# Driver function
# --------------------
def main():
    # --- I/O setup (side effects live here, not at import time) ---
    os.makedirs("logs", exist_ok=True)
    os.makedirs("traj", exist_ok=True)
    log_thermo_header(THERMO_CSV)

    # --------------------
    # Simulation parameters
    # --------------------
    N         = 200
    rho       = 0.8
    T         = 1.0
    sigma     = 1.0
    sigma_sq  = sigma**2
    epsilon   = 1.0
    rc        = 4.0                # cutoff (σ units)
    cutoff_sq = rc**2
    shift     = False              # truncated, NOT shifted
    L         = (N / rho) ** (1.0/3.0)


    # Long-run schedule
    equil_moves   = 2_500_000
    prod_moves    = 15_000_000
    moves_per_cyc = 1_000

    assert equil_moves % moves_per_cyc == 0, "equil_moves must be a multiple of moves_per_cyc"
    assert prod_moves  % moves_per_cyc == 0, "prod_moves  must be a multiple of moves_per_cyc"

    equil_cycles = equil_moves // moves_per_cyc
    prod_cycles  = prod_moves  // moves_per_cyc

    # MC tuning
    step = 0.2
    rng  = np.random.default_rng(56)

    # Initialize configuration
    pos = lattice_displace(L, N)

    # ----------------
    # Equilibration
    # ----------------
    for c in range(equil_cycles):
        # --- PHYSICS: MC moves 
        acc, dE, dW, pos = monte_carlo(
            pos,
            attemp=moves_per_cyc,
            step_space=step,
            temp=T,
            box_params=(L, cutoff_sq, sigma_sq, epsilon, shift),
            rng=rng
        )
        step = adjust_step(frac=acc, step_space=step, target_frac=0.5, box_size=L)

        # PHYSICS: compute thermodynamic properties
        obs = compute_observables(
            pos, N, L, T,
            rc=rc,
            sigma=sigma,
            epsilon=epsilon,
            shift=shift,
        )

        # I/O: log to CSV, write XYZ 
        moves_so_far = (c + 1) * moves_per_cyc
        log_thermo_row(THERMO_CSV, moves_so_far, c + 1, obs, acc_ratio=acc, step=step, L=L)

        if (c + 1) % XYZ_EVERY_CYCLES == 0:
            write_xyz(pos, L, moves_so_far)

    print(f"Equilibration done: {equil_moves:,} moves.")

    # ----------------
    # Production
    # ----------------
    for c in range(prod_cycles):
        # --- PHYSICS: MC moves 
        acc, dE, dW, pos = monte_carlo(
            pos,
            attemp=moves_per_cyc,
            step_space=step,
            temp=T,
            box_params=(L, cutoff_sq, sigma_sq, epsilon, shift),
            rng=rng
        )
        step = adjust_step(frac=acc, step_space=step, target_frac=0.5, box_size=L)

        # PHYSICS: compute thermodynamic properties 
        obs = compute_observables(
            pos, N, L, T,
            rc=rc,
            sigma=sigma,
            epsilon=epsilon,
            shift=shift,
        )

        # I/O: log to CSV, write XYZ
        global_cycle = equil_cycles + c + 1
        moves_so_far = (equil_cycles + c + 1) * moves_per_cyc

        log_thermo_row(THERMO_CSV, moves_so_far, global_cycle, obs, acc_ratio=acc, step=step, L=L)

        if (global_cycle % XYZ_EVERY_CYCLES) == 0:
            write_xyz(pos, L, moves_so_far)

    print(f"Production done: {prod_moves:,} moves.")
    print("Logs: logs/thermo.csv ; Traj: traj/frame_*.xyz")


if __name__ == "__main__":
    main()
