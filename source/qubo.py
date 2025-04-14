import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rc
import warnings

rc("text", usetex=True)

import pulser
from pulser.devices import DigitalAnalogDevice, MockDevice
from pulser.register import Register
from pulser.pulse import Pulse
from pulser.waveforms import BlackmanWaveform
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform


def get_Q_from_coords(coords, device=DigitalAnalogDevice) -> np.ndarray:
    """
    Compute the QUBO matrix from the coordinates of the qubits.
    """
    Q = squareform(device.interaction_coeff / pdist(coords) ** 6)
    return Q


def evaluate_mapping(
    coords, Q, device=DigitalAnalogDevice, lossf: str = "rmae"
) -> float:
    """
    Cost function to minimize.
    We want to embedd the QUBO problem in the device.
    """
    coords = np.reshape(coords, (len(Q), 2))
    # computing the matrix of the distances between all coordinate pairs
    new_Q = squareform(device.interaction_coeff / pdist(coords) ** 6)

    if lossf == "mae":
        return np.mean(np.abs(new_Q - Q))
    elif lossf == "rmse":
        return np.sqrt(np.mean((new_Q - Q) ** 2))
    else:
        try:
            return lossf(new_Q, Q)
        except:
            raise ValueError("lossf must be either 'mae', 'rmse' or a function")


def get_register_embedding(
    Q: np.ndarray,
    seed: int = 2,
    maxiter: int = 2000000,
    lossf: str = "rmse",
    maxfev=None,
    bounds=None,
    tol: float = 1e-6,
    max_rel_diff=0.2,
    method: str = "Nelder-Mead",
    device=DigitalAnalogDevice,
    ignore_warnings: bool = False,
) -> tuple:
    """
    Embed a QUBO problem in a register.
    This does not always work perfectly!
    """
    # if not all diagonal elements of Q are equal, raise a warning
    if (not np.allclose(np.diag(Q), np.diag(Q)[0])) and (not ignore_warnings):
        warnings.warn("The diagonal elements of the QUBO matrix are not equal.")
        warnings.warn(f"Diagonal elements: {np.diag(Q)}")

    bounds = bounds or [(-30, 30) for _ in range(len(Q) * 2)]

    np.random.seed(seed)
    x0 = np.random.random(len(Q) * 2)
    res = minimize(
        evaluate_mapping,
        x0,
        args=(Q, device, lossf),
        method=method,  # "nelder-mead", # method="cg", # method="cobyqa", # method = "l-bfgs-b", # method = "powell", # method = "trust-constr",
        tol=tol,
        options={"maxiter": maxiter, "maxfev": maxfev},
        bounds=bounds,
    )
    coords = np.reshape(res.x, (len(Q), 2))

    new_Q = get_Q_from_coords(coords, device)
    differences = np.abs(new_Q - Q)
    rel_diff = differences / (0.5 * new_Q + 0.5 * Q + 1e-6)
    # set diagonals in rel_diff to zero
    rel_diff[np.diag_indices_from(rel_diff)] = 0
    if np.any(rel_diff > max_rel_diff) and not ignore_warnings:
        warnings.warn(
            f"Relative differences between the QUBO matrix and the new matrix are too high: {np.max(rel_diff)}"
        )
        warnings.warn(f"rel_diff: \n{rel_diff}")

    return coords, res


def anneal(
    reg,
    Omega,
    delta_i=-1,
    delta_f=None,
    T: int = 4000,
    draw_pulse: bool = False,
    draw_distribution: bool = False,
    show: bool = True,
    device=DigitalAnalogDevice,
) -> dict:
    # We choose a median value between the min and the max
    delta_f = delta_f if delta_f is not None else -delta_i

    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_i, 0, delta_f]),
        0,
    )
    if draw_pulse:
        print(
            f"Pulse integral: {InterpolatedWaveform(T, [1e-9, Omega, 1e-9]).integral}"
        )
    seq = Sequence(reg, device)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")

    simul = QutipEmulator.from_sequence(seq)
    results = simul.run()
    final = results.get_final_state()
    count_dict = results.sample_final_state()

    if draw_pulse:
        seq.draw()
    if draw_distribution:
        plot_distribution(count_dict, show=show)

    return count_dict


def plot_distribution(
    C, solutions: list = [], show: bool = True, custom_ax=None, n_max_states: int = 10
):
    # Sort dictionary by values in descending order and take first n_max_states items
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True)[:n_max_states])
    color_dict = {key: "r" if key in solutions else "g" for key in C}
    if custom_ax is None:
        plt.figure(figsize=(8, 3))
    else:
        plt.sca(custom_ax)
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation="vertical")
    if show:
        plt.show()


def get_highest_counts(counts: dict, n: int):
    """
    Parameters:
    counts (dict): dictionary {name, count} of counts

    Returns:
    list: list [name, dict] of the n highest counts. Dict is {count, proba}
    """
    total = sum(counts.values())
    highest_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
    probas = [count / total for name, count in highest_counts]
    out = []
    for i in range(n):
        out.append(
            [highest_counts[i][0], {"count": highest_counts[i][1], "proba": probas[i]}]
        )
    return out


def draw_solutions(
    reg: Register,
    counts: dict,
    n: int,
    device=DigitalAnalogDevice,
    draw_graph: bool = False,
    show: bool = True,
    custom_ax=None,
    Omega: float = 1.0,
):
    """
    Parameters:
    reg (Register): register of the circuit
    counts (dict): dictionary {name, count} of counts
    n (int): number of solutions to draw
    """
    plt.rc("text", usetex=True)
    highest_counts = get_highest_counts(counts, len(counts))  # sorted by count
    probas = [val[1]["proba"] for val in highest_counts]
    if custom_ax is None:
        fig, axs = plt.subplots(1, n, figsize=(4 * n, 4))
    else:
        axs = custom_ax
    for ii in range(n):
        if n > 1:
            axs[ii].axis("off")
            axs[ii].set_title(f"Solution {ii+1}")
            plt.sca(axs[ii])
        reg.draw(
            blockade_radius=device.rydberg_blockade_radius(Omega),
            draw_graph=draw_graph,
            draw_half_radius=True,
            qubit_colors={
                f"q{i}": "red"
                for i, val in enumerate(highest_counts[ii][0])
                if val == "1"
            },
            show=False,
            custom_ax=axs[ii] if n > 1 else axs,
        )
        legend_elements = [
            Patch(facecolor="mistyrose", label=r"State $|1\rangle$"),
            Patch(facecolor="lightgreen", label=r"State $|0\rangle$"),
        ]
        plt.legend(handles=legend_elements)
        title = f"Sol. {highest_counts[ii][0]} "
        title += f"Probas: " + ", ".join(
            [
                f"${probas[i]:.2f}$" if i != ii else f"$\\underline{{{probas[i]:.2f}}}$"
                for i in range(min(n + 2, len(probas)))
            ]
        )
        plt.title(title)
    if show:
        plt.show()


def solve_qubo_bruteforce(Q: np.ndarray, n: int = 1) -> list:
    """
    Solve a QUBO problem using a brute-force approach.
    Args:
        Q (np.ndarray): The QUBO matrix.
        n (int): The number of solutions to return (sorted by energy).
    Returns:
        list:
            A list of tuples, each containing a solution and its energy.
    """
    solutions = []
    for i in range(2 ** len(Q)):
        sol = np.array(list(bin(i)[2:].zfill(len(Q))), dtype=int)
        energy = np.dot(sol, np.dot(Q, sol))
        solutions.append((sol, energy))
    return sorted(solutions, key=lambda x: x[1])[:n]


def scale_and_round_qubo(
    Q: np.ndarray,
    max_abs_value: int,
    max_scale_multiplier: int = 100,
) -> tuple[np.ndarray, float, dict]:
    """
    Scale and round QUBO matrix Q such that the unrounded Q is as close to integer as possible.
    For some embedding schemes an integer Q is needed.

    Parameters:
    Q: np.ndarray, the QUBO matrix
    max_abs_value: int, the maximum absolute value of the scaled QUBO matrix
    max_scale_multiplier: int, the maximum scale multiplier for scaling during optimization. Default is 100.
    """

    def int_loss(Q: np.ndarray, scale: float) -> float:
        Q_scaled = Q / scale
        Q_scaled_int = np.round(Q_scaled)
        return np.sqrt(
            np.mean(
                np.square(
                    (
                        # (Q_scaled - Q_scaled_int) / (0.5 * Q_scaled + 0.5 * Q_scaled_int) # relative error
                        Q_scaled - Q_scaled_int
                    )
                )
            )
        )

    max_scale = np.max(np.abs(Q)) / max_abs_value
    res = least_squares(
        partial(int_loss, Q),
        x0=max_scale * 2,
        bounds=(max_scale, max_scale_multiplier * max_scale),
        method="trf",
    )
    scale = res.x[0]
    loss = int_loss(Q, scale)
    props = {
        "loss": loss,
        "opt_props": res,
    }

    return (np.round(Q / scale), scale, props)
