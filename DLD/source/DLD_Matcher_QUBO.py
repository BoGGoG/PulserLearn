import numpy as np
from scipy.optimize import least_squares
from functools import partial
from .helpers import matprint


def calc_time_sum(t_u1, t_u2, t_mcp) -> float:
    ts = t_u1 + t_u2 - 2 * t_mcp
    return ts


def generate_event_u1_u2_MCP(
    time_sum: float = 120.0,
    bin_max: int = 255,
    binsize: float = 0.8,
    noise_std: float = 0.0,
    mcp=None,
    g=np.random.default_rng(seed=1337),
) -> np.ndarray:
    """
    Assume constant time sum
    time sum: t_{u_1} + t_{u_2} - 2 t_{mcp}

    Parameters:
    seed: int
    time_sum: float, in ns.
    binsize: float, in ns
    bin_max: int, max number of bins, so max length of time series
    """
    t_u2 = -1
    while t_u2 > bin_max or t_u2 < 0:
        t_mcp = g.normal(10, 10) if mcp is None else mcp
        t_u1 = g.uniform(0, bin_max)
        t_u2 = time_sum + 2 * t_mcp - t_u1 + g.normal(0, noise_std)
    # ts = calc_time_sum(t_u1, t_u2, t_mcp)
    return np.array([t_u1, t_u2, t_mcp])


def generate_double_u1_u2_MCP(
    time_sum: float = 120.0,
    bin_max: int = 255,
    binsize: float = 0.8,
    noise_std: float = 0.0,
    mcp=None,
    g=np.random.default_rng(seed=1337),
) -> np.ndarray:
    """
    asdf
    """
    t_u1_1, t_u2_1, t_mcp_1 = generate_event_u1_u2_MCP(
        time_sum=time_sum,
        bin_max=bin_max,
        binsize=binsize,
        noise_std=noise_std,
        g=g,
        mcp=mcp,
    )
    t_u1_2, t_u2_2, t_mcp_2 = generate_event_u1_u2_MCP(
        time_sum=time_sum,
        bin_max=bin_max,
        binsize=binsize,
        noise_std=noise_std,
        g=g,
        mcp=mcp,
    )

    return np.array(
        [
            [t_u1_1, t_u1_2],
            [t_u2_1, t_u2_2],
            [t_mcp_1, t_mcp_2],
        ]
    )


def generate_full_event(
    n_wires: int = 3,
    time_sum: float = 120.0,
    bin_max: int = 255,
    binsize: float = 0.8,
    noise_std: float = 0.0,
    mcp=None,
    g=np.random.default_rng(seed=1337),
) -> np.ndarray:
    return 0


def get_QUBO_matrix(ti_u1, ti_u2, ti_mcp, time_sum):
    delta_u1 = ti_u1[0] - ti_u1[1]
    delta_u2 = ti_u2[0] - ti_u2[1]
    delta_mcp = ti_mcp[0] - ti_mcp[1]
    ts_1 = calc_time_sum(ti_u1[0], ti_u2[0], ti_mcp[0])
    ts_2 = calc_time_sum(ti_u1[1], ti_u2[1], ti_mcp[1])

    Q = np.zeros((3, 3))
    Q[0, 0] = 2 * delta_u1**2 - 2 * delta_u1 * ts_1 + 2 * delta_u1 * ts_2
    Q[1, 1] = 2 * delta_u2**2 - 2 * delta_u2 * ts_1 + 2 * delta_u2 * ts_2
    Q[2, 2] = 4 * delta_mcp * (2 * delta_mcp + ts_1 - ts_2)
    Q[0, 1] = 2 * delta_u1 * delta_u2
    Q[1, 0] = Q[0, 1]
    Q[0, 2] = -4 * delta_u1 * delta_mcp
    Q[2, 0] = Q[0, 2]
    Q[1, 2] = -4 * delta_u2 * delta_mcp
    Q[2, 1] = Q[1, 2]

    return Q


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


def show_generate_event_u1_u2_MCP():
    g = np.random.default_rng(42)
    time_sum = 120.0
    e = generate_event_u1_u2_MCP(
        time_sum=time_sum, bin_max=255, binsize=0.8, noise_std=1.0, g=g
    )
    print(f"event: {e}")
    ts = calc_time_sum(e[0], e[1], e[2])
    print(f"ts: {ts}")

    e = generate_event_u1_u2_MCP(
        time_sum=time_sum,
        bin_max=255,
        binsize=0.8,
        noise_std=1.0,
        mcp=10.0,
        g=g,
    )
    print(f"single hit event: {e}")

    e12 = generate_double_u1_u2_MCP(
        time_sum=time_sum,
        bin_max=255,
        binsize=0.8,
        noise_std=1.0,
        g=g,
    )
    # randomly permute second dimension. perms = [0/1, 0/1, 0/1]
    # if 1, swap
    perms = g.integers(0, 2, size=3)
    for i in range(3):
        if perms[i] == 1:
            e12[i] = e12[i, ::-1]
    print(f"event: {e12}")
    ts = calc_time_sum(e[0], e[1], e[2])
    print(f"ts: {ts}")

    print(f"event: {e12}")
    ts_1 = calc_time_sum(e12[0, 0], e12[1, 0], e12[2, 0])
    ts_2 = calc_time_sum(e12[0, 1], e12[1, 1], e12[2, 1])
    print(f"ts_1: {ts_1}")
    print(f"ts_2: {ts_2}")

    Q = get_QUBO_matrix(e12[0], e12[1], e12[2], time_sum)
    matprint(Q)

    # q = np.array([1, 1, 1])
    # print(q.T @ Q @ q)
    sol = solve_qubo_bruteforce(Q, 4)
    print(sol)
    print(f"perms: {perms}")

    print("-" * 20)


if __name__ == "__main__":
    show_generate_event_u1_u2_MCP()
    #
    # e = generate_full_event()
    # print(e)
