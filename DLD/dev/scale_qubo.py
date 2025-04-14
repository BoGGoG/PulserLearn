import numpy as np
import matplotlib.pyplot as plt
from source.helpers import matprint
from source.DLD_Matcher_QUBO import (
    calc_time_sum,
    generate_event_u1_u2_MCP,
    generate_double_u1_u2_MCP,
    get_QUBO_matrix,
    solve_qubo_bruteforce,
    scale_and_round_qubo,
)
from tqdm import tqdm


def evaluate_scale_and_round_QUBO(n: int, max_abs_value: int = 50):
    """
    Statistically evaluate with random events
    """
    g = np.random.default_rng(42)
    time_sum = 120.0

    losses = []
    for i in tqdm(range(n)):
        e12 = generate_double_u1_u2_MCP(
            time_sum=time_sum,
            bin_max=255,
            binsize=0.8,
            noise_std=1.0,
            g=g,
        )
        Q = get_QUBO_matrix(e12[0], e12[1], e12[2], time_sum)
        scaled_Q, scale, props = scale_and_round_qubo(
            Q, max_abs_value=max_abs_value, max_scale_multiplier=500
        )
        # matprint(scaled_Q)
        losses.append(props["loss"])

    mean_loss = np.mean(losses)
    print(f"mean loss: {mean_loss}")
    plt.hist(losses, bins=50)
    plt.show()


if __name__ == "__main__":
    g = np.random.default_rng(43)
    time_sum = 120.0

    print("-" * 50, "\nTesting double event generation")
    e12 = generate_double_u1_u2_MCP(
        time_sum=time_sum,
        bin_max=255,
        binsize=0.8,
        noise_std=1.0,
        g=g,
    )
    print(f"event: {e12}")
    ts_1 = calc_time_sum(e12[0, 0], e12[1, 0], e12[2, 0])
    ts_2 = calc_time_sum(e12[0, 1], e12[1, 1], e12[2, 1])
    print(f"ts_1: {ts_1}")
    print(f"ts_2: {ts_2}")

    print("\n", "-" * 50, "\nTesting QUBO formulation with generated event")
    Q = get_QUBO_matrix(e12[0], e12[1], e12[2], time_sum)
    print("QUBO matrix:")
    matprint(Q)
    sols = solve_qubo_bruteforce(Q, 8)
    for s in sols:
        print(f"solution: {s[0]}, energy: {s[1]}")
    q = sols[0][0]
    print(f"For {q=}")
    print(f"{q.T @ Q @ q=}")

    print("QUBO matrix:")
    matprint(Q)

    scaled_Q, scale, props = scale_and_round_qubo(
        Q, max_abs_value=50, max_scale_multiplier=500
    )
    print(f"scale: {scale}")
    print("props:", props)
    matprint(scaled_Q)
    matprint(Q / scale)

    evaluate_scale_and_round_QUBO(1_000, max_abs_value=25)
