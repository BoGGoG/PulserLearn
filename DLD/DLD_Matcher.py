import numpy as np
from source.helpers import matprint
from source.DLD_Matcher_QUBO import (
    calc_time_sum,
    generate_event_u1_u2_MCP,
    generate_double_u1_u2_MCP,
    get_QUBO_matrix,
    solve_qubo_bruteforce,
)


def test_DLD_QUBO(n: int = 10, time_sum=120.0):
    g = np.random.default_rng(42)
    pred = []
    true = []
    vals = []
    for _ in range(n):
        e12 = generate_double_u1_u2_MCP(
            time_sum=time_sum,
            bin_max=255,
            binsize=0.8,
            noise_std=0.0,
            g=g,
        )  # shape (3, 2)
        # randomly permute second dimension. perms = [0/1, 0/1, 0/1]
        # if 1, swap
        perms = g.integers(0, 2, size=3)
        for i in range(3):
            if perms[i] == 1:
                e12[i] = e12[i, ::-1]

        Q = get_QUBO_matrix(e12[0], e12[1], e12[2], time_sum)
        s = solve_qubo_bruteforce(Q, 1)[0]
        pred.append(s[0])
        vals.append(s[1])
        true.append(perms)
    return np.array(true), np.array(pred), np.array(vals)


def check_correct(y, ypred):
    """
    y: binary array
    ypred: binary array

    check if all entries are the same or if all are the same if y is flipped (all entries)
    """
    out = np.all(y == ypred)
    out_flipped = np.all(y == 1 - ypred)

    return out or out_flipped


if __name__ == "__main__":
    g = np.random.default_rng(42)
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

    print("\n", "-" * 50, "\nTesting QUBO formulation with many events")
    true, pred, vals = test_DLD_QUBO(50)
    for t, p, v in zip(true, pred, vals):
        print(f"true: {t}, pred: {p}, energy: {v:10.4}, correct: {check_correct(t, p)}")

# Output of above code:
# (base) ➜  DLD git:(main) ✗ python -m DLD_Matcher
# --------------------------------------------------
# Testing double event generation
# event: [[111.91400214  24.01522371]
#  [ 34.93079065 133.49389111]
#  [ 13.0471708   19.40564716]]
# ts_1: 120.75045119580645
# ts_2: 118.69782049313767
#
#  --------------------------------------------------
# Testing QUBO formulation with generated event
# QUBO matrix:
#  15091.5  -17327.2   2235.61
# -17327.2     19834  -2506.84
#  2235.61  -2506.84   271.235
# solution: [1 1 1], energy: -5.115907697472721e-13
# solution: [0 0 0], energy: 0.0
# solution: [1 1 0], energy: 271.2353583660624
# solution: [0 0 1], energy: 271.23535836606374
# solution: [0 1 1], energy: 15091.543034716908
# solution: [1 0 0], energy: 15091.543034716913
# solution: [0 1 0], energy: 19833.99683508819
# solution: [1 0 1], energy: 19833.996835088197
# For q=array([1, 1, 1])
# q.T @ Q @ q=np.float64(-5.115907697472721e-13)
#
#  --------------------------------------------------
# Testing QUBO formulation with many events
# true: [1 1 1], pred: [1 1 1], energy: -9.663e-13, correct: True
# true: [1 1 0], pred: [0 0 1], energy:     -187.6, correct: True
# true: [1 1 0], pred: [0 0 1], energy:     -504.1, correct: True
# true: [1 0 0], pred: [0 1 1], energy: -7.525e+03, correct: True
# true: [1 1 0], pred: [1 1 0], energy:    -0.6417, correct: True
# true: [1 1 0], pred: [0 0 1], energy:     -1.586, correct: True
# true: [0 0 1], pred: [1 1 0], energy:     -699.3, correct: True
# true: [1 0 0], pred: [1 0 0], energy: -1.536e+03, correct: True
# true: [1 1 1], pred: [0 0 0], energy:        0.0, correct: True
# true: [0 0 1], pred: [1 1 0], energy:     -210.9, correct: True
# true: [0 1 0], pred: [1 0 1], energy:     -61.07, correct: True
# true: [1 1 0], pred: [1 1 0], energy:     -168.8, correct: True
# true: [0 0 0], pred: [0 0 0], energy:        0.0, correct: True
# true: [1 1 0], pred: [1 1 0], energy:     -143.6, correct: True
# true: [0 1 1], pred: [1 0 0], energy:     -338.5, correct: True
# true: [0 0 1], pred: [1 1 0], energy: -1.538e+03, correct: True
# true: [1 1 0], pred: [1 1 0], energy:     -388.4, correct: True
# true: [0 0 1], pred: [1 1 0], energy: -6.953e+03, correct: True
# true: [1 0 0], pred: [0 1 1], energy: -4.715e+03, correct: True
# true: [0 0 1], pred: [1 1 0], energy:     -798.0, correct: True
# true: [0 1 1], pred: [0 1 1], energy:     -262.5, correct: True
# true: [1 1 1], pred: [1 1 1], energy: -1.364e-12, correct: True
# true: [0 0 1], pred: [0 0 1], energy:  -2.62e+03, correct: True
# true: [0 0 0], pred: [0 0 0], energy:        0.0, correct: True
# true: [1 1 0], pred: [0 0 1], energy:     -956.1, correct: True
# true: [1 1 0], pred: [1 1 0], energy:     -5.087, correct: True
# true: [0 1 0], pred: [1 0 1], energy:     -247.3, correct: True
# true: [1 1 0], pred: [0 0 1], energy:     -552.5, correct: True
# true: [0 1 1], pred: [1 0 0], energy:     -689.2, correct: True
# true: [0 0 0], pred: [0 0 0], energy:        0.0, correct: True
# true: [0 1 1], pred: [1 0 0], energy:     -1.542, correct: True
# true: [0 1 0], pred: [1 0 1], energy: -2.127e+03, correct: True
# true: [0 0 1], pred: [0 0 1], energy: -1.112e+03, correct: True
# true: [1 0 0], pred: [0 1 1], energy: -4.525e+04, correct: True
# true: [1 0 1], pred: [0 1 0], energy:     -794.8, correct: True
# true: [1 1 0], pred: [0 0 1], energy:     -702.8, correct: True
# true: [0 1 1], pred: [0 1 1], energy: -1.123e+04, correct: True
# true: [0 0 0], pred: [1 1 1], energy: -1.009e-12, correct: True
# true: [0 1 0], pred: [1 0 1], energy:     -99.29, correct: True
# true: [0 1 0], pred: [0 1 0], energy:  -3.94e+03, correct: True
# true: [0 0 1], pred: [1 1 0], energy:     -771.7, correct: True
# true: [1 1 1], pred: [0 0 0], energy:        0.0, correct: True
# true: [1 1 0], pred: [1 1 0], energy:  -6.18e+03, correct: True
# true: [1 0 0], pred: [1 0 0], energy: -6.354e+03, correct: True
# true: [1 0 1], pred: [0 1 0], energy: -5.934e+03, correct: True
# true: [1 0 0], pred: [0 1 1], energy: -1.078e+04, correct: True
# true: [0 1 1], pred: [0 1 1], energy: -6.882e+03, correct: True
# true: [0 1 0], pred: [0 1 0], energy: -5.211e+03, correct: True
# true: [0 1 1], pred: [0 1 1], energy: -5.471e+03, correct: True
# true: [1 0 0], pred: [1 0 0], energy: -8.657e+03, correct: True
