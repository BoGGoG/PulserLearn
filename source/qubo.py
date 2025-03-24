import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib import rc
rc('text', usetex=True)

import pulser
from pulser.devices import DigitalAnalogDevice
from pulser.register import Register
from pulser.pulse import Pulse
from pulser.waveforms import BlackmanWaveform
from pulser import Pulse, Sequence, Register
from pulser_simulation import QutipEmulator
from pulser.waveforms import InterpolatedWaveform
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def anneal(reg, Omega, delta_i=-1, delta_f=1, T:int=4000, draw:bool=False) -> dict:
    # We choose a median value between the min and the max
    delta_f = -delta_i

    adiabatic_pulse = Pulse(
        InterpolatedWaveform(T, [1e-9, Omega, 1e-9]),
        InterpolatedWaveform(T, [delta_i, 0, delta_f]),
        0,
    )
    seq = Sequence(reg, DigitalAnalogDevice)
    seq.declare_channel("ising", "rydberg_global")
    seq.add(adiabatic_pulse, "ising")

    simul = QutipEmulator.from_sequence(seq)
    results = simul.run()
    final = results.get_final_state()
    count_dict = results.sample_final_state()

    if draw:
        seq.draw()
        plot_distribution(count_dict)
    
    return count_dict

def plot_distribution(C):
    C = dict(sorted(C.items(), key=lambda item: item[1], reverse=True))
    indexes = ["01011", "00111"]  # QUBO solutions
    color_dict = {key: "r" if key in indexes else "g" for key in C}
    plt.figure(figsize=(12, 6))
    plt.xlabel("bitstrings")
    plt.ylabel("counts")
    plt.bar(C.keys(), C.values(), width=0.5, color=color_dict.values())
    plt.xticks(rotation="vertical")
    plt.show()

def get_highest_counts(counts:dict, n:int):
    """ 
    Parameters:
    counts (dict): dictionary {name, count} of counts

    Returns:
    list: list [name, dict] of the n highest counts. Dict is {count, proba}
    """
    total = sum(counts.values())
    highest_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]
    probas = [count/total for name, count in highest_counts]
    out = []
    for i in range(n):
        out.append([highest_counts[i][0], {"count": highest_counts[i][1], "proba": probas[i]}])
    return out

def draw_solutions(reg:Register, counts:dict, n:int):
    """
    Parameters:
    reg (Register): register of the circuit
    counts (dict): dictionary {name, count} of counts
    n (int): number of solutions to draw
    """
    highest_counts = get_highest_counts(counts, len(counts)) # sorted by count
    probas = [val[1]["proba"] for val in highest_counts]
    for ii in range(n):
        reg.draw(
            blockade_radius=DigitalAnalogDevice.rydberg_blockade_radius(1.0),
            draw_graph=False,
            draw_half_radius=True,
            qubit_colors = {f"q{i}": "red" for i, val in enumerate(highest_counts[ii][0]) if val == '1'},
            show=False
        )
        legend_elements = [
            Patch(facecolor='mistyrose', label='State |1⟩'),
            Patch(facecolor='lightgreen', label='State |0⟩')
        ]
        plt.legend(handles=legend_elements)
        plt.title(f'Probas: ' + ', '.join([f'${probas[i]:.2f}$' if i != ii else f'$\\underline{{{probas[i]:.2f}}}$' for i in range(min(n+2, len(probas)))]))
        plt.show()