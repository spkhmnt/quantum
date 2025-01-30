import numpy as np
from math import gcd
from fractions import Fraction

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler

print("Imports Successful")

service = QiskitRuntimeService(
    channel="ibm_quantum",
    token='YOUR_FREE_TOKEN_FROM_IBM_GOES_HERE'
)
print("QiskitRuntimeService loaded and authenticated!")

N = 15
a = 7
N_COUNT = 8  # number of counting qubits


#brute force each period
def c_amod15(a, power):
    if a not in [2,4,7,8,11,13]:
        raise ValueError("'a' must be 2,4,7,8,11,13.")
    qc = QuantumCircuit(4)
    for _iteration in range(power):
        if a in [2, 13]:
            qc.swap(2,3); qc.swap(1,2); qc.swap(0,1)
        if a in [7, 8]:
            qc.swap(0,1); qc.swap(1,2); qc.swap(2,3)
        if a in [4, 11]:
            qc.swap(1,3); qc.swap(0,2)
        if a in [7, 11, 13]:
            for q in range(4):
                qc.x(q)
    gate = qc.to_gate()
    gate.name = f"{a}^{power} mod 15"
    return gate.control(1)

#reasonable implementation
def qft_dagger(n):
    qc = QuantumCircuit(n)
    # Swap qubits
    for qubit in range(n // 2):
        qc.swap(qubit, n - qubit - 1)
    # Apply controlled-phase and Hadamard
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi / 2**(j - m), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

qc = QuantumCircuit(N_COUNT + 4, N_COUNT)

for i in range(N_COUNT):
    qc.h(i)

qc.x(N_COUNT)

for i in range(N_COUNT):
    qc.append(c_amod15(a, 2**i), [i] + list(range(N_COUNT, N_COUNT + 4)))

qc.append(qft_dagger(N_COUNT), range(N_COUNT))
qc.measure(range(N_COUNT), range(N_COUNT))
qc.name = "Shor_Example"



backend = service.least_busy(simulator=False)
print("Using backend:", backend.name)
transpiled_circ = transpile(qc, backend=backend)
sampler = Sampler(mode=backend)
print("Sampler is ready.")
job = sampler.run([transpiled_circ], shots=4096)

print(f"Submitted job. Job ID: {job.job_id()}")
result = job.result()


pub_result = result[0]  # The single-circuit result
counts_dict = pub_result.data.c.get_counts()  # 'c' is the auto-named classical register
shots_used = sum(counts_dict.values())
counts_prob = {bitstring: c / shots_used for bitstring, c in counts_dict.items()}
print("\n===== Measurement Results (Top 10) =====")
sorted_counts = sorted(counts_prob.items(), key=lambda x: x[1], reverse=True)[:10]
for (bitstring, prob) in sorted_counts:
    print(f"  {bitstring}: p={prob:.4f}")

phases = []
for bitstring, prob in counts_prob.items():
    if prob > 0:
        decimal = int(bitstring, 2)
        phase = decimal / (2**N_COUNT)
        phases.append((bitstring, phase, prob))

phases.sort(key=lambda x: x[2], reverse=True)
best_bitstring, best_phase, best_prob = phases[0]
print(f"\nMost likely outcome: {best_bitstring} with p={best_prob:.4f}")
print(f"Corresponding phase (decimal/2^N_COUNT) = {best_phase}")

frac = Fraction(best_phase).limit_denominator(N)
r = frac.denominator
s = frac.numerator
print(f"Fraction from phase = {s}/{r}")

guess1 = gcd(a**(r//2) - 1, N)
guess2 = gcd(a**(r//2) + 1, N)
print("\nPotential factors from guess:")
print(f"  gcd({a}^({r//2}) - 1, {N}) = {guess1}")
print(f"  gcd({a}^({r//2}) + 1, {N}) = {guess2}")
print("\nDone!")
