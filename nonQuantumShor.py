import numpy as np
from math import gcd
from fractions import Fraction
import pandas as pd
from tabulate import tabulate

from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
from qiskit.circuit.library import QFT

print("Imports Successful")

service = QiskitRuntimeService(
        channel="ibm_cloud",
        token=,
        )


print("QiskitRuntimeService loaded and authenticated!")

# Test number(s) to factor
NUMBERS_TO_TEST = [77, 1591, 2021, 3127, 4633, 6149]

# Previously tested numbers
# [77, 1591, 2021, 3127, 4633, 6149, 99993, 143571]
# Period finding fails here: 1048574
# Largest possible semiprime number given current qubit allocation and IBM hardware: 2,199,013,170,649

# This period finding is CLASSICAL. In a true Shor's implementation, we would do this on QC.

def find_period(N, num_bits):
    """Find a value with a valid period modulo N and its transitions"""
    best_a = None
    best_period = float('inf')
    
    print(f"\nSearching for values with good periods for N={N}...")
    
    for a in range(2, N-1):
        if gcd(a, N) != 1: 
            continue
            
        x = a
        for r in range(1, N):
            if x == 1: 
                if r % 2 == 0 and (r//2) % 2 == 1: 
                    if r < best_period:
                        best_a = a
                        best_period = r
                break
            x = (x * a) % N
    
    if best_a:
        print(f"\nUsing best value: a={best_a} with period {best_period}")
        
        transitions = {}
        x = 1
        for _ in range(best_period):
            next_x = (x * best_a) % N
            transitions[x] = next_x
            print(f"Transition: {x} → {next_x}")
            x = next_x
            
        return best_a, transitions
        
    return None, None

# this is a brute force method which will not work for large N.
def c_amod_N(a, N, num_bits, transitions, power):
    """Controlled multiplication by a mod N raised to specified power"""
    qc = QuantumCircuit(num_bits + 1)
    
    # Convert transitions to bit patterns
    bit_transitions = {}
    for input_val, output_val in transitions.items():
        input_pattern = bin(input_val)[2:].zfill(num_bits)
        output_pattern = bin(output_val)[2:].zfill(num_bits)
        bit_transitions[input_pattern] = output_pattern
    
    # Create the transformation circuit using full bit patterns
    for input_pattern in bit_transitions:
        for i, bit in enumerate(input_pattern):
            if bit == '0':
                qc.x(i + 1)
                
        output_pattern = bit_transitions[input_pattern]
        
        for i in range(num_bits):
            if input_pattern[i] != output_pattern[i]:
                qc.cx(0, i + 1)
                
        for i, bit in enumerate(input_pattern):
            if bit == '0':
                qc.x(i + 1)
    
    base_gate = qc.to_gate()
    base_gate.name = f"{a}^{power} mod {N}"
    return base_gate

def qft_dagger(n):
    """Inverse quantum Fourier transform"""
    qc = QuantumCircuit(n)
    
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
        
    qc.name = "QFT†"
    return qc

def run_shors_experiment(N):
    num_bits = len(bin(N)[2:])
    N_COUNT = max(12, 2 * num_bits)
    print(f"Number {N} requires {num_bits} qubits to represent")
    print(f"Using {N_COUNT} counting qubits based on input size")
    
    a, transitions = find_period(N, num_bits)
    if a is None:
        print(f"No period-2 value found for {N}")
        return None
        
    print(f"Using N = {N} and a = {a}")
    
    qc = QuantumCircuit(N_COUNT + num_bits, N_COUNT)
    
    for i in range(N_COUNT):
        qc.h(i)
    
    qc.x(N_COUNT)
    
    for i in range(N_COUNT):
        qc.append(c_amod_N(a, N, num_bits, transitions, 2**i), 
                 [i] + list(range(N_COUNT, N_COUNT + num_bits)))
    
    qc.append(qft_dagger(N_COUNT), range(N_COUNT))
    
    qc.measure(range(N_COUNT), range(N_COUNT))
    
    qc.name = f"Shor{N}"
    backend = service.least_busy(simulator=False)
    print("Using backend:", backend.name)
    
    transpiled_circ = transpile(qc, backend=backend)
    sampler = Sampler(mode=backend)
    print("Sampler is ready.")
    
    job = sampler.run([transpiled_circ], shots=4096)
    print(f"Submitted job. Job ID: {job.job_id()}")
    
    result = job.result()
    pub_result = result[0]
    counts_dict = pub_result.data.c.get_counts()
    shots_used = sum(counts_dict.values())
    counts_prob = {bitstring: c / shots_used for bitstring, c in counts_dict.items()}
    
    # Calculate metrics
    total_success_prob = 0
    successful_states = []
    detailed_results = []
    
    print("\n===== Top 10 Measurement Results =====")
    sorted_counts = sorted(counts_prob.items(), key=lambda x: x[1], reverse=True)[:10]
    
    for bitstring, prob in counts_prob.items():
        decimal = int(bitstring, 2)
        phase = decimal / (2**N_COUNT)
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        
        # Store detailed results for CSV
        result_entry = {
            'N': N,
            'bitstring': bitstring,
            'decimal': decimal,
            'probability': prob,
            'phase': phase,
            'period': r,
            'backend': backend.name
        }
        
        if (r//2) % 2 != 0:
            guess1 = gcd(a**(r//2) - 1, N)
            guess2 = gcd(a**(r//2) + 1, N)
            if guess1 not in [1, N] or guess2 not in [1, N]:
                total_success_prob += prob
                successful_states.append(bitstring)
                result_entry['success'] = True
                result_entry['factor1'] = min(guess1, guess2)
                result_entry['factor2'] = max(guess1, guess2)
            else:
                result_entry['success'] = False
                result_entry['factor1'] = None
                result_entry['factor2'] = None
        else:
            result_entry['success'] = False
            result_entry['factor1'] = None
            result_entry['factor2'] = None
            
        detailed_results.append(result_entry)
    
    # Print top 10 results
    for i, (bitstring, prob) in enumerate(sorted_counts, 1):
        decimal = int(bitstring, 2)
        phase = decimal / (2**N_COUNT)
        print(f"\n{i}. State |{bitstring}⟩ ({decimal}):")
        print(f"   Probability: {prob:.4f}")
        print(f"   Phase: {phase:.4f}")
        
        frac = Fraction(phase).limit_denominator(N)
        r = frac.denominator
        s = frac.numerator
        print(f"   Fraction: {s}/{r}")
        
        if (r//2) % 2 != 0:
            guess1 = gcd(a**(r//2) - 1, N)
            guess2 = gcd(a**(r//2) + 1, N)
            print(f"   Period r={r}:")
            print(f"    - gcd({a}^({r//2}) - 1, {N}) = {guess1}")
            print(f"    - gcd({a}^({r//2}) + 1, {N}) = {guess2}")
            if guess1 not in [1, N] or guess2 not in [1, N]:
                print(f"   Found factors! {guess1} and {guess2}")
                print(f"   This state contributes {prob:.4f} to success probability")
        print("   " + "-" * 40)
    
    print(f"\nTotal probability of success: {total_success_prob:.4f}")
    print(f"Number of successful states: {len(successful_states)}")
    
    summary = {
        'N': N,
        'success_probability': total_success_prob,
        'num_successful_states': len(successful_states),
        'total_states': len(counts_prob),
        'num_qubits': num_bits + N_COUNT,
        'backend_name': backend.name
    }
    
    return summary, detailed_results

# Run experiments and collect results
summaries = []
all_detailed_results = []

for N in NUMBERS_TO_TEST:
    print(f"\n{'='*50}")
    print(f"Running experiment for N = {N}")
    print(f"{'='*50}")
    
    result = run_shors_experiment(N)
    if result:
        summary, detailed_results = result
        summaries.append(summary)
        all_detailed_results.extend(detailed_results)
    else:
        print(f"Failed to find valid period for N = {N}")

# Create summary DataFrame
df_summary = pd.DataFrame(summaries)

# Create detailed results DataFrame
df_detailed = pd.DataFrame(all_detailed_results)

# Save both to CSV
df_summary.to_csv('shors_summary.csv', index=False)
df_detailed.to_csv('shors_detailed.csv', index=False)

# Print ASCII bar chart of success probabilities
print("\n=== Success Probability Distribution ===")
max_prob = max(summary['success_probability'] for summary in summaries)
chart_width = 50

for summary in summaries:
    prob = summary['success_probability']
    bar_length = int((prob / max_prob) * chart_width)
    bar = '█' * bar_length
    print(f"N={summary['N']:5d} | {bar} {prob:.4f}")

# Print summary table
print("\n=== Experimental Summary ===")
print(tabulate(df_summary, headers='keys', tablefmt='grid', floatfmt='.4f'))

print("\nResults saved to:")
print("- shors_summary.csv (experiment summaries)")
print("- shors_detailed.csv (detailed measurement results)")

print("\nDone!")
