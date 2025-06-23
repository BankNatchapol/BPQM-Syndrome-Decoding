"""
Cirq version of BPQM decoders.
"""

# TODO: Implement Cirq-based BPQM decoders here

import cirq
from collections import Counter
# Example stub function
def cirq_decode(*args, **kwargs):
    raise NotImplementedError("Cirq BPQM decoder not yet implemented.")

"""BPQM decoders for Cirq implementation."""

from typing import Optional, Sequence, Tuple, List, Dict

import numpy as np
from numpy.typing import NDArray
import cirq

try:
    from .bpqm import tree_bpqm_cirq
    from .cloner import CirqCloner, CirqExtendedVarNodeCloner
    from .linearcode import CirqLinearCode
except ImportError:
    from bpqm import tree_bpqm_cirq
    from cloner import CirqCloner, CirqExtendedVarNodeCloner
    from linearcode import CirqLinearCode


def create_init_qc(
    code: CirqLinearCode,
    theta: float,
    codeword: Optional[NDArray[np.int_]] = None,
    prior: Optional[float] = None
) -> Tuple[cirq.Circuit, List[cirq.Qid]]:
    """
    Prepare an initialization circuit for data qubits - Cirq version.

    Parameters
    ----------
    code : CirqLinearCode
        Linear code defining number of qubits (code.n).
    theta : float
        Rotation angle for RY gates.
    codeword : Optional[NDArray[np.int_]]
        Binary array of length code.n; required if prior is None.
    prior : Optional[float]
        Probability weight for |0⟩ in the prior mixture.

    Returns
    -------
    circuit : cirq.Circuit
        Circuit initializing `code.n` qubits.
    qubits : List[cirq.Qid]
        List of qubits used in the circuit.
    """
    n = code.n
    qubits = [cirq.LineQubit(i) for i in range(n)]
    circuit = cirq.Circuit()

    if prior is None:
        if codeword is None:
            raise ValueError("codeword must be provided when prior is None")
        for j, bit in enumerate(codeword):
            angle = theta if bit == 0 else -theta
            circuit.append(cirq.ry(angle)(qubits[j]))
    else:
        # Build mixture amplitudes [a, b]
        mix = np.array([
            prior * np.cos(theta / 2) + (1 - prior) * np.cos(theta / 2),
            prior * np.sin(theta / 2) - (1 - prior) * np.sin(theta / 2)
        ], dtype=float)
        mix /= np.linalg.norm(mix)
        theta_mix = 2 * np.arctan2(mix[1], mix[0])
        for j in range(n):
            circuit.append(cirq.ry(theta_mix)(qubits[j]))
            if mix[1] < 0:
                circuit.append(cirq.Z(qubits[j]))

    return circuit, qubits


def decode_single_syndrome(
    syndrome_qc: cirq.Circuit,
    code: CirqLinearCode,
    theta: float,
    prior: Optional[float] = None,
    height: int = 2,
    shots: int = 512,
    debug: bool = False,
    run_simulation: bool = True
) -> Tuple[Optional[np.ndarray], Optional[List[int]], cirq.Circuit]:
    """
    Perform syndrome decoding via BPQM - Cirq version.

    Parameters
    ----------
    syndrome_qc : cirq.Circuit
        Circuit that prepares syndrome ancilla bits.
    code : CirqLinearCode
        Linear code describing parity checks.
    theta : float
        Rotation angle for initializing data qubits.
    prior : Optional[float], default=None
        Prior probability for the |0> state.
    height : int, default=2
        Depth of unrolling for the BPQM factor-tree.
    shots : int, default=512
        Number of measurement shots when running the simulation.
    debug : bool, default=False
        If True, print measurement counts and syndrome diagnostics.
    run_simulation : bool, default=True
        If False, return the decoding circuit without execution.

    Returns
    -------
    decoded_bits : Optional[np.ndarray]
        The most likely decoded bitstring, or None if not simulated.
    decoded_qubits : Optional[List[int]]
        List of qubit indices for the decoded bits, or None.
    qc_decode : cirq.Circuit
        The constructed BPQM decoding circuit.
    """
    # Determine ordering and build computation graphs
    order = list(range(code.n))
    computation_graphs = [
        code.get_computation_graph(f"x{b}", height, syndrome_mode=True)
        for b in order
    ]

    # Determine qubit counts - match Qiskit logic exactly
    n_ancilla = code.H.shape[0]
    max_data = max(sum(occ.values()) for _, occ, _ in computation_graphs)
    n_data = max(max_data, code.n)
    total_qubits = n_ancilla + n_data + len(order)

    # Create qubits
    all_qubits = [cirq.LineQubit(i) for i in range(total_qubits)]
    ancilla_qubits = all_qubits[:n_ancilla]
    data_qubits = all_qubits[n_ancilla:n_ancilla + n_data]
    output_qubits = all_qubits[n_ancilla + n_data:]

    # Initialize decode circuit
    qc_decode = cirq.Circuit()
    # Ensure all qubits appear by adding identity gates
    qc_decode.append(cirq.I(q) for q in all_qubits)
    
    # Create a temporary code for initialization - match Qiskit approach
    temp_code = CirqLinearCode(None, np.zeros((0, n_data), int))
    temp_code.n = n_data
    
    # If prior is None, use all-zero codeword like Qiskit
    if prior is None:
        codeword = np.zeros(n_data, dtype=int)
    else:
        codeword = None
    
    # Create data initialization circuit using the helper function
    data_init, data_init_qubits = create_init_qc(
        code=temp_code,
        theta=theta,
        codeword=codeword,
        prior=prior
    )
    
    # Apply data initialization to decode circuit with proper qubit mapping
    # Map the initialization qubits to our data qubits
    data_init_circuit = cirq.Circuit()
    # for q in range(total_qubits):
    #     data_init_circuit.append(cirq.I(cirq.LineQubit(q)))
    
    qubit_map = {data_init_qubits[i]: data_qubits[i] for i in range(len(data_init_qubits))}
    for moment in data_init:
        mapped_ops = []
        for op in moment:
            mapped_qubits = [qubit_map.get(q, q) for q in op.qubits]
            mapped_ops.append(op.gate(*mapped_qubits))
        data_init_circuit.append(cirq.Moment(mapped_ops))
    qc_decode.append(data_init_circuit)
    
    # Build and apply BPQM for each logical qubit
    for idx, (graph, occ, root) in enumerate(computation_graphs):
        
    
        # Map output nodes to qubit indices - match Qiskit mapping exactly
        leaves = [n for n, d in graph.nodes(data=True) if d.get("type") == "output"]
        qubit_map = {f"y{j}_0": j for j in range(code.n)}
        next_idx = code.n
        for leaf in leaves:
            level = int(leaf.split("_")[1])
            if level > 0:
                qubit_map[leaf] = next_idx
                next_idx += 1

        # Annotate output angles and qubit indices
        for leaf in leaves:
            count = occ[leaf.split("_")[0].replace("y", "x")]
            angle = np.arccos(np.cos(theta) ** (1.0 / count))
            graph.nodes[leaf]["angle"] = [(angle, {})]
            graph.nodes[leaf]["qubit_idx"] = qubit_map[leaf]

        # Remove unused check nodes
        to_remove = [n for n, d in graph.nodes(data=True)
                     if d.get("type") == "check" and graph.out_degree(n) == 0]
        graph.remove_nodes_from(to_remove)

        # Construct BPQM subcircuit - use all_qubits with offset like Qiskit
        qc_bpqm = cirq.Circuit()
        # qc_bpqm.append(cirq.I(q) for q in all_qubits)
        meas_idx, _ = tree_bpqm_cirq(graph, qc_bpqm, root=root, offset=n_ancilla)
        # print(qc_bpqm)
        qc_decode.append(qc_bpqm)

        # Entangling measurement and uncompute - match Qiskit exactly
        meas_qubit = all_qubits[meas_idx]
        target_qubit = output_qubits[idx]
        
        qc_decode.append(cirq.H(meas_qubit))
        qc_decode.append(cirq.CNOT(meas_qubit, target_qubit))
        qc_decode.append(cirq.H(meas_qubit))
        
        # Uncompute BPQM
        qc_decode.append(cirq.inverse(qc_bpqm))

    # Uncompute data initialization
    qc_decode.append(cirq.inverse(data_init_circuit))

    # Return circuit BEFORE simulation to preserve UCRY blocks
    if not run_simulation:
        return None, None, qc_decode

    # Create a copy of the circuit for simulation (this will decompose UCRY)
    sim_circuit = qc_decode.copy()
    
    # Prepare full circuit with syndrome prep and measurement
    full_circuit = cirq.Circuit()
    # Ensure all qubits appear by adding identity gates
    # full_circuit.append(cirq.I(q) for q in all_qubits)
    
    # Add syndrome preparation to ancilla qubits
    full_circuit.append(syndrome_qc)
    
    # Add decoding circuit (this will decompose UCRY blocks)
    full_circuit.append(sim_circuit)
    
    # Add measurements
    for i, qb in enumerate(output_qubits):
        full_circuit.append(cirq.measure(qb, key=f'm{i}'))

    # Execute on simulator
    simulator = cirq.Simulator()
    result = simulator.run(full_circuit, repetitions=shots)

    # Process results - match Qiskit bit reversal
    combined = np.hstack([result.measurements[f'm{i}'] for i in range(len(order))])

    # Convert to bitstrings (Qiskit-style: qubit0 is leftmost)
    bitstrings = [''.join(str(bit) for bit in row) for row in combined]

    # Get counts
    counts: Dict[str, int] = {}
    counts = dict(Counter(bitstrings))

    sorted_counts = dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))

    if debug:
        print("Counts:")
        for bits, cnt in sorted_counts.items():
            syn = (np.array([int(b) for b in bits]) @ code.H.T) % 2
            print(f"  {bits} → {cnt} : syndrome {syn}")

    best = next(iter(sorted_counts))
    decoded_bits = np.array([int(b) for b in best], dtype=int)
    decoded_qubits = [qb.x for qb in output_qubits]
    
    return decoded_bits, decoded_qubits, qc_decode 