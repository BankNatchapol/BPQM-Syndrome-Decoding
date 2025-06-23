"""
Cirq version of BPQM main logic.
"""

# TODO: Implement Cirq-based BPQM logic here

import cirq
import numpy as np
import networkx as nx
from typing import List, Tuple, Dict, Optional

import math
class UCRY(cirq.Gate):
    """Uniformly controlled RY rotation—for k controls inferred from angles length."""

    def __init__(self, angles: List[float], num_controls: int = 0):
        """
        Initialize UCRY gate.
        
        Args:
            angles: List of angles for each control configuration
            num_controls: Number of control qubits
        """
        self.angles = angles
        self.num_controls = num_controls

    def _num_qubits_(self) -> int:
        return self.num_controls + 1

    def _decompose_(self, qubits):
        ctrls = qubits[: self.num_controls]
        tgt = qubits[self.num_controls]
        # No controls: single RY
        if self.num_controls == 0:
            theta = self.angles[0]
            if not np.isclose(theta, 0):
                yield cirq.ry(theta).on(tgt)
            return

        # Decompose for each control pattern
        for i, theta in enumerate(self.angles):
            if theta is None or np.isclose(theta, 0):
                continue
            bits = format(i, f'0{self.num_controls}b')
            # Flip controls where bit == '0'
            for idx, b in enumerate(bits):
                if b == '0':
                    yield cirq.X(ctrls[idx])
            # Apply multi-controlled RY
            yield cirq.ControlledGate(cirq.ry(theta), num_controls=self.num_controls)(*ctrls, tgt)
            # Uncompute flips
            for idx, b in enumerate(bits):
                if b == '0':
                    yield cirq.X(ctrls[idx])

    def _inverse_(self):
        return UCRY([-theta if theta is not None else None for theta in self.angles])

    def _circuit_diagram_info_(self, args):
        symbols = [f"c{j}" for j in range(self.num_controls)] + ["RY"]
        return cirq.CircuitDiagramInfo(wire_symbols=symbols)

    def _value_equality_values_(self):
        return (self.angles,)

    def __repr__(self):
        return f"UCRY(angles={list(self.angles)})"
    __str__ = __repr__


class CirqBPQM:
    def __init__(self):
        pass
    # Add Cirq BPQM methods here 

"""Utilities to build BPQM subcircuits - Cirq version."""

def combine_variable_cirq(
    circuit: cirq.Circuit,
    idx1: int,
    angles1: List[Tuple[float, Dict[int, int]]],
    idx2: int,
    angles2: List[Tuple[float, Dict[int, int]]]
) -> Tuple[int, List[Tuple[float, Dict[int, int]]]]:
    """
    Combine two variable nodes of the computation tree - Cirq version.
    This is a direct port of the Qiskit implementation.
    """
    # Accumulate output angles for the merged subtree
    angles_out: List[Tuple[float, Dict[int, int]]] = []

    # 1) Gather all original control-qubit indices, preserving order like Qiskit.
    ctrl_orig = []
    if angles1:
        ctrl_orig.extend(angles1[0][1].keys())
    if angles2:
        ctrl_orig.extend(angles2[0][1].keys())
    # unique preserve order
    ctrl_orig = [bit for bit in dict.fromkeys(ctrl_orig) if bit != idx2]
    # maintain original control ordering
    control_qubits = [bit for bit in ctrl_orig]

    # 2) Prepare lookup arrays
    n_ctrl = len(control_qubits)
    angles_alpha = [None] * (2**n_ctrl)
    angles_beta  = [None] * (2**n_ctrl)

    # 3) Compute α/β for each conditioning
    for t1, c1 in angles1:
        for t2, c2 in angles2:
            # merge control mappings
            orig_controls = {**c1, **c2}
            controls = {bit: val for bit, val in orig_controls.items()}
            angles_out.append((np.arccos(np.cos(t1)*np.cos(t2)), controls))

            # index into multiplex array using original order
            idx_bin = 0
            for bit in ctrl_orig:
                idx_bin = (idx_bin << 1) | orig_controls.get(bit, 0)

            a_min = (
                np.cos(0.5*(t1-t2)) - np.cos(0.5*(t1+t2))
            ) / (np.sqrt(2)*np.sqrt(1 + np.cos(t1)*np.cos(t2)))
            b_min = (
                np.sin(0.5*(t1+t2)) + np.sin(0.5*(t1-t2))
            ) / (np.sqrt(2)*np.sqrt(1 - np.cos(t1)*np.cos(t2)))
            alpha = np.arccos(-a_min) + np.arccos(-b_min)
            beta  = np.arccos(-a_min) - np.arccos(-b_min)

            angles_alpha[idx_bin] = alpha
            angles_beta[idx_bin]  = beta

    # 4) Variable-node gadget
    circuit.append(cirq.CNOT(cirq.LineQubit(idx2), cirq.LineQubit(idx1)))
    circuit.append(cirq.X(cirq.LineQubit(idx1)))
    circuit.append(cirq.CNOT(cirq.LineQubit(idx1), cirq.LineQubit(idx2)))
    circuit.append(cirq.X(cirq.LineQubit(idx1)))

    # 5) Reverse controls to match Qiskit's UCRYGate ordering convention.
    reversed_ctrls = list(control_qubits)
    
    # 6) Append uniformly-controlled Ry's
    # Our Cirq UCRY is controls + [target].
    # We must match the qiskit control order, which is reversed.
    ucry_alpha_qubits = [cirq.LineQubit(i) for i in reversed_ctrls] + [cirq.LineQubit(idx2)] 
    ucry_beta_qubits = [cirq.LineQubit(i) for i in reversed_ctrls] + [cirq.LineQubit(idx2)]
    
    # 7) Append uniformly-controlled Ry's
    ucry_alpha = UCRY(angles_alpha, num_controls=len(control_qubits))
    circuit.append(ucry_alpha(*ucry_alpha_qubits))
    
    circuit.append(cirq.CNOT(cirq.LineQubit(idx1), cirq.LineQubit(idx2)))

    ucry_beta = UCRY(angles_beta, num_controls=len(control_qubits))
    circuit.append(ucry_beta(*ucry_beta_qubits))
    
    circuit.append(cirq.CNOT(cirq.LineQubit(idx1), cirq.LineQubit(idx2)))

    return idx1, angles_out


def combine_check_cirq(
    circuit: cirq.Circuit,
    idx1: int,
    angles1: List[Tuple[float, Dict[int, int]]],
    idx2: int,
    angles2: List[Tuple[float, Dict[int, int]]],
    check_id: int
) -> Tuple[int, List[Tuple[float, Dict[int, int]]]]:
    """
    Combine two check nodes in a BPQM circuit - Cirq version.
    This is a direct port of the Qiskit implementation.
    """
    angles_out: List[Tuple[float, Dict[int, int]]] = []

    # Apply gates - match Qiskit logic
    if check_id is not None:
        circuit.append(cirq.CZ(cirq.LineQubit(check_id), cirq.LineQubit(idx1)))
    circuit.append(cirq.CNOT(cirq.LineQubit(idx1), cirq.LineQubit(idx2)))

    # Combine angles with branching - match Qiskit logic
    for t1, c1 in angles1:
        for t2, c2 in angles2:
            orig_controls = {**c1, **c2}

            # Branch outputs - handle potential division by zero
            tout_0 = np.arccos((np.cos(t1) + np.cos(t2)) / (1. + np.cos(t1)*np.cos(t2)))
            tout_1 = np.arccos((np.cos(t1) - np.cos(t2)) / (1. - np.cos(t1)*np.cos(t2)))

            # map controls depending on the branch outcome
            ctrl0 = {bit: val for bit, val in orig_controls.items()}
            ctrl0[idx2] = 0
            ctrl1 = {bit: val for bit, val in orig_controls.items()}
            ctrl1[idx2] = 1

            angles_out.append((tout_0, ctrl0))
            angles_out.append((tout_1, ctrl1))

    return idx1, angles_out


def tree_bpqm_cirq(
    tree: nx.DiGraph,
    circuit: cirq.Circuit,
    root: str,
    offset: int = 0
) -> Tuple[int, List[Tuple[float, Dict[int, int]]]]:
    """Recursively build a BPQM circuit for ``tree`` rooted at ``root``, applying ``offset`` to all indices."""
    succs = list(tree.successors(root))
    # leaf
    if not succs:
        leaf_idx = tree.nodes[root]["qubit_idx"] + offset
        return leaf_idx, tree.nodes[root]["angle"]
    # single child
    if len(succs) == 1:
        idx_child, angles_child = tree_bpqm_cirq(tree, circuit, succs[0], offset=offset)
        if tree.nodes[root]["type"] == "check":
            check_id = tree.nodes[root].get("check_idx")
            if check_id is not None:
                circuit.append(cirq.CZ(cirq.LineQubit(check_id), cirq.LineQubit(idx_child)))
        return idx_child, angles_child

    # combine children
    idx, angles = tree_bpqm_cirq(tree, circuit, succs[0], offset=offset)
    for child in succs[1:]:
        idx2, angles2 = tree_bpqm_cirq(tree, circuit, child, offset=offset)
        ntype = tree.nodes[root]["type"]
        if ntype == "variable":
            if idx == idx2: # temporary fix
                continue
            idx, angles = combine_variable_cirq(circuit, idx, angles, idx2, angles2)
        elif ntype == "check":
            check_id = tree.nodes[root].get("check_idx")
            idx, angles = combine_check_cirq(circuit, idx, angles, idx2, angles2, check_id)
        else:
            raise ValueError(f"Unknown node type '{ntype}'")

    return idx, angles