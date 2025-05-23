"""High-level decoding routines for BPQM and classical benchmarks."""

from typing import Optional, Sequence, Union, Tuple, List

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.library import SaveProbabilitiesDict

import cvxpy as cp

from bpqm import tree_bpqm
from cloner import Cloner
from linearcode import LinearCode

def TP(exprs: Sequence[NDArray]) -> NDArray:
    """Return the Kronecker product of all matrices in ``exprs``."""
    out = exprs[0]
    for mat in exprs[1:]:
        out = np.kron(out, mat)
    return out

def decode_bpqm(
    code: LinearCode,
    theta: float,
    cloner: Cloner,
    height: int,
    mode: str,
    bit: Optional[int] = None,
    order: Optional[Sequence[int]] = None,
    only_zero_codeword: bool = True,
    debug: bool = False,
) -> float:
    """Decode either a single codeword using BPQM."""
    assert mode in ['bit','codeword'], "mode should be 'bit' or 'codeword'."
    if mode=='bit':
        assert bit!=None, "bit shouldn't be None when choosing mode'bit'."
        order=[bit]

    if order == None:
        order = list(range(code.n))

    # 1) build computation graphs
    cgraphs = [code.get_computation_graph(f"x{b}", height) for b in order]

    # 2) determine qubit counts
    n_data_qubits = max(sum(occ.values()) for _,occ,_ in cgraphs)
    n_data_qubits = max(n_data_qubits, code.n)
    n_qubits = n_data_qubits + len(order)-1

    # 3) generate main circuit
    meas_idx = 0
    qc = QuantumCircuit(n_qubits)
    for i,(graph,occ,root) in enumerate(cgraphs):
        # qubit mapping
        leaves = [n for n in graph.nodes() if graph.nodes[n]["type"]=="output"]
        leaves = sorted(leaves, key=lambda s:int(s.split("_")[1]))
        qm = {f"y{j}_0":j for j in range(code.n)}
        idx=code.n
        for node in leaves:
            if int(node.split("_")[1])>0:
                qm[node]=idx; idx+=1

        # annotate graph
        cloner.mark_angles(graph, occ)
        for node in leaves:
            graph.nodes[node]["qubit_idx"]=qm[node]

        # build BPQM + cloner
        qc_bpqm = QuantumCircuit(n_qubits)
        meas_idx, angles = tree_bpqm(graph, qc_bpqm, root=root)
        qc_cloner = cloner.generate_cloner_circuit(graph, occ, qm, n_qubits)

        # append & uncompute with compose
        qc.compose(qc_cloner, inplace=True); qc.barrier()
        qc.compose(qc_bpqm,   inplace=True); qc.barrier()
        if i < len(order)-1:
            qc.h(meas_idx); qc.cx(meas_idx, n_data_qubits+i); qc.h(meas_idx); qc.barrier()
            qc.compose(qc_bpqm.inverse(),   inplace=True); qc.barrier()
            qc.compose(qc_cloner.inverse(), inplace=True); qc.barrier()
        else:
            qc.h(meas_idx)

    # snapshot as dict
    cw_qubits = list(range(n_data_qubits, n_data_qubits+len(order)-1)) + [meas_idx]
    qc.append(SaveProbabilitiesDict(len(cw_qubits), label='prob'), cw_qubits)
    # qc.save_probabilities_dict(label='prob', qubits=cw_qubits)

    # simulate
    backend = AerSimulator(method='statevector')
    codewords = [[0]*code.n] if only_zero_codeword else code.get_codewords()
    prob=0.
    for cw in codewords:
        qc_init = QuantumCircuit(n_qubits)
        plus  = np.array([np.cos(0.5*theta),  np.sin(0.5*theta)])
        minus = np.array([np.cos(0.5*theta), -np.sin(0.5*theta)])
        for j,v in enumerate(cw):
            state = (plus if v == 0 else minus).tolist()           # ← convert here
            qc_init.initialize(state, [j])

        combined = qc_init.compose(qc)
        assert combined is not None, "Unexpected None for combined"

        full_qc   = transpile(combined, backend)
        result    = backend.run(full_qc).result()

        probs = result.data()['prob']
        key   = int("".join(str(cw[i]) for i in reversed(order)),2)
        prob += probs.get(key,0.0)/len(codewords)

    return prob

def create_init_qc(
    code: LinearCode,
    theta: float,
    codeword: Optional[Union[Sequence[int], np.ndarray]] = None,
    prior: Optional[float] = None
) -> QuantumCircuit:
    """
    Build the part of the circuit that encodes your classical bits
    into qubit states |Q(0,θ)> / |Q(1,θ)> or a prior‐weighted superposition.
    
    Parameters
    ----------
    n_data_qubits : int
        How many “data” qubits at the front correspond to the codeword bits.
    codeword : sequence of {0,1}
        The classical bitstring to embed. Length must be n_data_qubits.
    theta : float
        Angle θ (in radians) for amplitude encoding.
    prior : float, optional
        If given, all data‐qubits are initialized to prior⋅|Q(0)> + (1−prior)⋅|Q(1)>;
        otherwise each qubit is set to |Q(x,θ)> based on the codeword bit x.
    
    Returns
    -------
    QuantumCircuit
        A circuit of width n_total_qubits that prepares the data qubits.
    """
    qc_init = QuantumCircuit(code.n)
    
    # define the two pure‐state embeddings
    plus  = np.array([ np.cos(theta/2),  np.sin(theta/2) ])
    minus = np.array([ np.cos(theta/2), -np.sin(theta/2) ])
    plus  /= np.linalg.norm(plus)
    minus /= np.linalg.norm(minus)
    
    if prior is None:
        # bit‐wise initialization
        assert codeword is not None, "codeword must be provided when prior is None"
        for j, bit in enumerate(codeword):
            state = (plus if bit == 0 else minus).tolist()
            qc_init.initialize(state, [j])
    else:
        # uniform prior initialization
        mix = prior * plus + (1.0 - prior) * minus
        mix /= np.linalg.norm(mix)
        mix = mix.tolist()
        for j in range(code.n):
            qc_init.initialize(mix, [j])
    
    return qc_init

def decode_single_codeword(
    qc_init: QuantumCircuit,
    code: LinearCode,
    cloner: Cloner,
    height: int,
    order: Optional[Sequence[int]] = None,
    shots: int = 512,
    debug: bool = False,
    run_simulation: bool = True
) -> Tuple[Optional[np.ndarray], List[int], QuantumCircuit]:
    """
    Given a pre-built initialization circuit `qc_init`, append BPQM logic
    and measurement to decode a codeword under a pure‐state CQ channel.

    Parameters
    ----------
    qc_init : QuantumCircuit
        Initialization circuit produced by `create_init_qc(...)` of width = code.n.
    code : LinearCode
        Linear‐code object with H‐matrix and graph routines.
    cloner : Cloner
        Cloner instance for loopy-graph segments.
    height : int
        Unroll depth for the BPQM computation tree.
    order : sequence of ints, optional
        Bit decode order (default = all bits in [0..code.n-1]).
    shots : int, default=512
        Number of measurement shots.
    debug : bool, default=False
        If True, print sorted & reversed counts + syndrome.
    run_simulation : bool, default=True
        If False, return (None, decoded_qubits, qc_decode) without running.

    Returns
    -------
    decoded_bits : np.ndarray or None
        The decoded bits if `run_simulation=True`, otherwise None.
    decoded_qubits : List[int]
        The list of qubit indices measured in order.
    qc_decode : QuantumCircuit
        The BPQM decoding circuit (without initialization).
    """
    if order is None:
        order = list(range(code.n))

    # 1) Build all BPQM subcircuits
    cgraphs = [code.get_computation_graph(f"x{b}", height) for b in order]
    n_data_qubits = max(sum(occ.values()) for _, occ, _ in cgraphs)
    n_data_qubits = max(n_data_qubits, code.n)
    n_total_qubits = n_data_qubits + len(order) - 1

    qc_decode = QuantumCircuit(n_total_qubits)
    meas_idx = 0

    for i, (graph, occ, root) in enumerate(cgraphs):
        # map each “y” (output) node to a physical qubit
        leaves = sorted(
            [n for n in graph.nodes() if graph.nodes[n]["type"] == "output"],
            key=lambda s: int(s.split("_")[1])
        )
        qubit_map = {f"y{j}_0": j for j in range(code.n)}
        idx = code.n
        for leaf in leaves:
            if int(leaf.split("_")[1]) > 0:
                qubit_map[leaf] = idx
                idx += 1

        # mark angles & build the BPQM + cloner pieces
        cloner.mark_angles(graph, occ)
        for leaf in leaves:
            graph.nodes[leaf]["qubit_idx"] = qubit_map[leaf]
        qc_bpqm, _ = QuantumCircuit(n_total_qubits), None
        meas_idx, _ = tree_bpqm(graph, qc_bpqm, root=root)
        qc_cloner = cloner.generate_cloner_circuit(
            graph, occ, qubit_map, n_total_qubits
        )

        qc_decode.compose(qc_cloner, inplace=True)
        qc_decode.barrier()
        qc_decode.compose(qc_bpqm, inplace=True)
        qc_decode.barrier()

        if i < len(order) - 1:
            qc_decode.h(meas_idx)
            qc_decode.cx(meas_idx, n_data_qubits + i)
            qc_decode.h(meas_idx)
            qc_decode.barrier()
            qc_decode.compose(qc_bpqm.inverse(), inplace=True)
            qc_decode.barrier()
            qc_decode.compose(qc_cloner.inverse(), inplace=True)
            qc_decode.barrier()
        else:
            qc_decode.h(meas_idx)

    decoded_qubits = list(range(n_data_qubits,
                                n_data_qubits + len(order) - 1)) + [meas_idx]

    # If not running simulation, return placeholders
    if not run_simulation:
        return None, decoded_qubits, qc_decode

    # 2) Compose init + decode + measurements onto a full-width circuit
    full_qc = QuantumCircuit(n_total_qubits, len(order))
    full_qc.compose(
        qc_init,
        qubits=list(range(qc_init.num_qubits)),
        inplace=True
    )
    full_qc.compose(qc_decode, inplace=True)
    for idx, qb in enumerate(decoded_qubits):
        full_qc.measure(qb, idx)

    # 3) Run and post-process
    backend = AerSimulator()
    job     = backend.run(transpile(full_qc, backend), shots=shots)
    result  = job.result().get_counts()

    reversed_counts = {bits[::-1]: cnt for bits, cnt in result.items()}
    sorted_counts   = dict(sorted(
        reversed_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    if debug:
        print("Counts:")
        for bits, cnt in sorted_counts.items():
            syn_vec = (np.array([int(b) for b in bits]) @ code.H.T) % 2
            print(f"  {bits} → {cnt} : syndrome {syn_vec}")

    best = next(iter(sorted_counts))
    decoded_bits = np.array([int(b) for b in best], dtype=int)
    return decoded_bits, decoded_qubits, qc_decode

def decode_single_syndrome(
    qc_init: QuantumCircuit,
    code: LinearCode,
    cloner: Cloner,
    height: int,
    syndrome: Union[Sequence[int], np.ndarray],
    order: Optional[Sequence[int]] = None,
    shots: int = 512,
    debug: bool = False,
    run_simulation: bool = True
) -> Tuple[Optional[np.ndarray], list[int], QuantumCircuit]:
    """
    Given a pre-built `qc_init`, append BPQM logic conditioned on `syndrome`
    to produce a decode circuit and (optionally) run it.

    Parameters
    ----------
    qc_init : QuantumCircuit
        Initialization circuit of width = code.n.
    code : LinearCode
        The linear code defining the syndrome checks.
    cloner : Cloner
        Cloner instance for loopy-graph segments.
    height : int
        Unrolling depth for the BPQM tree.
    syndrome : Sequence[int] or np.ndarray
        Observed syndrome vector (length = n–k).
    order : Sequence[int], optional
        Bit indices defining decode order (default = all bits).
    shots : int, default=512
        Number of measurement shots.
    debug : bool, default=False
        If True, print reversed-and-sorted counts with their syndrome.
    run_simulation : bool, default=True
        If False, returns (None, decoded_qubits, qc_decode) without running.

    Returns
    -------
    decoded_bits : np.ndarray or None
        The decoded bit-string if `run_simulation=True`, otherwise None.
    decoded_qubits : list[int]
        The list of qubit indices that are measured (in order).
    qc_decode : QuantumCircuit
        The BPQM decoding circuit (without initialization).
    """
    if order is None:
        order = list(range(code.n))

    # Build conditioned BPQM subcircuits
    cgraphs = [
        code.get_computation_graph(f"x{b}", height, syndrome=syndrome)
        for b in order
    ]
    n_data_qubits = max(sum(occ.values()) for _, occ, _ in cgraphs)
    n_data_qubits = max(n_data_qubits, code.n)
    n_total_qubits = n_data_qubits + len(order) - 1

    # Build the BPQM decode circuit
    qc_decode = QuantumCircuit(n_total_qubits)
    meas_idx = 0
    for i, (graph, occ, root) in enumerate(cgraphs):
        leaves = sorted(
            [n for n in graph.nodes() if graph.nodes[n]["type"] == "output"],
            key=lambda s: int(s.split("_")[1])
        )
        qubit_map = {f"y{j}_0": j for j in range(code.n)}
        idx = code.n
        for leaf in leaves:
            if int(leaf.split("_")[1]) > 0:
                qubit_map[leaf] = idx
                idx += 1

        cloner.mark_angles(graph, occ)
        for leaf in leaves:
            graph.nodes[leaf]["qubit_idx"] = qubit_map[leaf]
        qc_bpqm   = QuantumCircuit(n_total_qubits)
        meas_idx, _ = tree_bpqm(graph, qc_bpqm, root=root)
        qc_cloner = cloner.generate_cloner_circuit(
            graph, occ, qubit_map, n_total_qubits
        )

        qc_decode.compose(qc_cloner, inplace=True)
        qc_decode.barrier()
        qc_decode.compose(qc_bpqm, inplace=True)
        qc_decode.barrier()

        if i < len(order) - 1:
            qc_decode.h(meas_idx)
            qc_decode.cx(meas_idx, n_data_qubits + i)
            qc_decode.h(meas_idx)
            qc_decode.barrier()
            qc_decode.compose(qc_bpqm.inverse(), inplace=True)
            qc_decode.barrier()
            qc_decode.compose(qc_cloner.inverse(), inplace=True)
            qc_decode.barrier()
        else:
            qc_decode.h(meas_idx)

    decoded_qubits = list(range(n_data_qubits, n_data_qubits + len(order) - 1)) + [meas_idx]

    # If not running simulation, return placeholders
    if not run_simulation:
        return None, decoded_qubits, qc_decode

    # Otherwise, compose init + decode + measurements and run
    full_qc = QuantumCircuit(n_total_qubits, len(order))
    full_qc.compose(qc_init, qubits=list(range(qc_init.num_qubits)), inplace=True)
    full_qc.compose(qc_decode, inplace=True)
    for idx, qb in enumerate(decoded_qubits):
        full_qc.measure(qb, idx)

    backend = AerSimulator()
    job     = backend.run(transpile(full_qc, backend), shots=shots)
    result  = job.result().get_counts()

    # Reverse bit-order and sort by count descending
    rev = {bits[::-1]: cnt for bits, cnt in result.items()}
    sorted_counts = dict(sorted(rev.items(), key=lambda x: x[1], reverse=True))

    if debug:
        print("Counts:")
        for bits, cnt in sorted_counts.items():
            syn_vec = (np.array([int(b) for b in bits]) @ code.H.T) % 2
            print(f"  {bits} → {cnt} : syndrome {syn_vec}")

    best = next(iter(sorted_counts))
    decoded_bits = np.array([int(b) for b in best], dtype=int)
    return decoded_bits, decoded_qubits, qc_decode

def decode_bit_optimal_quantum(code: LinearCode, theta: float, index: int) -> float:
    rho0 = np.zeros((2**code.n, 2**code.n), complex)
    rho1 = np.zeros((2**code.n, 2**code.n), complex)
    vecs = [
        np.array([[np.cos(0.5 * theta)], [np.sin(0.5 * theta)]]),
        np.array([[np.cos(0.5 * theta)], [-np.sin(0.5 * theta)]])
    ]
    codewords = code.get_codewords()

    for cw in [c for c in codewords if c[index] == '0']:
        psi = TP([vecs[int(b)] for b in cw])
        rho0 += psi @ psi.T / (0.5 * len(codewords))
    for cw in [c for c in codewords if c[index] == '1']:
        psi = TP([vecs[int(b)] for b in cw])
        rho1 += psi @ psi.T / (0.5 * len(codewords))

    eigs = np.linalg.eigvals(rho0 - rho1)
    return 0.5 + 0.25 * np.sum(np.abs(eigs))


def decode_codeword_PGM(code: LinearCode, theta: float) -> float:
    vecs = [
        np.array([[np.cos(0.5 * theta)], [np.sin(0.5 * theta)]]),
        np.array([[np.cos(0.5 * theta)], [-np.sin(0.5 * theta)]])
    ]
    codewords = code.get_codewords()

    rho = sum(
        (TP([vecs[int(b)] for b in cw]) @ TP([vecs[int(b)] for b in cw]).T)
        for cw in codewords
    ) / len(codewords)
    vals, vecs_mat = np.linalg.eig(rho)
    inv_sqrt = vecs_mat @ np.diag(vals**(-0.5)) @ np.linalg.inv(vecs_mat)

    return sum(
        abs((TP([vecs[int(b)] for b in cw]).T @ (inv_sqrt @ TP([vecs[int(b)] for b in cw])))).item()**2
        for cw in codewords
    ) / len(codewords)


def decode_codeword_optimal_quantum(code: LinearCode, theta: float) -> float:
    sigma = cp.Variable((2**code.n, 2**code.n), PSD=True)
    codewords = code.get_codewords()
    vecs = [
        np.array([[np.cos(0.5 * theta)], [np.sin(0.5 * theta)]]),
        np.array([[np.cos(0.5 * theta)], [-np.sin(0.5 * theta)]])
    ]

    constraints = []
    for cw in codewords:
        psi = TP([vecs[int(b)] for b in cw])
        constraints.append(sigma >> (psi @ psi.T) / len(codewords))

    prob = cp.Problem(cp.Minimize(cp.trace(sigma)), constraints).solve(solver=cp.SCS)
    return float(prob)


def decode_bit_optimal_classical(code: LinearCode, theta: float, index: int) -> float:
    if theta < 1e-8:
        return 0.5
    p_r = 0.5 * (1 + np.sin(theta))
    p_w = 0.5 * (1 - np.sin(theta))
    codewords = code.get_codewords()

    success = 0.0
    for m in range(2**code.n):
        y = list(map(float, bin(m)[2:].zfill(code.n)))
        def like(c):
            return (p_w**sum(abs(ci - yi) for ci, yi in zip(c, y)) *
                    p_r**sum(ci == yi for ci, yi in zip(c, y)))
        P0 = sum(like(c) for c in codewords if c[index] == '0')
        P1 = sum(like(c) for c in codewords if c[index] == '1')
        out = 0 if P0 > P1 else 1
        success += sum(like(c) for c in codewords if int(c[index]) == out) / len(codewords)
    return success


def decode_codeword_optimal_classical(code: LinearCode, theta: float) -> float:
    if theta < 1e-8:
        return 1.0 / (2**code.k)
    p_r = 0.5 * (1 + np.sin(theta))
    p_w = 0.5 * (1 - np.sin(theta))
    codewords = code.get_codewords()

    success = 0.0
    for m in range(2**code.n):
        y = list(map(float, bin(m)[2:].zfill(code.n)))
        best = max(
            codewords,
            key=lambda c: (p_w**sum(abs(ci - yi) for ci, yi in zip(c, y)) *
                           p_r**sum(ci == yi for ci, yi in zip(c, y)))
        )
        success += ((p_w**sum(abs(int(b)-yi) for b, yi in zip(best, y)) *
                     p_r**sum(int(b) == yi for b, yi in zip(best, y))) /
                    len(codewords))
    return success

