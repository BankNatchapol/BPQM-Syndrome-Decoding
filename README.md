# Belief Propagation with Quantum Messages (BPQM)

This repository re-implements the BPQM algorithm from the paper **“Quantum message-passing algorithm for optimal and efficient decoding”** by Christophe Piveteau and Joe Renes ([arXiv:2109.08170](https://arxiv.org/abs/2109.08170)).
Forked from [https://github.com/ChriPiv/quantum-message-passing-paper](https://github.com/ChriPiv/quantum-message-passing-paper) and updated for modern Python—and whichever Qiskit version is currently not obsolete (it seems to break its API by the hour).

## Setup

1. **Python version**: 3.10
2. Install dependencies:

   ```bash
   pip install numpy==2.2.5 scipy==1.15.3
   pip install pylatexenc==2.10
   pip install -r requirements.txt
   ```

## Example Usage

```python
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from decoders import LinearCode, VarNodeCloner, decode_bpqm, decode_single_codeword

# Define an 8-bit code (see Section 6 of the paper)
G = np.array([
    [1, 0, 0, 0, 1, 0, 0, 1],
    [0, 1, 0, 0, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 0],
    [0, 0, 0, 1, 0, 0, 1, 1],
])
H = np.array([
    [1, 1, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0, 1, 0],
    [1, 0, 0, 1, 0, 0, 0, 1],
])

code = LinearCode(G, H)

# Draw the factor graph
nx.draw(code.get_factor_graph(), with_labels=True)
plt.show()

# Set channel parameter and cloner
theta  = 0.2 * np.pi
cloner = VarNodeCloner(theta)  # ENU cloner

# Decode a single bit (bit index 4) with unrolling depth 2
p_bit = decode_bpqm(
    code,
    theta,
    cloner=cloner,
    height=2,
    mode='bit',
    bit=4,
    only_zero_codeword=True,
    debug=False
)
print("Success probability for bit 4:", p_bit)

# Decode the full codeword (bits [0,1,2,3]) with unrolling depth 2
p_codeword = decode_bpqm(
    code,
    theta,
    cloner=cloner,
    height=2,
    mode='codeword',
    order=[0,1,2,3],
    only_zero_codeword=True,
    debug=False
)
print("Success probability for the full codeword:", p_codeword)

# Decode a specific codeword and obtain the measurement outcome
decoded = decode_single_codeword(
    code,
    theta,
    cloner=cloner,
    height=2,
    codeword=[0]*code.n,
)
print("Decoded bits:", decoded)
```

## Documentation

* **`decoders.py`**

  * Implements `decode_bpqm` and related helper functions.
  * Set `debug=True` to print or visualize intermediate circuits and computational graphs.

* **`cloner.py`**

  * Defines approximate cloners for variable nodes in loopy graphs.
  * `VarNodeCloner` corresponds to the ENU cloner described in the paper.

* **`linearcode.py`**

  * Provides the `LinearCode` class for enumerating codewords and building factor graphs.

* **New:** Added `decode_single_codeword` function in `decoders.py` to directly decode an arbitrary codeword and return its measurement outcomes.
