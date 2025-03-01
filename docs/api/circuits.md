# Circuits API Reference

This page documents the classes and functions available in the `skq.circuits` module. This module provides tools for creating and manipulating quantum circuits, as well as implementations of quantum algorithms.

## Circuit Building Blocks

### Circuit

::: skq.circuits.Circuit
    options:
      show_root_heading: true
      show_source: true

### Concat

```python
from skq.circuits import Circuit, Concat
from skq.gates import H, I

# Apply Hadamard to first qubit and Identity to second qubit
parallel_gates = Concat([H(), I()])

# Create a circuit with these parallel gates
circuit = Circuit([parallel_gates])
```

::: skq.circuits.Concat
    options:
      show_root_heading: true
      show_source: true

## Quantum Algorithms

### Bell States

The `BellStates` class provides circuits for creating the four Bell states, which are maximally entangled two-qubit states.

```python
from skq.circuits import BellStates

# Create a circuit for the Φ+ Bell state (|00⟩ + |11⟩)/√2
bell_circuit = BellStates().circuit(configuration=1)

# Create a circuit for the Ψ- Bell state (|01⟩ - |10⟩)/√2
bell_circuit = BellStates().circuit(configuration=4)
```

::: skq.circuits.BellStates
    options:
      show_root_heading: true
      show_source: true

### GHZ States

The `GHZStates` class provides circuits for creating Greenberger-Horne-Zeilinger (GHZ) states, which are multi-qubit generalizations of Bell states.

```python
from skq.circuits import GHZStates

# Create a circuit for a 3-qubit GHZ state (|000⟩ + |111⟩)/√2
ghz_circuit = GHZStates().circuit(n_qubits=3)
```

::: skq.circuits.GHZStates
    options:
      show_root_heading: true
      show_source: true

### W State

The `WState` class provides a circuit for creating the 3-qubit W state, which is another type of entangled state.

```python
from skq.circuits import WState

# Create a circuit for the W state (|001⟩ + |010⟩ + |100⟩)/√3
w_circuit = WState().circuit()
```

::: skq.circuits.WState
    options:
      show_root_heading: true
      show_source: true

### Deutsch's Algorithm

The `Deutsch` class implements Deutsch's algorithm, which determines whether a function is constant or balanced with a single query.

```python
from skq.circuits import Deutsch

# Define a constant function (always returns 0)
def constant_function(x):
    return 0

# Create a circuit for Deutsch's algorithm
deutsch_circuit = Deutsch().circuit(f=constant_function)
```

::: skq.circuits.Deutsch
    options:
      show_root_heading: true
      show_source: true

### Deutsch-Jozsa Algorithm

The `DeutschJozsa` class implements the Deutsch-Jozsa algorithm, which is a generalization of Deutsch's algorithm to multiple qubits.

```python
from skq.circuits import DeutschJozsa

# Define a constant function for multiple bits
def constant_function(x):
    return 0

# Create a circuit for the Deutsch-Jozsa algorithm with 3 qubits
dj_circuit = DeutschJozsa().circuit(f=constant_function, n_bits=3)
```

::: skq.circuits.DeutschJozsa
    options:
      show_root_heading: true
      show_source: true

### Grover's Algorithm

The `Grover` class implements Grover's search algorithm, which can find a marked item in an unsorted database quadratically faster than classical algorithms.

```python
from skq.circuits import Grover
import numpy as np

# Create a target state to search for (|100⟩)
target_state = np.zeros(8)
target_state[4] = 1

# Create a circuit for Grover's algorithm
grover_circuit = Grover().circuit(
    target_state=target_state,
    n_qubits=3,
    n_iterations=1
)
```

::: skq.circuits.Grover
    options:
      show_root_heading: true
      show_source: true

## Circuit Conversion

SKQ circuits can be converted to other quantum computing frameworks:

```python
from skq.circuits import Circuit, Concat
from skq.gates import H, I, CX

# Create a Bell state circuit
bell_circuit = Circuit([
    Concat([H(), I()]),  # Apply H to first qubit, I to second qubit
    CX()                 # Apply CNOT with first qubit as control
])

# Convert to Qiskit
qiskit_circuit = bell_circuit.convert(framework="qiskit")

# Convert to OpenQASM
qasm_code = bell_circuit.convert(framework="qasm")
```

::: skq.circuits.convert_to_qiskit
    options:
      show_root_heading: true
      show_source: true

::: skq.circuits.convert_to_qasm
    options:
      show_root_heading: true
      show_source: true 