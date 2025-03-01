# Getting Started

## Installation

You can install `skq` using pip:

```bash
pip install -U skq
```

For development, you can clone the repository and install it in development mode:

```bash
git clone https://github.com/CarloLepelaars/skq.git
cd skq
pip install -e ".[dev]"
```

## Basic Usage

### Creating Quantum Gates

`skq` provides a simple interface to create quantum gates. Under the hood, gates inherit from NumPy.

```python
from skq.gates.qubit import H, X, Y, Z, I, CX, SWAP

# Single-qubit gates
h_gate = H()  # Hadamard gate
x_gate = X()  # Pauli-X gate (NOT gate)
y_gate = Y()  # Pauli-Y gate
z_gate = Z()  # Pauli-Z gate
i_gate = I()  # Identity gate

# Multi-qubit gates
cx_gate = CX()  # CNOT gate
swap_gate = SWAP()  # SWAP gate

# Convert gates to other frameworks
# Qiskit
h_gate_qiskit = h_gate.to_qiskit()
# OpenQASM
h_gate_qasm = h_gate.to_qasm()

# Display the gate
# ASCII
h_gate.draw()
# Matplotlib
h_gate.draw(output="mpl")
```

### Creating Quantum Circuits

You can create quantum circuits by combining gates:

```python
from skq.circuits import Circuit, Concat
import numpy as np

# Create a Bell state circuit
bell_circuit = Circuit([
    Concat([H(), I()]),  # Apply H to first qubit, I to second qubit
    CX()                 # Apply CNOT with first qubit as control
])

# Initial state |00⟩
initial_state = np.array([1, 0, 0, 0])

# Run the circuit
final_state = bell_circuit(initial_state)
print(final_state)
# [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```

### Converting to Other Frameworks

`skq` allows you to convert your circuits to other quantum computing frameworks:

```python
# Convert to Qiskit
qiskit_circuit = bell_circuit.convert(framework="qiskit")
print(qiskit_circuit.draw())

# Convert to OpenQASM
qasm_code = bell_circuit.convert(framework="qasm")
print(qasm_code)
```

### Implementing Quantum Algorithms

`skq` includes implementations of common quantum algorithms:

```python
from skq.circuits import Grover
import numpy as np

# Create a Grover search circuit for 3 qubits
# Searching for state |100⟩ (binary 4)
target_state = np.zeros(8)
target_state[4] = 1

grover_circuit = Grover().circuit(
    n_qubits=3,
    target_state=target_state,
    n_iterations=1
)

# Run the circuit
initial_state = np.array([1, 0, 0, 0, 0, 0, 0, 0])  # |000⟩
result = grover_circuit(initial_state)
print(result)
# The amplitude of state |100⟩ should be significantly higher
```

## Next Steps

- Learn how to [contribute](contributing.md) to the project.