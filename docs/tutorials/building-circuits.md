# Building Quantum Circuits

This tutorial explains how to build quantum circuits from scratch using `skq`. We'll cover:

1. Creating basic circuits
2. Combining gates in different ways
3. Building parameterized circuits
4. Creating custom gates

## Basic Circuit Construction

In `skq`, a circuit is a sequence of quantum gates that are applied to a quantum state:

```python
from skq.gates import H, X
from skq.circuits import Circuit
import numpy as np

# Create a simple circuit that applies H then X to a single qubit
circuit = Circuit([H(), X()])

# Apply to |0⟩ state
state = np.array([1, 0])
result = circuit(state)
print(result)
# Output: [0.70710678+0.j -0.70710678+0.j]
```

## Multi-Qubit Circuits

For multi-qubit circuits, you need to specify how gates act on different qubits:

```python
from skq.gates import H, I, CX
from skq.circuits import Circuit, Concat

# Create a Bell state circuit
bell_circuit = Circuit([
    Concat([H(), I()]),  # Apply H to first qubit, I to second qubit
    CX()                 # Apply CNOT with first qubit as control
])

# Apply to |00⟩ state
state = np.array([1, 0, 0, 0])
result = bell_circuit(state)
print(result)
# Output: [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```

## Custom Gates

You can create custom gates by defining their matrix representation:

```python
import numpy as np
from skq.gates.qubit import QubitGate

# Define a custom gate as a subclass of Gate
class MySqrtX(QubitGate):
    def __init__(self):
        # Define the matrix for √X gate
        matrix = np.array([
            [0.5 + 0.5j, 0.5 - 0.5j],
            [0.5 - 0.5j, 0.5 + 0.5j]
        ])
        super().__init__(matrix)

# Use the custom gate
sqrt_x = MySqrtX()
circuit = Circuit([sqrt_x])

# Apply to |0⟩ state
state = np.array([1, 0])
result = circuit(state)
print(result)
# Output: [0.5+0.5j 0.5-0.5j]
```

## Controlled Operations

You can create controlled versions of any gate:

```python
from skq.gates import X, controlled
import numpy as np

# Create a controlled-X gate (equivalent to CX)
cx_gate = controlled(X())

# Apply to |10⟩ state
state = np.array([0, 1, 0, 0])
result = cx_gate @ state
print(result)
# Output: [0 0 0 1]
# This is |11⟩
```

## Circuit Visualization

```python
from skq.circuits import Circuit, Concat
from skq.gates import H, I, CX

# Create a Bell state circuit
bell_circuit = Circuit([
    Concat([H(), I()]),
    CX()
])

# ASCII
bell_circuit.draw()
# Matplotlib
bell_circuit.draw(output="mpl")
```

## Next Steps

Now that you know how to build circuits, you can:

1. Learn about [Circuit Conversion](circuit-conversion.md) to other frameworks