# Circuit Conversion

One of the key features of `skq` is the ability to convert quantum circuits to other popular quantum computing frameworks. This allows you to:

1. Design and test circuits in `skq` using NumPy
2. Convert them to other frameworks for execution on real quantum hardware
3. Leverage the strengths of different frameworks

## Supported Frameworks

Currently, `skq` supports conversion to:

- [Qiskit](https://qiskit.org/) - IBM's quantum computing framework
- [OpenQASM](https://openqasm.com/) - Open Quantum Assembly Language

## Converting to Qiskit

To convert a circuit to Qiskit:

```python
from skq.gates import H, I, CX
from skq.circuits import Circuit, Concat

# Create a Bell state circuit
bell_circuit = Circuit([
    Concat([H(), I()]),  # Apply H to first qubit, I to second qubit
    CX()                 # Apply CNOT with first qubit as control
])

# Convert to Qiskit
qiskit_circuit = bell_circuit.convert(framework="qiskit")

# Draw the circuit
print(qiskit_circuit.draw())
# Output:
#      ┌───┐     
# q_0: ┤ H ├──■──
#      └───┘┌─┴─┐
# q_1: ─────┤ X ├
#           └───┘

# You can now use all Qiskit features
from qiskit import Aer, execute

# Run on Qiskit simulator
simulator = Aer.get_backend('statevector_simulator')
result = execute(qiskit_circuit, simulator).result()
statevector = result.get_statevector()
print(statevector)
# Output: [0.70710678+0.j 0.        +0.j 0.        +0.j 0.70710678+0.j]
```

## Converting to OpenQASM

OpenQASM is a text-based representation of quantum circuits:

```python
# Convert to OpenQASM
qasm_code = bell_circuit.convert(framework="qasm")
print(qasm_code)
# Output:
# h q[0];
# cx q[0], q[1];
```

## Converting Complex Algorithms

You can also convert more complex algorithms:

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

# Convert to Qiskit
qiskit_grover = grover_circuit.convert(framework="qiskit")
qiskit_grover.draw()
```

## Handling Framework-Specific Features

When converting circuits, `skq` handles framework-specific features automatically:

- Identity gates are removed when not needed
- Gates are mapped to their framework-specific equivalents
- Circuit structure is preserved

## Limitations

There are some limitations to be aware of:

1. Not all quantum operations have direct equivalents in all frameworks
2. Custom gates may require special handling
3. Some framework-specific optimizations may not be applied

## Next Steps

Now that you know how to convert circuits, you can:

1. Learn how to [build circuits from scratch](building-circuits.md)
