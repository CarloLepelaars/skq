# skq

![](https://img.shields.io/pypi/dm/skq)
![Python Version](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/CarloLepelaars/skq/main/pyproject.toml&query=%24.project%5B%22requires-python%22%5D&label=python&color=blue) 
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)



Scientific Toolkit for Quantum Computing

This library is used in the [q4p (Quantum Computing for Programmers)](https://github.com/CarloLepelaars/q4p) course.

![q4p](https://carlo.ai/images/q4p.png)

NOTE: This library is developed for educational purposes. While we strive for correctness of everything, the code is provided as is and not guaranteed to be bug-free. For sensitive applications make sure you check computations. 

## Why SKQ?

- Exploration: Play with fundamental quantum building blocks using [NumPy](https://numpy.org).
- Education: Learn quantum computing concepts and algorithms.
- Integration: Combine classical components with quantum components.
- Democratize quantum for Python programmers and data scientists: Develop quantum algorithms in your favorite environment and easily export to your favorite quantum computing platform for running on real quantum hardware.

## Install

```bash
pip install -U skq
```

## Quickstart

### Circuit Conversion

Run this code snippet to initialize Grover's algorithm and convert to Qiskit to run on quantum hardware. The algorithm can also be run within `skq` as a classical simulation.

```python
from skq.circuits import Grover

# Initialize Grover's search skq Circuit
circuit = Grover().circuit(n_qubits=3, target_state=np.array([0, 0, 0, 0, 1, 0, 0, 0]), n_iterations=1)

# Conversion to Qiskit
qiskit_circuit = circuit.convert(framework="qiskit")
qiskit_circuit.draw()
#      ┌───┐┌──────────────┐┌──────────────────┐┌─┐      
# q_0: ┤ H ├┤0             ├┤0                 ├┤M├──────
#      ├───┤│              ││                  │└╥┘┌─┐   
# q_1: ┤ H ├┤1 PhaseOracle ├┤1 GroverDiffusion ├─╫─┤M├───
#      ├───┤│              ││                  │ ║ └╥┘┌─┐
# q_2: ┤ H ├┤2             ├┤2                 ├─╫──╫─┤M├
#      └───┘└──────────────┘└──────────────────┘ ║  ║ └╥┘
# c: 3/══════════════════════════════════════════╩══╩══╩═
#                                                0  1  2 

# Run circuit as classical simulation
print(grover([1,0,0,0,0,0,0,0]))
# array([0.03125, 0.03125, 0.03125, 0.03125, 0.78125, 0.03125, 0.03125, 0.03125])
```

### Circuits from scratch

You can also build your own custom circuits from scratch using individual gates. All gates can be converted to popular frameworks like Qiskit and OpenQASM.

```python
from skq.gates import H, I, CX
from skq.circuits import Concat, Circuit

H() # Hadamard gate (NumPy array)
# H([[ 0.70710678+0.j,  0.70710678+0.j],
#    [ 0.70710678+0.j, -0.70710678+0.j]])

I() # Identity gate (NumPy array)
# I([[1.+0.j, 0.+0.j],
#    [0.+0.j, 1.+0.j]])

CX() # CNOT gate (NumPy array)
# CX([[1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
#     [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
#     [0.+0.j, 0.+0.j, 0.+0.j, 1.+0.j],
#     [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j]])

# Initialize Bell State skq Circuit
circuit = Circuit([Concat([H(), I()]), CX()])

# Simulate circuit classically
state = np.array([1, 0, 0, 0]) # |00> state
circuit(state)
# array([0.70710678+0.j, 0, 0, 0.70710678+0.j])

# Conversion to Qiskit (Identity gates are removed)
qiskit_circuit = circuit.convert(framework="qiskit")
qiskit_circuit.draw()
#      ┌───┐     
# q_0: ┤ H ├──■──
#      └───┘┌─┴─┐
# q_1: ─────┤ X ├
#           └───┘

# Conversion to OpenQASM
qasm_circuit = circuit.convert(framework="qasm")
print(qasm_circuit)
# h q[0];
# cx q[0], q[1];
```

![q4p](https://carlo.ai/images/q4p.png)