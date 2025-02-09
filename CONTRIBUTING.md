# Contributing

Thank you for considering to contribute to `skq`! Here we provide some general guidelines to streamline the contribution process. The goal of this library is to empower Python programmers to easily build their own quantum circuits and simulate them. `skq` also gives people to option to convert to popular quantum computing frameworks so the circuits can be put on real quantum computers. Lastly, `skq` is used in the [q4p (Quantum Computing for Programmers)](https://github.com/CarloLepelaars/q4p) course for quantum education.

## Before you start

- Fork [skq](https://github.com/CarloLepelaars/skq) from Github.

- install `skq` in editable mode:

```bash
pip install uv
uv pip install -e .
```

## How you can contribute

We always welcome contributions to `skq`. There are several aspects to this repository:

1. **Gates:** This section provides all the gates from which quantum circuits are built. We welcome corrections, new features and gates. If you would like to contribute a new gate, please create a Github issue first so we can discuss the idea. Please create a Pull Request (PR) with your changes so we can review them. Look at the existing gates for inheritance structures and implementation details.

For example, for Qubit gates we inherit from `QubitGate` and implement at least `__new__` `to_qiskit`, `to_pennylane` and `to_qasm` methods.

```python
class I(QubitGate):
    """
    Identity gate:
    [[1, 0]
    [0, 1]]
    """
    def __new__(cls):
        return super().__new__(cls, np.eye(2))

    def to_qiskit(self) -> qiskit.circuit.library.IGate:
        return qiskit.circuit.library.IGate()

    def to_pennylane(self, wires: list[int] | int) -> qml.I:
        return qml.I(wires=wires)

    def to_qasm(self, qubits: list[int]) -> str:
        return f"id q[{qubits[0]}];"
```

2. **Quantum info:** The `quantum_info` folder contains objects and function for quantum information analysis.

3. **Circuits:** The `circuits` folder contains more high-level objects to create and convert quantum circuits.

## PR submission guidelines

- Keep each PR focused. While it's more convenient, do not combine several unrelated contributions together. It can be a good idea to split contributions into multiple PRs.
- Do not turn an already submitted PR into your development playground. If after you submitted a pull request you discovered that more work is needed - close the PR, do the required work and then submit a new PR. Otherwise each of your commits requires attention from maintainers of the project.
- Make sure to add tests for new features.
