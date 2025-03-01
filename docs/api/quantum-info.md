# Quantum Information API Reference

This page documents the classes and functions available in the `skq.quantum_info` module. This module provides tools for working with quantum states, density matrices, quantum channels, Hamiltonians, and other quantum information concepts.

## State Representations

### Statevector

The `Statevector` class provides a representation for pure quantum states.

```python
from skq.quantum_info import Statevector

# Create a state vector
state = Statevector([1/np.sqrt(2), 1/np.sqrt(2)])
print(state.num_qubits)  # 1
print(state.is_normalized())  # True
```

::: skq.quantum_info.Statevector
    options:
      show_root_heading: true
      show_source: true

### Predefined States

SKQ provides several predefined quantum states:

::: skq.quantum_info.ZeroState
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.OneState
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.PlusState
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.MinusState
    options:
      show_root_heading: true
      show_source: true

### Bell States

::: skq.quantum_info.PhiPlusState
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.PhiMinusState
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.PsiPlusState
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.PsiMinusState
    options:
      show_root_heading: true
      show_source: true

### Multi-qubit States

::: skq.quantum_info.GHZState
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.WState
    options:
      show_root_heading: true
      show_source: true

## Density Matrices

Density matrices represent both pure and mixed quantum states.

```python
from skq.quantum_info import DensityMatrix

# Create a density matrix
rho = DensityMatrix(np.array([[0.5, 0], [0, 0.5]]))
print(rho.is_pure())  # False
print(rho.is_mixed())  # True
```

::: skq.quantum_info.DensityMatrix
    options:
      show_root_heading: true
      show_source: true

### Thermal States

::: skq.quantum_info.GibbsState
    options:
      show_root_heading: true
      show_source: true

## Quantum Channels

Quantum channels represent physical operations on quantum states.

```python
from skq.quantum_info import DepolarizingChannel

# Create a depolarizing channel with 10% noise
channel = DepolarizingChannel(0.1)
```

::: skq.quantum_info.QuantumChannel
    options:
      show_root_heading: true
      show_source: true

### Predefined Channels

::: skq.quantum_info.QubitResetChannel
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.DepolarizingChannel
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.PauliNoiseChannel
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.CompletelyDephasingChannel
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.AmplitudeDampingChannel
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.PhaseFlipChannel
    options:
      show_root_heading: true
      show_source: true

## Hamiltonians

Hamiltonians represent the energy of quantum systems.

```python
from skq.quantum_info import Hamiltonian, IsingHamiltonian

# Create an Ising model Hamiltonian for 3 qubits
H = IsingHamiltonian(num_qubits=3, J=1.0, h=0.5)
print(H.ground_state_energy())
```

::: skq.quantum_info.Hamiltonian
    options:
      show_root_heading: true
      show_source: true

### Predefined Hamiltonians

::: skq.quantum_info.IsingHamiltonian
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.HeisenbergHamiltonian
    options:
      show_root_heading: true
      show_source: true

## Hadamard Matrices

Hadamard matrices are useful in quantum information theory and quantum algorithms.

```python
from skq.quantum_info import generate_hadamard_matrix

# Generate a Hadamard matrix of order 4
H = generate_hadamard_matrix(4)
```

::: skq.quantum_info.HadamardMatrix
    options:
      show_root_heading: true
      show_source: true

::: skq.quantum_info.generate_hadamard_matrix
    options:
      show_root_heading: true
      show_source: true

## Superoperators

Superoperators are linear maps that transform operators to operators.

::: skq.quantum_info.SuperOperator
    options:
      show_root_heading: true
      show_source: true 