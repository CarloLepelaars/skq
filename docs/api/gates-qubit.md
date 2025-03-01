# Qubit Gates API Reference

This page documents the qubit gates available in the `skq.gates.qubit` module.

## Qubit Gate Base Class

The `QubitGate` class serves as the foundation for all qubit-based quantum gates in SKQ.

::: skq.gates.qubit.base.QubitGate
    options:
      show_root_heading: true
      show_source: true

## Single-Qubit Gates

### Identity Gate (I)

The Identity gate leaves the qubit state unchanged.

**Matrix Representation:**
```
I = [1 0]
    [0 1]
```

::: skq.gates.qubit.single.I
    options:
      show_root_heading: true
      show_source: true

### Pauli-X Gate (NOT)

The Pauli-X gate is the quantum equivalent of the classical NOT gate. It flips the state of the qubit.

**Matrix Representation:**
```
X = [0 1]
    [1 0]
```

::: skq.gates.qubit.single.X
    options:
      show_root_heading: true
      show_source: true

### Pauli-Y Gate

The Pauli-Y gate rotates the qubit state around the Y-axis of the Bloch sphere.

**Matrix Representation:**
```
Y = [0  -i]
    [i   0]
```

::: skq.gates.qubit.single.Y
    options:
      show_root_heading: true
      show_source: true

### Pauli-Z Gate

The Pauli-Z gate rotates the qubit state around the Z-axis of the Bloch sphere.

**Matrix Representation:**
```
Z = [1  0]
    [0 -1]
```

::: skq.gates.qubit.single.Z
    options:
      show_root_heading: true
      show_source: true

### Hadamard Gate (H)

The Hadamard gate creates a superposition of the |0⟩ and |1⟩ states.

**Matrix Representation:**
```
H = 1/√2 * [1  1]
           [1 -1]
```

::: skq.gates.qubit.single.H
    options:
      show_root_heading: true
      show_source: true

### Phase Gate

The general Phase gate applies a phase shift to the |1⟩ state.

**Matrix Representation:**
```
Phase(φ) = [1      0    ]
           [0  e^(iφ)]
```

::: skq.gates.qubit.single.Phase
    options:
      show_root_heading: true
      show_source: true

### S Gate (π/2 Phase)

The S gate is a special case of the Phase gate with φ = π/2.

**Matrix Representation:**
```
S = [1  0]
    [0  i]
```

::: skq.gates.qubit.single.S
    options:
      show_root_heading: true
      show_source: true

### T Gate (π/4 Phase)

The T gate is a special case of the Phase gate with φ = π/4.

**Matrix Representation:**
```
T = [1       0    ]
    [0  e^(iπ/4)]
```

::: skq.gates.qubit.single.T
    options:
      show_root_heading: true
      show_source: true

### RX Gate (X-Rotation)

The RX gate rotates the qubit state around the X-axis of the Bloch sphere.

**Matrix Representation:**
```
RX(φ) = [    cos(φ/2)  -i·sin(φ/2)]
        [-i·sin(φ/2)      cos(φ/2)]
```

::: skq.gates.qubit.single.RX
    options:
      show_root_heading: true
      show_source: true

### RY Gate (Y-Rotation)

The RY gate rotates the qubit state around the Y-axis of the Bloch sphere.

**Matrix Representation:**
```
RY(φ) = [cos(φ/2)  -sin(φ/2)]
        [sin(φ/2)   cos(φ/2)]
```

::: skq.gates.qubit.single.RY
    options:
      show_root_heading: true
      show_source: true

### RZ Gate (Z-Rotation)

The RZ gate rotates the qubit state around the Z-axis of the Bloch sphere.

**Matrix Representation:**
```
RZ(φ) = [e^(-iφ/2)       0    ]
        [    0      e^(iφ/2)]
```

::: skq.gates.qubit.single.RZ
    options:
      show_root_heading: true
      show_source: true

### R Gate (General Rotation)

The R gate implements a general rotation by composing RZ, RY, and RZ rotations.

**Matrix Representation:**
```
R(θ,φ,λ) = RZ(λ) · RY(φ) · RZ(θ)
```

::: skq.gates.qubit.single.R
    options:
      show_root_heading: true
      show_source: true

### U3 Gate (Universal Rotation)

The U3 gate is a universal single-qubit gate that can represent any single-qubit operation.

**Matrix Representation:**
```
U3(θ,φ,δ) = RZ(δ) · RY(φ) · RX(θ)
```

::: skq.gates.qubit.single.U3
    options:
      show_root_heading: true
      show_source: true

### Measure Gate

The Measure gate performs a measurement on the qubit.

::: skq.gates.qubit.single.Measure
    options:
      show_root_heading: true
      show_source: true

## Multi-Qubit Gates

### CNOT (CX) Gate

The CNOT (Controlled-NOT) gate flips the target qubit if the control qubit is |1⟩.

**Matrix Representation:**
```
CX = [1 0 0 0]
     [0 1 0 0]
     [0 0 0 1]
     [0 0 1 0]
```

::: skq.gates.qubit.multi.CX
    options:
      show_root_heading: true
      show_source: true

### Controlled-Y (CY) Gate

The CY gate applies a Y gate to the target qubit if the control qubit is |1⟩.

**Matrix Representation:**
```
CY = [1 0  0   0]
     [0 1  0   0]
     [0 0  0  -i]
     [0 0  i   0]
```

::: skq.gates.qubit.multi.CY
    options:
      show_root_heading: true
      show_source: true

### Controlled-Z (CZ) Gate

The CZ gate applies a Z gate to the target qubit if the control qubit is |1⟩.

**Matrix Representation:**
```
CZ = [1 0 0  0]
     [0 1 0  0]
     [0 0 1  0]
     [0 0 0 -1]
```

::: skq.gates.qubit.multi.CZ
    options:
      show_root_heading: true
      show_source: true

### Controlled-H (CH) Gate

The CH gate applies a Hadamard gate to the target qubit if the control qubit is |1⟩.

**Matrix Representation:**
```
CH = [1 0                0                 0              ]
     [0 1                0                 0              ]
     [0 0  1/√2          1/√2             ]
     [0 0  1/√2         -1/√2             ]
```

::: skq.gates.qubit.multi.CH
    options:
      show_root_heading: true
      show_source: true

### Controlled-Phase (CPhase) Gate

The CPhase gate applies a phase shift to the |11⟩ state.

**Matrix Representation:**
```
CPhase(φ) = [1 0 0      0     ]
            [0 1 0      0     ]
            [0 0 1      0     ]
            [0 0 0  e^(iφ)]
```

::: skq.gates.qubit.multi.CPhase
    options:
      show_root_heading: true
      show_source: true

### Controlled-S (CS) Gate

The CS gate is a special case of the CPhase gate with φ = π/2.

**Matrix Representation:**
```
CS = [1 0 0 0]
     [0 1 0 0]
     [0 0 1 0]
     [0 0 0 i]
```

::: skq.gates.qubit.multi.CS
    options:
      show_root_heading: true
      show_source: true

### Controlled-T (CT) Gate

The CT gate is a special case of the CPhase gate with φ = π/4.

**Matrix Representation:**
```
CT = [1 0 0      0     ]
     [0 1 0      0     ]
     [0 0 1      0     ]
     [0 0 0  e^(iπ/4)]
```

::: skq.gates.qubit.multi.CT
    options:
      show_root_heading: true
      show_source: true

### SWAP Gate

The SWAP gate exchanges the states of two qubits.

**Matrix Representation:**
```
SWAP = [1 0 0 0]
       [0 0 1 0]
       [0 1 0 0]
       [0 0 0 1]
```

::: skq.gates.qubit.multi.SWAP
    options:
      show_root_heading: true
      show_source: true

### Controlled-SWAP (CSWAP) Gate

The CSWAP gate, also known as the Fredkin gate, swaps two qubits if the control qubit is |1⟩.

**Matrix Representation:**
```
CSWAP = [1 0 0 0 0 0 0 0]
        [0 1 0 0 0 0 0 0]
        [0 0 1 0 0 0 0 0]
        [0 0 0 1 0 0 0 0]
        [0 0 0 0 1 0 0 0]
        [0 0 0 0 0 0 1 0]
        [0 0 0 0 0 1 0 0]
        [0 0 0 0 0 0 0 1]
```

::: skq.gates.qubit.multi.CSwap
    options:
      show_root_heading: true
      show_source: true

### Toffoli (CCX) Gate

The Toffoli gate, or CCX gate, applies an X gate to the target qubit if both control qubits are |1⟩.

**Matrix Representation:**
```
CCX = [1 0 0 0 0 0 0 0]
      [0 1 0 0 0 0 0 0]
      [0 0 1 0 0 0 0 0]
      [0 0 0 1 0 0 0 0]
      [0 0 0 0 1 0 0 0]
      [0 0 0 0 0 1 0 0]
      [0 0 0 0 0 0 0 1]
      [0 0 0 0 0 0 1 0]
```

::: skq.gates.qubit.multi.CCX
    options:
      show_root_heading: true
      show_source: true

### CCY Gate

The CCY gate applies a Y gate to the target qubit if both control qubits are |1⟩.

**Matrix Representation:**
```
CCY = [1 0 0 0 0 0 0    0   ]
      [0 1 0 0 0 0 0    0   ]
      [0 0 1 0 0 0 0    0   ]
      [0 0 0 1 0 0 0    0   ]
      [0 0 0 0 1 0 0    0   ]
      [0 0 0 0 0 1 0    0   ]
      [0 0 0 0 0 0 0   -i   ]
      [0 0 0 0 0 0 i    0   ]
```

::: skq.gates.qubit.multi.CCY
    options:
      show_root_heading: true
      show_source: true

### CCZ Gate

The CCZ gate applies a Z gate to the target qubit if both control qubits are |1⟩.

**Matrix Representation:**
```
CCZ = [1 0 0 0 0 0 0  0]
      [0 1 0 0 0 0 0  0]
      [0 0 1 0 0 0 0  0]
      [0 0 0 1 0 0 0  0]
      [0 0 0 0 1 0 0  0]
      [0 0 0 0 0 1 0  0]
      [0 0 0 0 0 0 1  0]
      [0 0 0 0 0 0 0 -1]
```

::: skq.gates.qubit.multi.CCZ
    options:
      show_root_heading: true
      show_source: true

### Multi-Controlled X (MCX) Gate

The MCX gate applies an X gate to the target qubit if all control qubits are |1⟩.

::: skq.gates.qubit.multi.MCX
    options:
      show_root_heading: true
      show_source: true

### Multi-Controlled Y (MCY) Gate

The MCY gate applies a Y gate to the target qubit if all control qubits are |1⟩.

::: skq.gates.qubit.multi.MCY
    options:
      show_root_heading: true
      show_source: true

### Multi-Controlled Z (MCZ) Gate

The MCZ gate applies a Z gate to the target qubit if all control qubits are |1⟩.

::: skq.gates.qubit.multi.MCZ
    options:
      show_root_heading: true
      show_source: true

### Quantum Fourier Transform (QFT) Gate

The QFT gate implements the Quantum Fourier Transform, a key component in many quantum algorithms.

**Matrix Representation (for n=2):**
```
QFT = 1/2 * [1  1  1  1]
             [1  i -1 -i]
             [1 -1  1 -1]
             [1 -i -1  i]
```

::: skq.gates.qubit.multi.QFT
    options:
      show_root_heading: true
      show_source: true

### Deutsch Oracle

The Deutsch Oracle implements the oracle for the Deutsch algorithm.

::: skq.gates.qubit.multi.DeutschOracle
    options:
      show_root_heading: true
      show_source: true

### Deutsch-Jozsa Oracle

The Deutsch-Jozsa Oracle implements the oracle for the Deutsch-Jozsa algorithm.

::: skq.gates.qubit.multi.DeutschJozsaOracle
    options:
      show_root_heading: true
      show_source: true

### Phase Oracle

The Phase Oracle marks a target state with a phase shift, as used in Grover's search algorithm.

::: skq.gates.qubit.multi.PhaseOracle
    options:
      show_root_heading: true
      show_source: true

### Grover Diffusion Operator

The Grover Diffusion Operator amplifies the amplitude of the marked state in Grover's search algorithm.

::: skq.gates.qubit.multi.GroverDiffusion
    options:
      show_root_heading: true
      show_source: true

### Controlled Unitary (CU) Gate

The CU gate applies a unitary operation conditionally based on a control qubit.

::: skq.gates.qubit.multi.CU
    options:
      show_root_heading: true
      show_source: true

### Multi-Controlled Unitary (MCU) Gate

The MCU gate applies a unitary operation conditionally based on multiple control qubits.

::: skq.gates.qubit.multi.MCU
    options:
      show_root_heading: true
      show_source: true

### Cross-Resonance (CR) Gate

The CR gate is a simple Cross-Resonance gate used in superconducting qubit architectures.

::: skq.gates.qubit.multi.CR
    options:
      show_root_heading: true
      show_source: true

### Symmetric Echoed Cross-Resonance (SymmetricECR) Gate

The SymmetricECR gate is a symmetric echoed Cross-Resonance gate.

::: skq.gates.qubit.multi.SymmetricECR
    options:
      show_root_heading: true
      show_source: true

### Asymmetric Echoed Cross-Resonance (AsymmetricECR) Gate

The AsymmetricECR gate is an asymmetric echoed Cross-Resonance gate.

::: skq.gates.qubit.multi.AsymmetricECR
    options:
      show_root_heading: true
      show_source: true 