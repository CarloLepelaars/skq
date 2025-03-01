# Qutrit Gates API Reference

This page documents the qutrit gates available in the `skq.gates.qutrit` module. Qutrits are quantum systems with 3 basis states (|0⟩, |1⟩, |2⟩) and can model spin-1 particles like photons and gluons.

## Qutrit Gate Base Class

The `QutritGate` class serves as the foundation for all qutrit-based quantum gates in SKQ.

::: skq.gates.qutrit.base.QutritGate
    options:
      show_root_heading: true
      show_source: true

## Single-Qutrit Gates

### Identity Gate (QutritI)

The Identity gate leaves the qutrit state unchanged.

**Matrix Representation:**

$$
\text{QutritI} = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

::: skq.gates.qutrit.single.QutritI
    options:
      show_root_heading: true
      show_source: true

### X Gate (QutritX)

The X gate for a qutrit performs a cyclic permutation of the basis states: |0⟩ → |1⟩ → |2⟩ → |0⟩.

**Matrix Representation:**

$$
\text{QutritX} = \begin{pmatrix}
0 & 0 & 1 \\
1 & 0 & 0 \\
0 & 1 & 0
\end{pmatrix}
$$

::: skq.gates.qutrit.single.QutritX
    options:
      show_root_heading: true
      show_source: true

### Y Gate (QutritY)

The Y gate for a qutrit performs a cyclic permutation with phase: |0⟩ → -i|1⟩ → -i|2⟩ → -i|0⟩.

**Matrix Representation:**

$$
\text{QutritY} = \begin{pmatrix}
0 & 0 & -i \\
-i & 0 & 0 \\
0 & -i & 0
\end{pmatrix}
$$

::: skq.gates.qutrit.single.QutritY
    options:
      show_root_heading: true
      show_source: true

### Z Gate (QutritZ)

The Z gate for a qutrit applies different phases to each basis state.

**Matrix Representation:**

$$
\text{QutritZ} = \begin{pmatrix}
1 & 0 & 0 \\
0 & e^{2\pi i/3} & 0 \\
0 & 0 & e^{-2\pi i/3}
\end{pmatrix}
$$

::: skq.gates.qutrit.single.QutritZ
    options:
      show_root_heading: true
      show_source: true

### Hadamard Gate (QutritH)

The Hadamard gate for a qutrit creates a superposition of the three basis states.

**Matrix Representation:**

$$
\text{QutritH} = \frac{1}{\sqrt{3}} \begin{pmatrix}
1 & 1 & 1 \\
1 & e^{2\pi i/3} & e^{4\pi i/3} \\
1 & e^{4\pi i/3} & e^{2\pi i/3}
\end{pmatrix}
$$

Where $e^{2\pi i/3}$ is the cube root of unity.

::: skq.gates.qutrit.single.QutritH
    options:
      show_root_heading: true
      show_source: true

### T Gate (QutritT)

The T gate for a qutrit applies smaller phase shifts than the Z gate.

**Matrix Representation:**

$$
\text{QutritT} = \begin{pmatrix}
1 & 0 & 0 \\
0 & e^{2\pi i/9} & 0 \\
0 & 0 & e^{-2\pi i/9}
\end{pmatrix}
$$

::: skq.gates.qutrit.single.QutritT
    options:
      show_root_heading: true
      show_source: true

### R Gate (QutritR)

The R gate for a qutrit is a non-Clifford gate that applies a phase flip to the |2⟩ state.

**Matrix Representation:**

$$
\text{QutritR} = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & -1
\end{pmatrix}
$$

::: skq.gates.qutrit.single.QutritR
    options:
      show_root_heading: true
      show_source: true

### Phase Gate (QutritPhase)

The general Phase gate for qutrits applies arbitrary phase shifts to each basis state.

**Matrix Representation:**

$$
\text{QutritPhase}(\phi_0,\phi_1,\phi_2) = \begin{pmatrix}
e^{i\phi_0} & 0 & 0 \\
0 & e^{i\phi_1} & 0 \\
0 & 0 & e^{i\phi_2}
\end{pmatrix}
$$

::: skq.gates.qutrit.single.QutritPhase
    options:
      show_root_heading: true
      show_source: true

### S Gate (QutritS)

The S gate for qutrits is a special case of the Phase gate.

**Matrix Representation:**

$$
\text{QutritS} = \begin{pmatrix}
1 & 0 & 0 \\
0 & e^{2\pi i/3} & 0 \\
0 & 0 & e^{4\pi i/3}
\end{pmatrix}
$$

::: skq.gates.qutrit.single.QutritS
    options:
      show_root_heading: true
      show_source: true

## Multi-Qutrit Gates

### Multi-Qutrit Identity (QutritMI)

The Identity gate for multiple qutrits.

**Matrix Representation (for 1 qutrit):**

$$
\text{QutritMI} = \begin{pmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & 1
\end{pmatrix}
$$

::: skq.gates.qutrit.multi.QutritMI
    options:
      show_root_heading: true
      show_source: true

### Controlled-X Type A (QutritCXA)

The CNOT gate for qutrits with control on |1⟩. This gate performs a cyclic permutation on the target qutrit if the control qutrit is in state |1⟩.

**Matrix Representation (9×9 matrix):**

$$
\text{QutritCXA} = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

::: skq.gates.qutrit.multi.QutritCXA
    options:
      show_root_heading: true
      show_source: true

### Controlled-X Type B (QutritCXB)

The CNOT gate for qutrits with control on |2⟩. This gate performs a cyclic permutation on the target qutrit if the control qutrit is in state |2⟩.

**Matrix Representation (9×9 matrix):**

$$
\text{QutritCXB} = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0
\end{pmatrix}
$$

::: skq.gates.qutrit.multi.QutritCXB
    options:
      show_root_heading: true
      show_source: true

### Controlled-X Type C (QutritCXC)

The CNOT gate for qutrits with control on |0⟩. This gate performs a cyclic permutation on the target qutrit if the control qutrit is in state |0⟩.

**Matrix Representation (9×9 matrix):**

$$
\text{QutritCXC} = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0
\end{pmatrix}
$$

::: skq.gates.qutrit.multi.QutritCXC
    options:
      show_root_heading: true
      show_source: true

### SWAP Gate (QutritSWAP)

The SWAP gate for qutrits exchanges the states of two qutrits.

**Matrix Representation (9×9 matrix):**

$$
\text{QutritSWAP} = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

::: skq.gates.qutrit.multi.QutritSWAP
    options:
      show_root_heading: true
      show_source: true 