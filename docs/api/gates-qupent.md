# Qupent Gates API Reference

This page documents the qupent gates available in the `skq.gates.qupent` module. Qupents are quantum systems with 5 basis states (|0⟩, |1⟩, |2⟩, |3⟩, |4⟩) and can model spin-2 particles like the graviton.

## Qupent Gate Base Class

The `QupentGate` class serves as the foundation for all qupent-based quantum gates in SKQ.

::: skq.gates.qupent.base.QupentGate
    options:
      show_root_heading: true
      show_source: true

## Single-Qupent Gates

### Identity Gate (QupentI)

The Identity gate leaves the qupent state unchanged.

**Matrix Representation:**

$$
\text{QupentI} = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0 \\
0 & 0 & 0 & 0 & 1
\end{pmatrix}
$$

::: skq.gates.qupent.single.QupentI
    options:
      show_root_heading: true
      show_source: true

### X Gate (QupentX)

The X gate for a qupent performs a cyclic permutation of the basis states: |0⟩ → |1⟩ → |2⟩ → |3⟩ → |4⟩ → |0⟩.

**Matrix Representation:**

$$
\text{QupentX} = \begin{pmatrix}
0 & 0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 & 0 \\
0 & 0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1 & 0
\end{pmatrix}
$$

::: skq.gates.qupent.single.QupentX
    options:
      show_root_heading: true
      show_source: true

### Z Gate (QupentZ)

The Z gate for a qupent applies different phases to each basis state, using the fifth roots of unity.

**Matrix Representation:**

$$
\text{QupentZ} = \begin{pmatrix}
1 & 0 & 0 & 0 & 0 \\
0 & e^{2\pi i/5} & 0 & 0 & 0 \\
0 & 0 & e^{4\pi i/5} & 0 & 0 \\
0 & 0 & 0 & e^{6\pi i/5} & 0 \\
0 & 0 & 0 & 0 & e^{8\pi i/5}
\end{pmatrix}
$$

Where $e^{2\pi i/5}$ is the fifth root of unity.

::: skq.gates.qupent.single.QupentZ
    options:
      show_root_heading: true
      show_source: true

### Hadamard Gate (QupentH)

The Hadamard gate for a qupent creates a superposition of the five basis states with different phases.

**Matrix Representation:**

$$
\text{QupentH} = \frac{1}{\sqrt{5}} \begin{pmatrix}
1 & 1 & 1 & 1 & 1 \\
1 & e^{2\pi i/5} & e^{4\pi i/5} & e^{6\pi i/5} & e^{8\pi i/5} \\
1 & e^{4\pi i/5} & e^{8\pi i/5} & e^{2\pi i/5} & e^{6\pi i/5} \\
1 & e^{6\pi i/5} & e^{2\pi i/5} & e^{8\pi i/5} & e^{4\pi i/5} \\
1 & e^{8\pi i/5} & e^{6\pi i/5} & e^{4\pi i/5} & e^{2\pi i/5}
\end{pmatrix}
$$

This is a generalized Fourier transform matrix for dimension 5.

::: skq.gates.qupent.single.QupentH
    options:
      show_root_heading: true
      show_source: true

### T Gate (QupentT)

The T gate for a qupent applies smaller phase shifts than the Z gate.

**Matrix Representation:**

$$
\text{QupentT} = \begin{pmatrix}
e^{0\pi i/10} & 0 & 0 & 0 & 0 \\
0 & e^{\pi i/10} & 0 & 0 & 0 \\
0 & 0 & e^{2\pi i/10} & 0 & 0 \\
0 & 0 & 0 & e^{3\pi i/10} & 0 \\
0 & 0 & 0 & 0 & e^{4\pi i/10}
\end{pmatrix}
$$

::: skq.gates.qupent.single.QupentT
    options:
      show_root_heading: true
      show_source: true 