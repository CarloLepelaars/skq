# Ququart Gates API Reference

This page documents the ququart gates available in the `skq.gates.ququart` module. Ququarts are quantum systems with 4 basis states (|0⟩, |1⟩, |2⟩, |3⟩) and can model spin-3/2 particles.

## Ququart Gate Base Class

The `QuquartGate` class serves as the foundation for all ququart-based quantum gates in SKQ.

::: skq.gates.ququart.base.QuquartGate
    options:
      show_root_heading: true
      show_source: true

## Single-Ququart Gates

### Identity Gate (QuquartI)

The Identity gate leaves the ququart state unchanged.

**Matrix Representation:**

$$
\text{QuquartI} = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

::: skq.gates.ququart.single.QuquartI
    options:
      show_root_heading: true
      show_source: true

### X Gate (QuquartX)

The X gate for a ququart performs a cyclic permutation of the basis states: |0⟩ → |1⟩ → |2⟩ → |3⟩ → |0⟩.

**Matrix Representation:**

$$
\text{QuquartX} = \begin{pmatrix}
0 & 0 & 0 & 1 \\
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & 0
\end{pmatrix}
$$

::: skq.gates.ququart.single.QuquartX
    options:
      show_root_heading: true
      show_source: true

### Z Gate (QuquartZ)

The Z gate for a ququart applies different phases to each basis state.

**Matrix Representation:**

$$
\text{QuquartZ} = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & i & 0 & 0 \\
0 & 0 & -1 & 0 \\
0 & 0 & 0 & -i
\end{pmatrix}
$$

::: skq.gates.ququart.single.QuquartZ
    options:
      show_root_heading: true
      show_source: true

### Hadamard Gate (QuquartH)

The Hadamard gate for a ququart creates a superposition of the four basis states with different phases.

**Matrix Representation:**

$$
\text{QuquartH} = \frac{1}{2} \begin{pmatrix}
1 & 1 & 1 & 1 \\
1 & i & -1 & -i \\
1 & -1 & 1 & -1 \\
1 & -i & -1 & i
\end{pmatrix}
$$

::: skq.gates.ququart.single.QuquartH
    options:
      show_root_heading: true
      show_source: true

### T Gate (QuquartT)

The T gate for a ququart applies a phase shift to the |1⟩ state.

**Matrix Representation:**

$$
\text{QuquartT} = \begin{pmatrix}
1 & 0 & 0 & 0 \\
0 & e^{i\pi/4} & 0 & 0 \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{pmatrix}
$$

::: skq.gates.ququart.single.QuquartT
    options:
      show_root_heading: true
      show_source: true 