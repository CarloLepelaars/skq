# Global Phase Gates API Reference

This page documents the global phase gates available in the `skq.gates.global_phase` module.

## GlobalPhase

Global phase gates apply a phase shift to the entire quantum state.

**Matrix Representation:**

$$
\text{GlobalPhase}(\phi) = e^{i\phi} \cdot I
$$

Where $I$ is the identity matrix of appropriate dimension.

::: skq.gates.global_phase.GlobalPhase
    options:
      show_root_heading: true
      show_source: true

## Predefined Phase Gates

SKQ provides several predefined global phase gates:

### Identity (No Phase Shift)

**Matrix Representation:**

$$
\text{Identity} = e^{i \cdot 0} \cdot I = I
$$

::: skq.gates.global_phase.Identity
    options:
      show_root_heading: true
      show_source: true

### QuarterPhase (π/2)

**Matrix Representation:**

$$
\text{QuarterPhase} = e^{i\pi/2} \cdot I = i \cdot I
$$

::: skq.gates.global_phase.QuarterPhase
    options:
      show_root_heading: true
      show_source: true

### HalfPhase (π)

**Matrix Representation:**

$$
\text{HalfPhase} = e^{i\pi} \cdot I = -1 \cdot I
$$

::: skq.gates.global_phase.HalfPhase
    options:
      show_root_heading: true
      show_source: true

### ThreeQuarterPhase (3π/2)

**Matrix Representation:**

$$
\text{ThreeQuarterPhase} = e^{i3\pi/2} \cdot I = -i \cdot I
$$

::: skq.gates.global_phase.ThreeQuarterPhase
    options:
      show_root_heading: true
      show_source: true

### FullPhase (2π)

**Matrix Representation:**

$$
\text{FullPhase} = e^{i2\pi} \cdot I = I
$$

::: skq.gates.global_phase.FullPhase
    options:
      show_root_heading: true
      show_source: true