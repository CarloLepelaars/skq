# Constants API Reference

This page documents the constants available in the `skq.constants` module. These constants are used throughout the SKQ framework.

## Physical Constants

### PLANCK_CONSTANT
Planck constant in J·s (joule-seconds).

Value: 6.62607015 × 10^-34 J·s

```python
from skq.constants import PLANCK_CONSTANT

print(PLANCK_CONSTANT)  # 6.62607015e-34
```

### REDUCED_PLANCK_CONSTANT
Reduced Planck constant (ħ) in J·s (joule-seconds).

Value: 1.054571817 × 10^-34 J·s

```python
from skq.constants import REDUCED_PLANCK_CONSTANT

print(REDUCED_PLANCK_CONSTANT)  # 1.054571817e-34
```

### BOLTZMANN_CONSTANT
Boltzmann constant in J/K (joules per kelvin).

Value: 1.380649 × 10^-23 J/K

```python
from skq.constants import BOLTZMANN_CONSTANT

print(BOLTZMANN_CONSTANT)  # 1.380649e-23
```

### SPEED_OF_LIGHT
Speed of light in vacuum in m/s (meters per second).

Value: 2.99792458 × 10^8 m/s

```python
from skq.constants import SPEED_OF_LIGHT

print(SPEED_OF_LIGHT)  # 299792458.0
```

### ELECTRON_CHARGE
Elementary charge (charge of an electron) in C (coulombs).

Value: 1.602176634 × 10^-19 C

```python
from skq.constants import ELECTRON_CHARGE

print(ELECTRON_CHARGE)  # 1.602176634e-19
```

### PERMEABILITY_OF_FREE_SPACE
Permeability of free space (vacuum permeability) in N/A² (newtons per ampere squared).

Value: 4π × 10^-7 N/A²

```python
from skq.constants import PERMEABILITY_OF_FREE_SPACE
import numpy as np

print(PERMEABILITY_OF_FREE_SPACE)  # 1.2566370614359173e-06 (4π × 10^-7)
print(PERMEABILITY_OF_FREE_SPACE == 4 * np.pi * 1e-7)  # True
```

### PERMITTIVITY_OF_FREE_SPACE
Permittivity of free space (vacuum permittivity) in F/m (farads per meter).

Value: 8.854187817 × 10^-12 F/m

```python
from skq.constants import PERMITTIVITY_OF_FREE_SPACE

print(PERMITTIVITY_OF_FREE_SPACE)  # 8.854187817e-12
```

::: skq.constants
    options:
      show_root_heading: true
      show_source: true 