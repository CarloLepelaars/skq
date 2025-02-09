# skq

![](https://img.shields.io/pypi/dm/skq)
![Python Version](https://img.shields.io/badge/dynamic/toml?url=https://raw.githubusercontent.com/CarloLepelaars/skq/main/pyproject.toml&query=%24.project%5B%22requires-python%22%5D&label=python&color=blue) 
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)



Scientific Toolkit for Quantum Computing

This library is used in the [q4p (Quantum Computing for Programmers)](https://github.com/CarloLepelaars/q4p) course.

NOTE: This library is developed for educational purposes. While we strive for correctness of everything, the code is provided as is and not guaranteed to be bug-free. For sensitive applications make to check computations. 

## Why SKQ?

- Exploration: Play with fundamental quantum building blocks (NumPy).
- Education: Learn quantum computing concepts and algorithms.
- Integration: Combine classical components with quantum components.
- Democratize quantum for Python programmers and data scientists: Develop quantum algorithms in your favorite environment and easily export to your favorite quantum computing platform for running on real quantum hardware.

## Install

```bash
pip install skq
```

The default `skq` installation contains conversion to `qiskit`. PennyLane and PyQuil support can be installed as optional dependencies.

### All backends
```bash
pip install skq[all]
```

### PennyLane
```bash
pip install skq[pennylane]
```

### PyQuil
```bash
pip install skq[pyquil]
```
