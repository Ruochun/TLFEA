# TLFEA

Total Lagrangian Finite Element Analysis - A minimal FEA framework for flexible multibody dynamics.

## Overview

This project provides a starting point for Total Lagrangian finite element analysis, featuring:
- **One element type**: FEAT10 (10-node tetrahedral element)
- **One solver**: SyncedNesterov (iterative gradient-based solver)
- **CMake build system**: Easy to build and extend

Based on the [Total-Lagrangian-FEA](https://github.com/uwsbel/Total-Lagrangian-FEA) project from SBEL at UW-Madison.

## Requirements

- CMake 3.18 or higher
- CUDA Toolkit 11.0 or higher
- Eigen3 library
- C++17 compatible compiler

## Building

```bash
mkdir build
cd build
cmake ..
make
```

## Running the Example

```bash
cd build
./bin/test_feat10_nesterov
```

This example simulates a cantilever beam with gravity loading using the FEAT10 element and SyncedNesterov solver.

## Project Structure

```
TLFEA/
├── src/
│   ├── elements/      # FEA element implementations (FEAT10)
│   ├── solvers/       # Iterative solvers (SyncedNesterov)
│   ├── materials/     # Material models (St. Venant-Kirchhoff)
│   └── utils/         # Utility functions
├── examples/          # Example programs
├── data/              # Test data and meshes
└── CMakeLists.txt     # Build configuration
```

## License

See LICENSE file for details.
