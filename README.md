# TLFEA

Total Lagrangian Finite Element Analysis - A minimal FEA framework for flexible multibody dynamics.

## Overview

This project provides a starting point for Total Lagrangian finite element analysis, featuring:
- **One element type**: FEAT10 (10-node tetrahedral element)
- **One solver**: SyncedNesterov (iterative gradient-based solver)
- **CMake build system**: Easy to build and extend

Based on the [Total-Lagrangian-FEA](https://github.com/uwsbel/Total-Lagrangian-FEA) project from SBEL at UW-Madison.

## Quick Start

### Requirements

- CMake 3.18 or higher
- CUDA Toolkit 11.0 or higher
- Eigen3 library
- C++17 compatible compiler
- MoPhiEssentials (included as submodule)

### Building

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/Ruochun/TLFEA.git
cd TLFEA

# Or if already cloned, initialize submodules
git submodule update --init --recursive

# Build
mkdir build
cd build
cmake ..
make
```

See [docs/BUILDING.md](docs/BUILDING.md) for detailed build instructions.

### Running the Examples

#### Basic Infrastructure Test

```bash
cd build
./bin/test_feat10_nesterov
```

This example tests the FEAT10 element and SyncedNesterov solver infrastructure.

#### Cantilever Beam Simulation (Realistic Use Case)

```bash
cd build
./bin/cantilever_beam_simulation
```

This example demonstrates a realistic physics simulation:
- Loads a tetrahedral mesh representing a cantilever beam
- Applies boundary conditions (fixed end)
- Simulates the beam deflecting under gravity
- Exports results to VTK files for visualization in ParaView

See [examples/README_CANTILEVER.md](examples/README_CANTILEVER.md) for detailed information about this example.

## Project Structure

```
TLFEA/
├── src/
│   ├── elements/      # FEA element implementations (FEAT10)
│   ├── solvers/       # Iterative solvers (SyncedNesterov)
│   ├── materials/     # Material models (St. Venant-Kirchhoff)
│   └── utils/         # Utility functions
├── external/
│   └── MoPhiEssentials/  # Low-level GPU infrastructure (submodule)
├── examples/          # Example programs
├── data/              # Test data and meshes
├── docs/              # Documentation
└── CMakeLists.txt     # Build configuration
```

## Dependencies

### MoPhiEssentials

This project uses [MoPhiEssentials](https://github.com/Ruochun/MoPhiEssentials) as a submodule for:
- **GPU Error Handling**: `MOPHI_GPU_CALL` macro for CUDA error checking
- **Logging System**: Thread-safe logging with `MOPHI_INFO`, `MOPHI_WARNING`, `MOPHI_ERROR`
- **Common Utilities**: Shared GPU/CPU data structures and memory management

The integration provides consistent error handling and logging across the codebase.

## Documentation

- [Building Guide](docs/BUILDING.md) - Detailed build and installation instructions
- [Architecture](docs/ARCHITECTURE.md) - System design and component overview

## Features

### Implemented
- ✅ FEAT10 (10-node tetrahedral) element
- ✅ SyncedNesterov iterative solver
- ✅ St. Venant-Kirchhoff hyperelastic material
- ✅ GPU acceleration via CUDA
- ✅ CMake build system
- ✅ Basic infrastructure test example
- ✅ Realistic cantilever beam simulation example with VTK output
- ✅ MoPhiEssentials integration for error handling and logging

### Future Extensions
- Additional element types (ANCF beam/shell elements)
- More material models (Neo-Hookean, Mooney-Rivlin)
- Additional solvers (Newton-Raphson, VBD)
- Collision detection and contact handling
- Advanced examples and benchmarks

## License

See LICENSE file for details.

## Acknowledgments

This project is based on the [Total-Lagrangian-FEA](https://github.com/uwsbel/Total-Lagrangian-FEA) research code developed by the Simulation-Based Engineering Laboratory (SBEL) at the University of Wisconsin-Madison.

