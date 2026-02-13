# TLFEA Architecture

## Overview

TLFEA (Total Lagrangian Finite Element Analysis) is a minimal FEA framework designed for flexible multibody dynamics simulation using GPU acceleration.

## Design Philosophy

This implementation follows a **minimal starting point** approach:
- **ONE element type**: FEAT10 (10-node tetrahedral element)
- **ONE solver**: SyncedNesterov (iterative gradient-based solver)
- **Clean architecture**: Modular design for easy extension

## Directory Structure

```
TLFEA/
├── src/
│   ├── elements/       # Finite element implementations
│   │   ├── FEAT10Data.cu/cuh          # FEAT10 element data structure and GPU kernels
│   │   ├── FEAT10DataFunc.cuh         # Device functions for FEAT10 computations
│   │   ├── ElementBase.h              # Abstract base class for all elements
│   │   └── ANCF*.cuh                  # Stub headers (for future extension)
│   │
│   ├── solvers/        # Iterative solvers
│   │   ├── SyncedNesterov.cu/cuh      # Nesterov accelerated gradient solver
│   │   └── SolverBase.h               # Abstract base class for solvers
│   │
│   ├── materials/      # Material models
│   │   ├── MaterialModel.cuh          # Material model interface
│   │   ├── SVK.cuh                    # St. Venant-Kirchhoff hyperelastic model
│   │   └── MooneyRivlin.cuh           # Stub (for future extension)
│   │
│   └── utils/          # Utility functions
│       ├── cpu_utils.h/cc             # CPU-side utilities (mesh I/O, etc.)
│       ├── cuda_utils.h               # CUDA helper macros and functions
│       ├── quadrature_utils.h         # Gauss quadrature rules
│       └── mesh_utils.h/cc            # Mesh manipulation utilities
│
├── examples/           # Example programs
│   └── test_feat10_nesterov.cc        # Beam deflection demo
│
├── data/               # Test data
│   └── meshes/T10/                    # Tetrahedral mesh files
│
└── docs/               # Documentation
```

## Key Components

### 1. Elements (`src/elements/`)

**FEAT10Data**: Manages the 10-node tetrahedral element
- Stores node positions, velocities, and material properties
- Computes element mass matrix, internal forces, and stiffness
- Supports GPU-accelerated computation using CUDA

**ElementBase**: Abstract interface that all element types must implement
- Provides virtual methods for mass matrix, internal force, constraints
- Enables polymorphic solver design

### 2. Solvers (`src/solvers/`)

**SyncedNesterov**: Nesterov accelerated gradient descent solver
- First-order momentum-based optimization
- Fully GPU-synchronized for performance
- Handles constraints via Lagrange multipliers
- Suitable for quasi-static and dynamic simulations

**Key Features**:
- Accelerated convergence compared to vanilla gradient descent
- Adaptive time stepping
- Constraint projection for boundary conditions

### 3. Materials (`src/materials/`)

**SVK (St. Venant-Kirchhoff)**: Hyperelastic material model
- Suitable for moderate deformations
- Simple constitutive law: `P = F * (2μE + λtr(E)I)`
- Where E is the Green-Lagrange strain tensor

**MaterialModel**: Interface for constitutive models
- Computes stress from deformation gradient
- Extensible to other models (Neo-Hookean, Mooney-Rivlin, etc.)

### 4. Utilities (`src/utils/`)

**cpu_utils**: CPU-side helpers
- Mesh file I/O (reads .node and .ele files)
- Data conversion between Eigen and CUDA formats

**cuda_utils**: CUDA macros
- Error checking wrappers
- Device memory management helpers

**quadrature_utils**: Numerical integration
- Gauss quadrature points and weights
- Supports various quadrature orders

## Computational Pipeline

1. **Mesh Loading**: Read tetrahedral mesh from disk
2. **Initialization**: Transfer mesh to GPU, allocate buffers
3. **Preprocessing**: Compute reference configuration quantities (shape function gradients, etc.)
4. **Mass Matrix Assembly**: Build element and global mass matrices
5. **Time Integration Loop**:
   - Compute internal forces from current deformation
   - Compute constraint forces
   - Update velocities and positions using Nesterov solver
   - Check convergence
6. **Postprocessing**: Extract results (positions, stresses) back to CPU

## GPU Acceleration

TLFEA uses CUDA for GPU acceleration:
- **Element-level parallelism**: Each CUDA thread processes one element or DOF
- **Cooperative groups**: Synchronization within thread blocks for iterative solvers
- **Memory coalescing**: Structured access patterns for optimal bandwidth

## Extension Points

To add new functionality:

1. **New Element Type**: Inherit from `ElementBase`, implement virtual methods
2. **New Solver**: Inherit from `SolverBase`, implement `Solve()` method
3. **New Material**: Implement material model interface, add to element stress computation
4. **New Constraints**: Extend constraint handling in element data structures

## References

- Based on [Total-Lagrangian-FEA](https://github.com/uwsbel/Total-Lagrangian-FEA) by SBEL, UW-Madison
- St. Venant-Kirchhoff model: Classical continuum mechanics
- Nesterov acceleration: Nesterov, Y. (1983). "A method for solving the convex programming problem with convergence rate O(1/k²)"
