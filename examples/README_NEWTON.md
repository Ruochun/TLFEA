# Newton Solver Examples

This directory contains examples demonstrating the use of the SyncedNewton solver, a GPU-accelerated Newton method for finite element analysis.

## Overview

The SyncedNewton solver is a fully synchronized Newton method without line search that computes the full gradient and Hessian per iteration. It uses cuDSS for efficient sparse linear system solving.

## Prerequisites

**Important**: The SyncedNewton solver requires the cuDSS library, which is available in CUDA Toolkit 12.4+. If cuDSS is not available, the examples will not compile.

To check if cuDSS is available:
```bash
# Check CUDA version
nvcc --version

# Look for libcudss in your CUDA installation
find /usr/local/cuda* -name "libcudss*"
```

## Examples

### 1. test_feat10_newton

A simple test case that demonstrates the SyncedNewton solver on a small FEAT10 cube mesh.

**Features:**
- Loads a cube mesh with FEAT10 (10-node tetrahedral) elements
- Fixes nodes on the z=0 plane
- Applies a point load
- Runs 50 timesteps using the Newton solver

**Build:**
```bash
cd build
cmake ..
make test_feat10_newton
```

**Run:**
```bash
./bin/test_feat10_newton
```

### 2. beam_simulation_newton

A comprehensive beam deflection simulation using the SyncedNewton solver.

**Features:**
- Loads a beam mesh from VTU file
- Applies realistic boundary conditions (fixed end, loaded end)
- Runs a dynamic simulation
- Outputs VTK files for visualization

**Build:**
```bash
cd build
cmake ..
make beam_simulation_newton
```

**Run:**
```bash
./bin/beam_simulation_newton
```

**Output:**
The simulation generates VTK files (`output_beam_newton_#####.vtk`) that can be visualized in ParaView.

## Solver Parameters

The SyncedNewton solver uses the following parameters:

```cpp
struct SyncedNewtonParams {
  Real inner_atol;     // Absolute tolerance for inner (Newton) iterations
  Real inner_rtol;     // Relative tolerance for inner iterations
  Real outer_tol;      // Tolerance for outer (constraint) iterations
  Real rho;            // Augmented Lagrangian penalty parameter
  int max_outer;       // Maximum outer iterations
  int max_inner;       // Maximum inner (Newton) iterations
  Real time_step;      // Timestep size
};
```

**Example configuration:**
```cpp
SyncedNewtonParams params = {
  1.0e-6,   // inner_atol
  1.0e-6,   // inner_rtol
  1.0e-6,   // outer_tol
  1e14,     // rho
  5,        // max_outer
  30,       // max_inner
  1.0e-4    // time_step
};
```

## Comparison with Nesterov Solver

The repository also includes examples using the SyncedNesterov solver:
- `test_feat10_nesterov.cc` - Nesterov solver on cube mesh
- `beam_simulation.cc` - Beam simulation with Nesterov solver

### When to use Newton vs Nesterov:

**Newton Solver:**
- Better for stiff problems requiring tight convergence
- Quadratic convergence near the solution
- Requires more computation per iteration (Hessian assembly)
- Requires cuDSS library

**Nesterov Solver:**
- Better for smooth, non-stiff problems
- First-order method (linear convergence)
- Lower cost per iteration
- No external library dependencies beyond CUDA

## Technical Notes

### Implementation Details

The SyncedNewton solver:
1. Assembles the sparse Hessian matrix using GPU kernels
2. Uses cuDSS for sparse Cholesky factorization
3. Solves the Newton system to get the velocity update
4. Updates positions and constraints
5. Checks convergence based on gradient norm and constraint violations

### cuDSS Integration

cuDSS (CUDA Direct Sparse Solver) provides GPU-accelerated sparse linear system solving:
- Supports CSR (Compressed Sparse Row) format
- Efficient sparse Cholesky factorization
- Refactorization support for repeated solves with same sparsity pattern
- Iterative refinement for improved accuracy

### Memory Requirements

The Newton solver requires more GPU memory than Nesterov due to:
- Sparse Hessian storage (CSR format)
- cuDSS internal buffers
- Additional work vectors for linear solve

For large meshes, ensure sufficient GPU memory is available.

## Troubleshooting

### cuDSS not found

If you see:
```
cuDSS library not found. SyncedNewton solver will not be available.
```

Solutions:
1. Install CUDA Toolkit 12.4 or later
2. Ensure cuDSS is included in your CUDA installation
3. Update CMake to point to the correct CUDA installation

### Compilation errors

If you encounter compilation errors related to cuDSS:
1. Check that `cudss.h` is in your CUDA include path
2. Verify `libcudss.so` is in your CUDA library path
3. Update the CMakeLists.txt to point to the correct paths if needed

### Runtime errors

If the solver fails at runtime:
1. Check that the mesh is valid (positive volumes)
2. Verify boundary conditions are properly set
3. Try adjusting solver parameters (tolerances, max iterations)
4. Check GPU memory usage (use `nvidia-smi`)

## References

- Original implementation: https://github.com/uwsbel/Total-Lagrangian-FEA
- cuDSS documentation: https://docs.nvidia.com/cuda/cudss/
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
