# Getting Started with TLFEA

This guide will help you get up and running with TLFEA quickly.

## What is TLFEA?

TLFEA (Total Lagrangian Finite Element Analysis) is a GPU-accelerated FEA framework designed for simulating flexible multibody dynamics. This minimal implementation features:

- **FEAT10 Element**: 10-node tetrahedral element for 3D solid mechanics
- **SyncedNesterov Solver**: Fast iterative solver with Nesterov acceleration
- **SVK Material**: St. Venant-Kirchhoff hyperelastic material model
- **CUDA Acceleration**: Leverages GPU for fast computation

## Installation

### Step 1: Install Dependencies

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install cmake libeigen3-dev
```

**CUDA Toolkit:**
Download and install from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)

### Step 2: Clone and Build

```bash
# Clone the repository
git clone https://github.com/Ruochun/TLFEA.git
cd TLFEA

# Create build directory
mkdir build
cd build

# Configure and build
cmake ..
make -j$(nproc)
```

## Running Your First Simulation

The included example (`test_feat10_nesterov`) simulates a cantilever beam:

```bash
cd build
./bin/test_feat10_nesterov
```

### What the Example Does

1. Loads a cubic tetrahedral mesh from `data/meshes/T10/cube.1.node`
2. Fixes all nodes on the z=0 plane (creating a cantilever)
3. Applies a horizontal point load
4. Solves for equilibrium using the Nesterov solver
5. Prints the deformed positions

### Expected Output

You should see output like:
```
mesh read nodes: 27
mesh read elements: 48
gpu_t10_data created
gpu_t10_data initialized
Fixed nodes (z == 0):
0 1 2 3 4 5 6 7 8
...
x12:
0.00000000000000000 1.00000000000000000 ...
```

The solver iterates to find the equilibrium configuration under the applied load.

## Understanding the Code

### Main Components

```cpp
// 1. Load mesh
int n_nodes = ANCFCPUUtils::FEAT10_read_nodes("data/meshes/T10/cube.1.node", nodes);
int n_elems = ANCFCPUUtils::FEAT10_read_elements("data/meshes/T10/cube.1.ele", elements);

// 2. Create GPU data structure
GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);

// 3. Set up material properties
gpu_t10_data.SetSVK(E, nu);  // Young's modulus, Poisson's ratio
gpu_t10_data.SetDensity(rho0);

// 4. Create solver
SyncedNesterovSolver solver(&gpu_t10_data, n_constraints);

// 5. Solve
solver.Solve();
```

### Key Parameters

- `E = 7e8` Pa: Young's modulus (aluminum-like stiffness)
- `nu = 0.33`: Poisson's ratio
- `rho0 = 2700` kg/m³: Density (aluminum)
- `time_step = 1e-3` s: Time step for integration

## Next Steps

### Modify the Example

1. **Change material properties**: Edit `E`, `nu`, `rho0` in the example
2. **Apply different loads**: Modify the external force vector
3. **Use different meshes**: Create your own `.node` and `.ele` files
4. **Adjust solver parameters**: Tune convergence tolerances and iteration counts

### Create Your Own Mesh

Meshes are in Tetgen format:
- `.node` file: Node coordinates
- `.ele` file: Element connectivity

You can use [Tetgen](http://wias-berlin.de/software/tetgen/) to generate meshes:
```bash
tetgen -pq1.414a0.001 your_geometry.poly
```

### Add New Materials

To implement a new material model:
1. Create a new header in `src/materials/`
2. Implement the stress computation from deformation gradient
3. Include in `FEAT10DataFunc.cuh`

### Extend with More Elements

Future extensions could add:
- ANCF beam elements (for cables, ropes)
- ANCF shell elements (for thin structures)
- Other solid elements (hex, prism)

## Troubleshooting

### Build Fails with "nvcc not found"
- Make sure CUDA is installed: `which nvcc`
- Add CUDA to PATH: `export PATH=/usr/local/cuda/bin:$PATH`

### Runtime Error: "CUDA out of memory"
- Reduce mesh size
- Use a GPU with more memory
- Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU

### Solver Doesn't Converge
- Increase `max_inner` or `max_outer` iterations
- Adjust `inner_tol` or `outer_tol` tolerances
- Reduce the time step
- Check material parameters are physically reasonable

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md): Understand the system design
- [BUILDING.md](BUILDING.md): Detailed build instructions
- [README.md](../README.md): Project overview

## Getting Help

If you encounter issues:
1. Check the troubleshooting section above
2. Review the example code for correct usage patterns
3. Open an issue on GitHub with:
   - Your system configuration
   - Build/runtime error messages
   - Steps to reproduce the problem

## Contributing

Contributions are welcome! Consider:
- Adding new example programs
- Implementing additional material models
- Improving documentation
- Optimizing performance
- Adding unit tests

See the main README for the project structure and extension points.
