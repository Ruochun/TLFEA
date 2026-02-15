# Beam VTU Demo

This example demonstrates loading a tetrahedral mesh from a VTU file and performing realistic engineering analysis using TLFEA's solvers.

## Overview

The `beam_vtu_demo` program showcases the following workflow:

1. **Load mesh from VTU file** - Uses MoPhiEssentials' `mophi::LoadVtu()` function to load the mesh from `data/meshes/T10/beam.vtu`
2. **Store in mophi::Mesh** - The loaded mesh data is stored in a `mophi::Mesh` object
3. **Convert to TLFEA format** - Converts the 4-node tetrahedral elements to 10-node FEAT10 elements by generating mid-edge nodes
4. **Apply boundary conditions** - Fixes all nodes at the left end (where x ≈ 0) to simulate a fixed support
5. **Apply loads** - Applies a downward force at the right end (where x ≈ 10) to simulate a load
6. **Solve with TLFEA** - Uses the SyncedNesterov solver to compute the beam's response
7. **Save results** - Outputs deformed configurations to VTK files for visualization in ParaView

## Physical Setup

This simulation models a cantilever beam with:
- **Fixed end**: Left side (x ≈ 0) - all degrees of freedom constrained
- **Free end**: Right side (x ≈ 10) - subjected to a downward force
- **Material**: Aluminum (E = 700 MPa, ν = 0.33, ρ = 2700 kg/m³)
- **Load**: 1000 N downward force distributed over nodes at the free end

## Input Files

- `data/meshes/T10/beam.vtu` - VTU mesh file containing:
  - 81 nodes
  - 24 tetrahedral elements (type 10)
  - Beam geometry spanning approximately from x=0 to x=10

## Output Files

The simulation generates VTK files at regular intervals:
- `output_beam_vtu_00000.vtk` - Initial configuration
- `output_beam_vtu_00010.vtk` - After 10 timesteps
- `output_beam_vtu_00020.vtk` - After 20 timesteps
- ...
- `output_beam_vtu_00100.vtk` - Final configuration

Each VTK file contains:
- Deformed node positions
- Displacement vectors (for visualization)
- Full mesh connectivity

## Building and Running

### Prerequisites
- CUDA Toolkit 11.0 or higher
- Eigen3 library
- CMake 3.18 or higher

### Build
```bash
cd build
cmake ..
make beam_vtu_demo
```

### Run
```bash
cd build
./bin/beam_vtu_demo
```

The program will:
1. Load the mesh from `data/meshes/T10/beam.vtu`
2. Display mesh statistics
3. Run 100 timesteps of simulation
4. Output VTK files every 10 timesteps
5. Display progress messages

Expected output:
```
=======================================================
  Beam VTU Demo - Realistic Engineering Analysis
=======================================================
Loading mesh from: data/meshes/T10/beam.vtu
Mesh loaded successfully
Number of nodes: 81
Number of owned nodes: 81
Number of tets: 24
Number of hexes: 0
Converting mesh to TLFEA format...
Nodes: 81, Elements: 24
Total nodes with mid-edge nodes: 225
Mesh bounds: x ∈ [0, 10]
Boundary conditions: Fixed XX nodes at x ≈ 0
External forces: Applied -1000 N at XX nodes at x ≈ 10
Material properties: E=7.000000e+08 Pa, nu=0.33, rho=2700 kg/m³
Reference configuration computed
Solver initialized: SyncedNesterov with h=1.000000e-03
Starting simulation: 100 timesteps, output every 10 steps
Saved initial state to output_beam_vtu_00000.vtk
...
Simulation Complete!
```

## Visualization

To visualize the results in ParaView:

```bash
paraview output_beam_vtu_*.vtk
```

In ParaView:
1. Open all the VTK files as a sequence
2. Apply the "Warp by Vector" filter using the "displacement" field to visualize deformation
3. Use the time slider to animate the simulation
4. Color by displacement magnitude to see stress patterns

## Key Features

### VTU Mesh Loading
The demo uses MoPhiEssentials' VTU loader which provides:
- Robust XML parsing using pugixml
- Support for ASCII format VTU files
- Automatic handling of node ownership and halo information
- Conversion to efficient internal mesh representation

### FEAT10 Element Conversion
Since VTU files typically store linear tetrahedral elements (4 nodes), the demo automatically:
- Generates mid-edge nodes for each element edge
- Creates a shared mid-edge node map to avoid duplicates
- Converts 4-node tets to 10-node FEAT10 elements
- Updates the node coordinate matrix accordingly

### Boundary Conditions
The demo demonstrates how to:
- Identify boundary nodes by spatial location (x ≈ 0)
- Apply fixed constraints using `SetNodalFixed()`
- Apply distributed loads using `SetExternalForce()`

## Comparison to Cantilever Beam Simulation

This demo is similar to `cantilever_beam_simulation.cc` but differs in:

| Feature | cantilever_beam_simulation | beam_vtu_demo |
|---------|---------------------------|---------------|
| Mesh source | Text files (.node, .ele) | VTU file |
| Mesh loader | Custom FEAT10 reader | MoPhiEssentials VTU loader |
| Mesh storage | Direct Eigen matrices | mophi::Mesh class |
| Element type | Pre-generated FEAT10 | Convert from linear tets |
| Load type | Gravity on all nodes | Point load at free end |
| Mesh size | Simple cube | Realistic beam geometry |

## Troubleshooting

**Issue**: Error loading mesh
- Check that `data/meshes/T10/beam.vtu` exists
- Verify the file is a valid VTU format
- Ensure MoPhiEssentials is properly built

**Issue**: Solver fails to converge
- Try reducing the time step (increase `params.time_step`)
- Increase damping (second parameter in `SetDamping()`)
- Check that boundary conditions are properly applied

**Issue**: No output files generated
- Verify write permissions in the current directory
- Check console for error messages about file writing

## Further Development

Potential extensions to this demo:
- Load different mesh formats (e.g., Exodus, GMSH)
- Apply more complex boundary conditions (e.g., time-varying loads)
- Implement material models (e.g., neo-Hookean, Mooney-Rivlin)
- Add contact detection and resolution
- Parallelize across multiple GPUs using mesh partitioning
