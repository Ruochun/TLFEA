# Cantilever Beam Simulation Example

This example demonstrates a realistic use case of the TLFEA framework, simulating a cantilever beam under gravity loading.

## Overview

The simulation performs the following steps:

1. **Load Mesh**: Reads a tetrahedral mesh (FEAT10 10-node elements) from `data/meshes/T10/cube.1.node` and `data/meshes/T10/cube.1.ele`
2. **Apply Boundary Conditions**: Fixes all nodes at the z=0 plane (one end of the beam)
3. **Apply Loading**: Applies gravity force in the negative z-direction to simulate the beam's own weight
4. **Run Simulation**: Executes 100 timesteps using the SyncedNesterov solver
5. **Export Results**: Saves the deformed mesh to VTK files every 10 timesteps

## Material Properties

The simulation uses material properties for Aluminum:
- Young's Modulus (E): 7×10¹⁰ Pa
- Poisson's Ratio (ν): 0.33
- Density (ρ): 2700 kg/m³

## Building

```bash
cd build
cmake ..
make cantilever_beam_simulation
```

## Running

From the build directory:

```bash
./bin/cantilever_beam_simulation
```

## Output Files

The simulation generates VTK files with the naming pattern `output_beam_#####.vtk` where ##### is the timestep number (00000, 00010, 00020, etc.).

These files contain:
- Deformed node positions
- Displacement vectors (relative to initial configuration)
- Element connectivity

## Visualization

The output VTK files can be visualized using ParaView or other VTK-compatible visualization tools:

### Using ParaView

1. Open ParaView
2. File → Open → Select all `output_beam_*.vtk` files
3. Click "Apply" in the Properties panel
4. Use the animation controls to see the beam deflection over time
5. Optionally, add a "Warp By Vector" filter using the "displacement" field to see the deformation more clearly

### Command Line (if ParaView is installed)

```bash
paraview output_beam_*.vtk
```

## Simulation Parameters

You can modify the simulation by editing `examples/cantilever_beam_simulation.cc`:

- `N_TIMESTEPS`: Number of simulation steps (default: 100)
- `OUTPUT_FREQUENCY`: How often to save VTK files (default: every 10 steps)
- Material properties (E, nu, rho0)
- Solver parameters (timestep h, tolerances, etc.)

## Expected Results

The cantilever beam should exhibit:
- Deflection in the negative z-direction (downward) due to gravity
- Maximum deflection at the free end (z=1 plane)
- Zero deflection at the fixed end (z=0 plane)
- Gradual increase in deflection over time until equilibrium is reached
