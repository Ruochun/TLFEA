/**
 * Cantilever Beam Simulation with Physics
 *
 * Author: GitHub Copilot Agent
 *
 * This example demonstrates a realistic use case of the TLFEA framework:
 * 1. Load a tetrahedral mesh (FEAT10 elements) representing a cantilever beam
 * 2. Apply boundary conditions (fix one end of the beam)
 * 3. Apply a realistic load (gravity acting on the entire beam)
 * 4. Run a dynamic simulation over multiple timesteps
 * 5. Export the results to VTK files that can be visualized in ParaView or similar tools
 *
 * This simulates a cantilever beam fixed at one end (z=0 plane) with gravity
 * acting downward (-z direction). The beam deflects over time under its own weight.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "elements/FEAT10Data.cuh"
#include "solvers/SyncedNesterov.cuh"
#include "utils/cpu_utils.h"
#include "utils/mesh_utils.h"
#include "utils/quadrature_utils.h"

// Material properties for aluminum
const double E = 7e8;      // Young's modulus (Pa) - Aluminum
const double nu = 0.33;    // Poisson's ratio - Aluminum
const double rho0 = 2700;  // Density (kg/m³) - Aluminum

// Simulation parameters
const int N_TIMESTEPS = 100;      // Number of timesteps to simulate
const int OUTPUT_FREQUENCY = 10;  // Output every N timesteps

int main() {
    std::cout << "=======================================================" << std::endl;
    std::cout << "  Cantilever Beam Simulation with Physics" << std::endl;
    std::cout << "=======================================================" << std::endl;

    // Read mesh data
    Eigen::MatrixXd nodes;
    Eigen::MatrixXi elements;

    int n_nodes = ANCFCPUUtils::FEAT10_read_nodes("data/meshes/T10/cube.1.node", nodes);
    int n_elems = ANCFCPUUtils::FEAT10_read_elements("data/meshes/T10/cube.1.ele", elements);

    std::cout << "Mesh loaded: " << n_nodes << " nodes, " << n_elems << " elements" << std::endl;

    // Initialize GPU data structure
    GPU_FEAT10_Data gpu_t10_data(n_elems, n_nodes);
    gpu_t10_data.Initialize();

    // Extract coordinate vectors from nodes matrix
    Eigen::VectorXd h_x12(n_nodes), h_y12(n_nodes), h_z12(n_nodes);
    for (int i = 0; i < n_nodes; i++) {
        h_x12(i) = nodes(i, 0);  // X coordinates
        h_y12(i) = nodes(i, 1);  // Y coordinates
        h_z12(i) = nodes(i, 2);  // Z coordinates
    }

    // ==========================================================================
    // Apply boundary conditions: Fix nodes at z=0 (one end of the beam)
    // ==========================================================================

    std::vector<int> fixed_node_indices;
    for (int i = 0; i < h_z12.size(); ++i) {
        if (std::abs(h_z12(i)) < 1e-8) {  // tolerance for floating point
            fixed_node_indices.push_back(i);
        }
    }

    Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
    for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
        h_fixed_nodes(i) = fixed_node_indices[i];
    }

    std::cout << "Boundary conditions: Fixed " << fixed_node_indices.size() << " nodes at z=0" << std::endl;
    gpu_t10_data.SetNodalFixed(h_fixed_nodes);

    // ==========================================================================
    // Apply external forces: Gravity acting on the entire beam
    // ==========================================================================

    Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
    h_f_ext.setZero();

    // Apply gravity force (F = m * g) to all nodes
    // Note: For a more accurate simulation, the mass should be computed from the
    // actual mesh volume and distributed properly to nodes. Here we use a simple
    // approximation assuming the mesh represents a 1m × 1m × 1m cube.
    double gravity = -9.81;    // m/s² (negative because downward)
    double mesh_volume = 1.0;  // m³ (approximate volume of unit cube mesh)
    double total_mass = rho0 * mesh_volume;
    double mass_per_node = total_mass / n_nodes;  // Simple uniform distribution

    for (int i = 0; i < n_nodes; i++) {
        h_f_ext(3 * i + 2) = mass_per_node * gravity;  // Apply force in z-direction
    }

    gpu_t10_data.SetExternalForce(h_f_ext);
    std::cout << "External forces: Applied gravity (g = " << gravity << " m/s²)" << std::endl;

    // ==========================================================================
    // Setup material and element properties
    // ==========================================================================

    // Get quadrature data
    const Eigen::VectorXd& tet5pt_x_host = Quadrature::tet5pt_x;
    const Eigen::VectorXd& tet5pt_y_host = Quadrature::tet5pt_y;
    const Eigen::VectorXd& tet5pt_z_host = Quadrature::tet5pt_z;
    const Eigen::VectorXd& tet5pt_weights_host = Quadrature::tet5pt_weights;

    // Setup element data
    gpu_t10_data.Setup(tet5pt_x_host, tet5pt_y_host, tet5pt_z_host, tet5pt_weights_host, h_x12, h_y12, h_z12, elements);

    gpu_t10_data.SetDensity(rho0);
    gpu_t10_data.SetDamping(0.0, 0.1);  // Add some damping for stability
    gpu_t10_data.SetSVK(E, nu);         // St. Venant-Kirchhoff material model

    std::cout << "Material properties: E=" << std::scientific << E << std::defaultfloat << " Pa, nu=" << nu
              << ", rho=" << static_cast<int>(rho0) << " kg/m³" << std::endl;

    // ==========================================================================
    // Compute reference configuration data
    // ==========================================================================

    gpu_t10_data.CalcDnDuPre();
    gpu_t10_data.CalcMassMatrix();
    gpu_t10_data.CalcConstraintData();
    gpu_t10_data.ConvertToCSR_ConstraintJacT();

    std::cout << "Reference configuration computed" << std::endl;

    // ==========================================================================
    // Setup solver
    // ==========================================================================

    // alpha, rho, inner_tol, outer_tol, max_outer, max_inner, time_step
    SyncedNesterovParams params = {1.0e-8, 1e14, 1.0e-6, 1.0e-6, 5, 300, 1.0e-3};

    SyncedNesterovSolver solver(&gpu_t10_data, gpu_t10_data.get_n_constraint());
    solver.Setup();
    solver.SetParameters(&params);

    std::cout << "Solver initialized: SyncedNesterov with h=" << std::scientific << params.time_step
              << std::defaultfloat << std::endl;

    // ==========================================================================
    // Run simulation and output results
    // ==========================================================================

    std::cout << "Starting simulation: " << N_TIMESTEPS << " timesteps, output every " << OUTPUT_FREQUENCY << " steps"
              << std::endl;

    // Output initial configuration
    Eigen::VectorXd x12, y12, z12;
    gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);

    std::stringstream ss;
    ss << "output_beam_" << std::setfill('0') << std::setw(5) << 0 << ".vtk";
    bool success = ANCFCPUUtils::WriteFEAT10ToVTK(ss.str(), nodes, elements, x12, y12, z12);
    if (success) {
        std::cout << "Saved initial state to " << ss.str() << std::endl;
    } else {
        std::cerr << "Failed to write initial VTK file" << std::endl;
    }

    // Run simulation loop
    for (int step = 1; step <= N_TIMESTEPS; step++) {
        solver.Solve();

        // Output at specified frequency
        if (step % OUTPUT_FREQUENCY == 0) {
            gpu_t10_data.RetrievePositionToCPU(x12, y12, z12);

            // Create filename with timestep
            std::stringstream filename;
            filename << "output_beam_" << std::setfill('0') << std::setw(5) << step << ".vtk";

            bool write_success = ANCFCPUUtils::WriteFEAT10ToVTK(filename.str(), nodes, elements, x12, y12, z12);
            if (write_success) {
                std::cout << "Step " << step << "/" << N_TIMESTEPS << ": Saved to " << filename.str() << std::endl;
            } else {
                std::cerr << "Step " << step << "/" << N_TIMESTEPS << ": Failed to write VTK file" << std::endl;
            }
        }
    }

    // ==========================================================================
    // Cleanup
    // ==========================================================================

    gpu_t10_data.Destroy();

    std::cout << "=======================================================" << std::endl;
    std::cout << "  Simulation Complete!" << std::endl;
    std::cout << "  Output files: output_beam_*.vtk" << std::endl;
    std::cout << "  Visualize with: paraview output_beam_*.vtk" << std::endl;
    std::cout << "=======================================================" << std::endl;

    return 0;
}
