/**
 * Beam Simulation
 *
 * Author: GitHub Copilot Agent
 *
 * This example demonstrates loading a mesh from a VTU file and performing
 * a realistic engineering analysis:
 * 1. Load a tetrahedral mesh (TET10 elements) from beam.vtu using MoPhiEssentials
 * 2. Apply boundary conditions (fix one end at x≈0, apply load at x≈10)
 * 3. Run a dynamic simulation using TLFEA's solvers
 * 4. Export the results to VTK files for visualization
 *
 * This simulates a cantilever beam with one end fixed and a concentrated load
 * applied to the free end, similar to a structural engineering test setup.
 */

#include <cuda_runtime.h>

#include <Eigen/Dense>
#include <MoPhiEssentials.h>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "elements/FEAT10Data.cuh"
#include "solvers/SyncedNesterov.cuh"
#include "utils/cpu_utils.h"
#include "utils/mesh_utils.h"
#include "utils/quadrature_utils.h"

// Material properties for aluminum
const double E = 7e10;     // Young's modulus: 7e10 Pa (70 GPa)
const double nu = 0.33;    // Poisson's ratio
const double rho0 = 2700;  // Density (kg/m³)

// Simulation parameters
const int N_TIMESTEPS = 10000;     // Number of timesteps to simulate
const int OUTPUT_FREQUENCY = 100;  // Output every N timesteps

int main() {
    std::cout << "=======================================================" << std::endl;
    std::cout << "  Beam Simulation" << std::endl;
    std::cout << "=======================================================" << std::endl;

    // ==========================================================================
    // Load mesh from VTU file using MoPhiEssentials
    // ==========================================================================

    std::string vtu_filename = "data/meshes/T10/beam.vtu";
    std::cout << "Loading mesh from: " << vtu_filename << std::endl;

    mophi::Mesh mesh;
    try {
        mesh = mophi::LoadVtu(vtu_filename);
    } catch (const std::exception& e) {
        std::cerr << "Error loading VTU file: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "Mesh loaded successfully:" << std::endl;
    std::cout << "  Nodes: " << mesh.NumLocalNodes() << std::endl;
    std::cout << "  TET4 elements: " << mesh.NumOwnedTets() << std::endl;
    std::cout << "  TET10 elements: " << mesh.NumOwnedTet10s() << std::endl;
    std::cout << "  HEX8 elements: " << mesh.NumOwnedHexes() << std::endl;

    // ==========================================================================
    // Extract mesh data for TLFEA
    // ==========================================================================

    // Check that we have tet10 elements (required for FEAT10)
    if (mesh.NumOwnedTet10s() == 0) {
        std::cerr << "Error: No TET10 elements found in '" << vtu_filename << "'." << std::endl;
        std::cerr << "Found: " << mesh.NumOwnedTets() << " TET4, " << mesh.NumOwnedTet10s() << " TET10, "
                  << mesh.NumOwnedHexes() << " HEX8 elements." << std::endl;
        std::cerr << "FEAT10 requires 10-node tetrahedral elements (VTK cell type 24)." << std::endl;
        std::cerr << "Please verify the mesh format. See examples/README_BEAM.md for requirements." << std::endl;
        return 1;
    }

    int n_nodes = mesh.NumLocalNodes();
    int n_elems = mesh.NumOwnedTet10s();

    // Convert mophi::Mesh geometry to Eigen format
    Eigen::MatrixXd nodes(n_nodes, 3);
    for (int i = 0; i < n_nodes; i++) {
        const auto& node = mesh.geom.nodes[i];
        nodes(i, 0) = node.x();
        nodes(i, 1) = node.y();
        nodes(i, 2) = node.z();
    }

    // Convert mophi::Mesh connectivity to Eigen format
    Eigen::MatrixXi elements(n_elems, 10);
    for (int i = 0; i < n_elems; i++) {
        const auto& elem = mesh.topo.tet10s[i];
        for (int j = 0; j < 10; j++) {
            elements(i, j) = elem[j];
        }
    }

    std::cout << "Converted mesh data to TLFEA format" << std::endl;

    // ==========================================================================
    // Initialize GPU data structure
    // ==========================================================================

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
    // Apply boundary conditions
    // ==========================================================================

    // Find the x-coordinate range to determine beam ends
    double x_min = h_x12.minCoeff();
    double x_max = h_x12.maxCoeff();
    double x_range = x_max - x_min;

    std::cout << "Beam x-coordinate range: [" << x_min << ", " << x_max << "]" << std::endl;

    // Fix nodes at x ≈ 0 (the left end of the beam)
    std::vector<int> fixed_node_indices;
    double fixed_tolerance = 0.1 * x_range;  // 10% of beam length
    for (int i = 0; i < h_x12.size(); ++i) {
        if (std::abs(h_x12(i) - x_min) < fixed_tolerance) {
            fixed_node_indices.push_back(i);
        }
    }

    Eigen::VectorXi h_fixed_nodes(fixed_node_indices.size());
    for (size_t i = 0; i < fixed_node_indices.size(); ++i) {
        h_fixed_nodes(i) = fixed_node_indices[i];
    }

    std::cout << "Boundary conditions: Fixed " << fixed_node_indices.size() << " nodes at x ≈ " << x_min << std::endl;
    gpu_t10_data.SetNodalFixed(h_fixed_nodes);

    // ==========================================================================
    // Apply external forces
    // ==========================================================================

    Eigen::VectorXd h_f_ext(gpu_t10_data.get_n_coef() * 3);
    h_f_ext.setZero();

    // Apply a concentrated load at x ≈ x_max (the right end of the beam)
    // This simulates a load applied to the free end of a cantilever beam
    std::vector<int> loaded_node_indices;
    double load_tolerance = 0.1 * x_range;  // 10% of beam length
    for (int i = 0; i < h_x12.size(); ++i) {
        if (std::abs(h_x12(i) - x_max) < load_tolerance) {
            loaded_node_indices.push_back(i);
        }
    }

    // Apply a downward force in the -z direction
    // Coordinate system: x=axial (beam length), y=width, z=height (positive up)
    double total_load = -1000.0;  // Total load in Newtons (negative = downward in z)
    double load_per_node = total_load / loaded_node_indices.size();

    for (int i : loaded_node_indices) {
        h_f_ext(3 * i + 2) = load_per_node;  // Apply force in z-direction (index 2)
    }

    gpu_t10_data.SetExternalForce(h_f_ext);
    std::cout << "External forces: Applied " << total_load << " N load to " << loaded_node_indices.size()
              << " nodes at x ≈ " << x_max << std::endl;

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

    std::cout << "Material properties: E=" << (E / 1e9) << " GPa, nu=" << nu << ", rho=" << rho0 << " kg/m³"
              << std::endl;

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
    SyncedNesterovParams params = {1.0e-8, 1e14, 1.0e-6, 1.0e-6, 5, 300, 1.0e-4};

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
        // Continue simulation even if output fails - computation is still valuable
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
